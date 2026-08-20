"""
Collect 10y daily OHLCV for every commodity used in the project, from
TradingView's public (anonymous, unauthenticated) chart data feed.

TradingView doesn't expose a plain REST endpoint for historical bars; the
charting UI itself pulls them over a websocket protocol. This replays that
protocol directly (session handshake -> resolve_symbol -> create_series),
same as open-source tools like tvdatafeed do.

Output: data-new/commodities-data/<name>.csv
"""
from __future__ import annotations

import json
import random
import string
import time
from datetime import UTC, datetime

import pandas as pd
import websocket

from data_new_common import DATA_NEW, clean_ohlcv, clip_last_n_years, get_logger

logger = get_logger("collect_commodities")

OUT_DIR = DATA_NEW / "commodities-data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# name -> TradingView symbol, chosen for longest available free daily history
SYMBOLS = {
    "gold": "OANDA:XAUUSD",
    "silver": "TVC:SILVER",
    "copper": "COMEX:HG1!",
    "crude_oil": "TVC:USOIL",
    "natural_gas": "NYMEX:NG1!",
    "wheat": "CBOT:ZW1!",
}

# A bar opening at or after this UTC hour belongs to the *next* trading day.
# See _bars_to_df for why this is evening and not midday.
EVENING_OPEN_HOUR_UTC = 17

RETURN_THRESHOLD = 0.25
YEARS = 10
N_BARS = 6000  # comfortably covers 10y of daily bars plus margin
WS_URL = "wss://data.tradingview.com/socket.io/websocket"


def _session(prefix: str) -> str:
    return prefix + "_" + "".join(random.choice(string.ascii_lowercase) for _ in range(12))


def _send(ws, func: str, params: list) -> None:
    msg = json.dumps({"m": func, "p": params}, separators=(",", ":"))
    ws.send("~m~" + str(len(msg)) + "~m~" + msg)


def fetch_bars(symbol: str, n_bars: int = N_BARS, timeout: float = 20.0, attempts: int = 4) -> pd.DataFrame:
    """
    Fetch daily bars, retrying the whole handshake on a transport failure.

    TradingView's socket refuses or times out often enough that a single-shot
    fetch loses a symbol or two on most runs - observed live: gold and copper
    both timed out while the other four succeeded in the same pass. Under a
    scheduled collector that silently leaves those two a day stale, so each
    symbol gets several attempts before it is called a failure.
    """
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            df = _fetch_bars_once(symbol, n_bars, timeout)
            if not df.empty:
                return df
            last_err = RuntimeError("no bars returned")
        except Exception as e:  # noqa: BLE001 - any transport error is retryable
            last_err = e
        if attempt < attempts:
            logger.warning(f"  -> {symbol}: attempt {attempt}/{attempts} failed ({last_err}); retrying")
            time.sleep(2.0 * attempt)
    raise RuntimeError(f"all {attempts} attempts failed: {last_err}")


def _fetch_bars_once(symbol: str, n_bars: int, timeout: float) -> pd.DataFrame:
    ws = websocket.create_connection(WS_URL, header=["Origin: https://www.tradingview.com"], timeout=timeout)
    try:
        session = _session("cs")
        _send(ws, "set_auth_token", ["unauthorized_user_token"])
        _send(ws, "chart_create_session", [session, ""])
        _send(ws, "resolve_symbol", [session, "symbol_1", "=" + json.dumps({"symbol": symbol, "adjustment": "splits"})])
        _send(ws, "create_series", [session, "s1", "s1", "symbol_1", "1D", n_bars, ""])

        bars: dict[float, list] = {}
        start = time.time()
        while time.time() - start < timeout:
            try:
                raw = ws.recv()
            except Exception:
                break
            for part in raw.split("~m~"):
                part = part.strip()
                if not part.startswith("{"):
                    continue
                try:
                    msg = json.loads(part)
                except ValueError:
                    continue
                if msg.get("m") in ("timescale_update", "du"):
                    p = msg.get("p", [])
                    if len(p) > 1 and isinstance(p[1], dict):
                        s1 = p[1].get("s1", {})
                        for row in s1.get("s", []):
                            v = row["v"]
                            bars[v[0]] = v
                if msg.get("m") == "series_completed":
                    return _bars_to_df(bars)
        return _bars_to_df(bars)
    finally:
        try:
            ws.close()
        except Exception:
            pass


def _bars_to_df(bars: dict[float, list]) -> pd.DataFrame:
    """
    TradingView's daily bar timestamp is the bar's *open*, not a trading-day
    label. For instruments whose electronic session opens in the evening UTC
    (OANDA spot FX/metals, COMEX/NYMEX/TVC futures - typically 21:00-23:00
    UTC), the bar spans almost entirely into the next UTC calendar day, and
    that's the date TradingView's own chart displays it under - e.g. a bar
    opening 2026-08-16T21:00 UTC is shown on tradingview.com as "Aug 17".
    Naively truncating the open timestamp to a date understates every such
    bar's date by one day, so those roll forward.

    The threshold is EVENING (>=17:00 UTC), not midday. An earlier version
    used >=12:00, which silently corrupted CBOT wheat. ZW1! normally opens
    its overnight session at 00:00/01:00 UTC (no roll, correct), but on the
    shortened session the day after a holiday - July 5th, the day after
    Thanksgiving, Dec 26, Jan 2 - CBOT runs the day session only, opening
    13:30/14:30 UTC. Those are same-day bars that TradingView dates to that
    same calendar day, yet >=12:00 rolled them a day forward. 14 bars over
    10y were misdated; 7 of them landed on a date that already held a real
    bar and overwrote it via the duplicate drop below (e.g. 2025-01-03's
    true close of 529.25 was replaced by 545.75, the Jan 2 half-day close),
    and 7 landed on a Saturday. Wheat also has genuine Sunday-evening opens
    at 21:00/22:00 UTC which must still roll to Monday - >=17:00 keeps those
    while leaving the half-day sessions alone.

    Verified against every symbol in SYMBOLS over 6000 bars: gold, silver,
    copper, crude_oil and natural_gas open only at 21:00-23:00 UTC, so this
    threshold leaves their dates byte-identical to the old rule.
    """
    if not bars:
        return pd.DataFrame()
    rows = sorted(bars.values(), key=lambda v: v[0])
    df = pd.DataFrame(rows, columns=["Time", "Open", "High", "Low", "Close", "Volume"])
    opened = pd.to_datetime(df["Time"], unit="s")
    rolls_to_next_day = opened.dt.hour >= EVENING_OPEN_HOUR_UTC
    df["Date"] = opened.dt.normalize() + pd.to_timedelta(rolls_to_next_day.astype(int), unit="D")
    df = df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _drop_forming_bar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the newest bar if it's today's (UTC) - these symbols trade almost
    continuously, so the bar the feed hands back for the current calendar day
    is still accumulating range at the moment we ask for it: its Open and
    running High may be final, but its Low and Close are whatever the price
    happens to be right now, not the day's actual settle. Compared directly
    against TradingView's own finalized bar hours later, that difference was
    large enough to matter (e.g. silver's Close off by more than a dollar).
    A bar dated strictly before today has had its full 24h window elapse and
    is safe to treat as closed.
    """
    if df.empty:
        return df
    today = pd.Timestamp(datetime.now(UTC).date())
    if df.index[-1] >= today:
        return df.iloc[:-1]
    return df


def run(names=None):
    items = {n: SYMBOLS[n] for n in names} if names else SYMBOLS
    ok, failed = [], []
    for name, symbol in items.items():
        logger.info(f"Fetching {name} ({symbol}) from TradingView...")
        try:
            df = fetch_bars(symbol)
            if df.empty:
                logger.warning(f"  -> No data returned for {name}")
                failed.append(name)
                continue

            df, removed = clean_ohlcv(df, return_threshold=RETURN_THRESHOLD)
            df = clip_last_n_years(df, YEARS)
            df = _drop_forming_bar(df)

            if df.empty:
                failed.append(name)
                continue

            out_path = OUT_DIR / f"{name}.csv"
            df.to_csv(out_path)
            logger.info(f"  -> Saved {name}: {len(df)} rows ({df.index.min().date()} -> {df.index.max().date()}), removed {removed} bad rows")
            ok.append(name)
        except Exception as e:
            logger.error(f"  -> FAILED {name} ({symbol}): {e}")
            failed.append(name)
        time.sleep(1.0)

    logger.info("=" * 60)
    logger.info(f"Commodities collection complete: {len(ok)} ok, {len(failed)} failed")
    if failed:
        logger.warning(f"Failed: {failed}")
    return len(failed)


if __name__ == "__main__":
    import sys
    # Exit non-zero on any failure. run() used to return None and exit 0 even
    # when symbols failed, so a scheduled run reported success while those
    # assets quietly went stale.
    raise SystemExit(1 if run(sys.argv[1:] or None) else 0)
