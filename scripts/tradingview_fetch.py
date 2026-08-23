"""
Shared TradingView websocket fetch module for commodities data and macro proxies.
"""
from __future__ import annotations

import json
import logging
import random
import string
import time

import pandas as pd
import websocket

logger = logging.getLogger("tradingview_fetch")

WS_URL = "wss://data.tradingview.com/socket.io/websocket"
EVENING_OPEN_HOUR_UTC = 17


def _session(prefix: str) -> str:
    return prefix + "_" + "".join(random.choice(string.ascii_lowercase) for _ in range(12))


def _send(ws, func: str, params: list) -> None:
    msg = json.dumps({"m": func, "p": params}, separators=(",", ":"))
    ws.send("~m~" + str(len(msg)) + "~m~" + msg)


def _bars_to_df(bars: dict[float, list]) -> pd.DataFrame:
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


def fetch_bars(symbol: str, n_bars: int = 6000, timeout: float = 20.0, attempts: int = 4) -> pd.DataFrame:
    """
    Fetch daily bars from TradingView, retrying on failure.
    """
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            df = _fetch_bars_once(symbol, n_bars, timeout)
            if not df.empty:
                return df
            last_err = RuntimeError("no bars returned")
        except Exception as e:
            last_err = e
        if attempt < attempts:
            logger.warning(f"  -> {symbol}: attempt {attempt}/{attempts} failed ({last_err}); retrying")
            time.sleep(2.0 * attempt)
    raise RuntimeError(f"all {attempts} attempts failed for {symbol}: {last_err}")
