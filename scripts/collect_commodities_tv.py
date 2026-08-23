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
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import websocket

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_new_common import (
    DATA_NEW,
    clean_ohlcv,
    clip_last_n_years,
    get_logger,
    is_already_current,
    merge_incremental,
    read_existing_csv,
    trim_tail,
)
from tradingview_fetch import fetch_bars

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

RETURN_THRESHOLD = 0.25
YEARS = 10
N_BARS = 6000  # comfortably covers 10y of daily bars plus margin


def _drop_forming_bar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the newest bar if it's today's (UTC) - these symbols trade almost
    continuously, so the bar the feed hands back for the current calendar day
    is still accumulating range at the moment we ask for it: its Open and
    running High may be final, but its Low and Close are whatever the price
    happens to be right now, not the day's actual settle.
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
        try:
            out_path = OUT_DIR / f"{name}.csv"
            existing = read_existing_csv(out_path)

            if is_already_current(existing, tz="UTC"):
                logger.info(f"  -> {name}: already up to date ({existing.index.max().date()})")
                ok.append(name)
                continue

            logger.info(f"Fetching {name} ({symbol}) from TradingView...")
            trimmed = trim_tail(existing, 10)
            n_bars = 100 if not trimmed.empty else N_BARS

            new_df = fetch_bars(symbol, n_bars=n_bars)
            if new_df.empty and trimmed.empty:
                logger.warning(f"  -> No data returned for {name}")
                failed.append(name)
                continue

            merged = merge_incremental(trimmed, new_df)
            df, removed = clean_ohlcv(merged, return_threshold=RETURN_THRESHOLD)
            df = clip_last_n_years(df, YEARS)
            df = _drop_forming_bar(df)

            if df.empty:
                failed.append(name)
                continue

            df.to_csv(out_path)
            logger.info(f"  -> Saved {name}: {len(df)} rows ({df.index.min().date()} -> {df.index.max().date()}), removed {removed} bad rows")
            ok.append(name)
        except Exception as e:
            logger.error(f"  -> FAILED {name} ({symbol}): {e}")
            failed.append(name)
        time.sleep(0.5)

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
