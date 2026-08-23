"""
Collect 10y daily OHLCV for every crypto asset used in the project, from
Binance's public spot klines API (no auth required).

Source: https://api.binance.com/api/v3/klines
Output: data-new/crypto-data/<TICKER>.csv
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_new_common import (
    DATA_NEW,
    clean_ohlcv,
    clip_last_n_years,
    get_logger,
    merge_incremental,
    read_existing_csv,
    trim_tail,
)
from tradingview_fetch import fetch_bars as fetch_tv_bars

logger = get_logger("collect_crypto")

OUT_DIR = DATA_NEW / "crypto-data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = [
    "BTC", "ETH", "SOL", "BNB", "AVAX", "ADA", "NEAR", "SUI", "APT", "TRX",
    "LINK", "AAVE", "UNI", "CRV", "LDO", "PENDLE", "SNX",
    "TAO", "FET", "RNDR", "FIL", "GRT", "INJ", "IMX",
    "PEPE", "WIF",

]

# Binance renamed/relisted a few of these since the tickers were curated.
SYMBOL_OVERRIDES = {
    "RNDR": "RENDERUSDT",
}

RETURN_THRESHOLD = 0.60  # crypto is volatile; this only catches bad prints
YEARS = 10
KLINES_URL = "https://api.binance.com/api/v3/klines"


def binance_symbol(ticker: str) -> str:
    return SYMBOL_OVERRIDES.get(ticker, f"{ticker}USDT")


def _get_with_retry(url: str, params: dict, retries: int = 4):
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=20)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            if isinstance(data, dict) and "code" in data and data.get("code") != 0:
                raise RuntimeError(f"Binance API error: {data.get('msg')}")
            if isinstance(data, dict) and "msg" in data:
                raise RuntimeError(f"Binance API error: {data.get('msg')}")
            return data
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise last_err


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    out = []
    cursor = start_ms
    while cursor < end_ms:
        batch = _get_with_retry(
            KLINES_URL,
            {
                "symbol": symbol,
                "interval": "1d",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
        )
        if not batch:
            break
        out.extend(batch)
        last_open = batch[-1][0]
        if last_open <= cursor:
            break
        cursor = last_open + 1
        if len(batch) < 1000:
            break
        time.sleep(0.2)
    return out


def _drop_forming_bar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binance's klines endpoint happily returns the current UTC day's candle
    while it's still accumulating - crypto trades 24/7, so unlike an exchange
    with fixed hours there's never a point where "today" is guaranteed
    closed. That row's Close is just whatever the price is right now, not a
    settled value.
    """
    if df.empty:
        return df
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    if df.index[-1] >= today:
        return df.iloc[:-1]
    return df


def run(tickers=None):
    tickers = tickers or TICKERS
    end = datetime.now(timezone.utc)
    default_start = end - timedelta(days=365 * YEARS + 10)
    end_ms = int(end.timestamp() * 1000)
    default_start_ms = int(default_start.timestamp() * 1000)

    ok, failed = [], []
    for ticker in tickers:
        symbol = binance_symbol(ticker)
        logger.info(f"Fetching {ticker} ({symbol}) from Binance...")
        try:
            out_path = OUT_DIR / f"{ticker}.csv"
            existing = read_existing_csv(out_path)
            trimmed = trim_tail(existing, 10)

            if not trimmed.empty:
                cutoff_date = trimmed.index.max()
                fetch_start_ms = int(cutoff_date.timestamp() * 1000)
            else:
                fetch_start_ms = default_start_ms

            new_df = pd.DataFrame()
            try:
                klines = fetch_klines(symbol, fetch_start_ms, end_ms)
                if klines:
                    df_k = pd.DataFrame(klines, columns=[
                        "OpenTime", "Open", "High", "Low", "Close", "Volume",
                        "CloseTime", "QuoteVolume", "Trades", "TakerBaseVol", "TakerQuoteVol", "Ignore",
                    ])
                    df_k["Date"] = pd.to_datetime(df_k["OpenTime"], unit="ms").dt.normalize()
                    for c in ["Open", "High", "Low", "Close", "Volume"]:
                        df_k[c] = pd.to_numeric(df_k[c], errors="coerce")
                    new_df = df_k[["Date", "Open", "High", "Low", "Close", "Volume"]].set_index("Date")
                    new_df = new_df[~new_df.index.duplicated(keep="last")].sort_index()
            except Exception as binance_err:
                logger.warning(f"  -> Binance API failed for {ticker} ({binance_err}); trying TradingView fallback...")
                tv_symbol = f"BINANCE:{symbol}"
                n_bars = 100 if not trimmed.empty else 6000
                new_df = fetch_tv_bars(tv_symbol, n_bars=n_bars)

            if new_df.empty and trimmed.empty:
                logger.warning(f"  -> No data returned for {symbol}")
                failed.append(ticker)
                continue

            merged = merge_incremental(trimmed, new_df)
            df, removed = clean_ohlcv(merged, return_threshold=RETURN_THRESHOLD)
            df = clip_last_n_years(df, YEARS)
            df = _drop_forming_bar(df)

            if df.empty:
                logger.warning(f"  -> {ticker}: nothing left after cleaning")
                failed.append(ticker)
                continue

            df.to_csv(out_path)
            logger.info(f"  -> Saved {ticker}: {len(df)} rows ({df.index.min().date()} -> {df.index.max().date()}), removed {removed} bad rows")
            ok.append(ticker)
        except Exception as e:
            logger.error(f"  -> FAILED {ticker} ({symbol}): {e}")
            failed.append(ticker)
        time.sleep(0.3)

    logger.info("=" * 60)
    logger.info(f"Crypto collection complete: {len(ok)} ok, {len(failed)} failed")
    if failed:
        logger.warning(f"Failed tickers: {failed}")
    return len(failed)




if __name__ == "__main__":
    import sys
    # Non-zero exit on any failure so a scheduled run does not report success
    # while some tickers quietly went stale.
    raise SystemExit(1 if run(sys.argv[1:] or None) else 0)
