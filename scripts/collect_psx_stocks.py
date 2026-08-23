"""
Collect daily OHLCV for every PSX-listed stock used in the project, straight
from PSX's own Data Portal Services (dps.psx.com.pk) - no third-party
aggregator involved.

Source: POST https://dps.psx.com.pk/historical (returns each symbol's full
available history, which happens to be ~10 years on PSX's site).
Output: data-new/psx-data/<TICKER>.csv
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_new_common import (

    DATA_NEW,
    clean_ohlcv,
    clip_last_n_years,
    drop_unclosed_sessions,
    get_logger,
    merge_incremental,
    read_existing_csv,
    trim_tail,
)

logger = get_logger("collect_psx")

OUT_DIR = DATA_NEW / "psx-data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = [
    "ABL", "ABOT", "AGP", "AHCL", "AICL", "AIRLINK", "AKBL", "APL", "ATLH", "ATRL",
    "BAFL", "BAHL", "BNWM", "BOP", "BWCL", "CHCC", "CNERGY", "COLG", "CPHL", "DGKC",
    "EFERT", "ENGROH", "FABL", "FATIMA", "FCCL", "FFC", "FFL", "GADT", "GAL", "GHGL",
    "GHNI", "GLAXO", "HALEON", "HBL", "HCAR", "HINOON", "HMB", "HUBC", "HUMNL", "IBFL",
    "ILP", "INDU", "INIL", "ISL", "JDWS", "JVDC", "KAPCO", "KEL", "KOHC", "KTML",
    "LCI", "LOTCHEM", "LUCK", "MARI", "MCB", "MEBL", "MEHT", "MLCF", "MTL", "MUREB",
    "NATF", "NBP", "NESTLE", "NETSOL", "NML", "OGDC", "PABC", "PAEL", "PAKT", "PGLC",
    "PIBTL", "PIOC", "PKGS", "POL", "POWER", "PPL", "PSEL", "PSO", "PSX", "PTC",
    "RMPL", "SAZEW", "SCBPL", "SEARL", "SHFA", "SNGP", "SRVI", "SSGC", "SSOM", "SYS",
    "TGL", "THALL", "TRG", "UBL", "UNITY", "UPFL", "YOUW",
]

RETURN_THRESHOLD = 0.30  # PSX circuit breakers are 5-7.5%; 30% only catches glitches
YEARS = 10
HISTORICAL_URL = "https://dps.psx.com.pk/historical"
MARKET_TZ = "Asia/Karachi"  # PSX publishes dates in local time
SESSION_CLOSE = "15:30"  # regular board close; after this today's bar is final

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://dps.psx.com.pk/historical",
    "Origin": "https://dps.psx.com.pk",
}

ROW_RE = re.compile(
    r"<tr>\s*<td[^>]*>([^<]+)</td>\s*<td[^>]*>([^<]+)</td>\s*<td[^>]*>([^<]+)</td>"
    r"\s*<td[^>]*>([^<]+)</td>\s*<td[^>]*>([^<]+)</td>\s*<td[^>]*>([^<]+)</td>\s*</tr>"
)


def parse_table(html: str) -> pd.DataFrame:
    rows = ROW_RE.findall(html)
    if not rows:
        return pd.DataFrame()
    records = []
    for date_s, o, h, l, c, v in rows:
        try:
            date = pd.to_datetime(date_s.strip(), format="%b %d, %Y")
        except ValueError:
            continue
        records.append({
            "Date": date,
            "Open": float(o.replace(",", "")),
            "High": float(h.replace(",", "")),
            "Low": float(l.replace(",", "")),
            "Close": float(c.replace(",", "")),
            "Volume": float(v.replace(",", "")),
        })
    df = pd.DataFrame(records).set_index("Date").sort_index()
    return df


def fetch_symbol(symbol: str, retries: int = 4) -> pd.DataFrame:
    """
    POST one symbol's history, retrying on transport failure.
    """
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                HISTORICAL_URL,
                headers=HEADERS,
                data={"symbol": symbol},
                timeout=30,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}")
            return parse_table(resp.text)
        except Exception as e:  # noqa: BLE001 - any transport error is retryable
            last_err = e
            if attempt < retries:
                logger.warning(f"  -> {symbol}: attempt {attempt}/{retries} failed ({type(e).__name__}); retrying")
                time.sleep(2.0 * attempt)
    raise RuntimeError(f"all {retries} attempts failed: {last_err}")


def run():
    ok, failed = [], []
    for ticker in TICKERS:
        logger.info(f"Fetching {ticker} from PSX DPS...")
        try:
            out_path = OUT_DIR / f"{ticker}.csv"
            existing = read_existing_csv(out_path)
            trimmed = trim_tail(existing, 10)

            new_df = fetch_symbol(ticker)
            if new_df.empty and trimmed.empty:
                logger.warning(f"  -> No data returned for {ticker}")
                failed.append(ticker)
                continue

            merged = merge_incremental(trimmed, new_df)
            df, removed = clean_ohlcv(merged, return_threshold=RETURN_THRESHOLD, adjust_stock_splits=True)
            df = clip_last_n_years(df, YEARS)
            df, unclosed = drop_unclosed_sessions(
                df, tz=MARKET_TZ, session_close=SESSION_CLOSE
            )
            if unclosed:
                logger.info(f"  -> {ticker}: dropped {unclosed} unclosed session(s)")

            if len(df) < 50:
                logger.warning(f"  -> {ticker}: too little data after cleaning ({len(df)} rows)")
                failed.append(ticker)
                continue

            df.to_csv(out_path)
            logger.info(f"  -> Saved {ticker}: {len(df)} rows ({df.index.min().date()} -> {df.index.max().date()}), removed {removed} bad rows")
            ok.append(ticker)
        except Exception as e:
            logger.error(f"  -> FAILED {ticker}: {e}")
            failed.append(ticker)
        time.sleep(0.6)

    logger.info("=" * 60)
    logger.info(f"PSX collection complete: {len(ok)} ok, {len(failed)} failed")
    if failed:
        logger.warning(f"Failed tickers: {failed}")
    return len(failed)



if __name__ == "__main__":
    # Non-zero exit on any failure so a scheduled run does not report success
    # while some tickers quietly went stale.
    raise SystemExit(1 if run() else 0)
