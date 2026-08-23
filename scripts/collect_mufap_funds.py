"""
Collect 10y daily NAV history for every mutual fund used in the project,
straight from MUFAP's own site (mufap.com.pk) - the only public source for
Pakistani open-end fund NAV history.

Two calls per fund lineage:
  1. GET  /FundProfile/FundDirectory       - discover FundID(s) per fund name
  2. POST /AMC/GetFundDetailbyAMCByDate    - full NAV history for one FundID

A handful of "umbrella" fund names (mostly Voluntary Pension Scheme funds)
publish 2-4 FundIDs under the identical fund name, one per sub-category
(e.g. VPS-Debt / VPS-Equity / VPS-Money Market). Collapsing those into a
single series - which is what the previous dataset did - interleaves
unrelated NAV series and is a likely source of the "data discrepancies"
this collection is meant to fix. Instead each sub-category is written out
as its own file, named "<Fund> (<Category>).csv".

Output: data-new/mufap-data/<Fund>[ (Category)].csv
"""
from __future__ import annotations

import html
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_new_common import (
    DATA_NEW,
    clean_price_series,
    clip_last_n_years,
    drop_unclosed_sessions,
    get_logger,
    is_already_current,
    merge_incremental,
    read_existing_csv,
    safe_filename,
    trim_tail,
)

logger = get_logger("collect_mufap")

OUT_DIR = DATA_NEW / "mufap-data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RETURN_THRESHOLD = 0.50  # NAV should be stable day to day; catches obvious glitches
MARKET_TZ = "Asia/Karachi"  # MUFAP publishes NAV dates in local time
SESSION_CLOSE = "19:00"
YEARS = 10

BASE = "https://www.mufap.com.pk"
DIRECTORY_URL = f"{BASE}/FundProfile/FundDirectory"
DETAIL_URL = f"{BASE}/AMC/GetFundDetailbyAMCByDate"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": DIRECTORY_URL,
}

CARD_RE = re.compile(r'card-title">([^<]*)<[\s\S]{0,4000}?FundID=(\d+)')


def target_fund_names() -> list[str]:
    """Fund universe = all existing funds in data-new/mufap-data or all discovered in MUFAP."""
    if OUT_DIR.exists():
        names = []
        for p in OUT_DIR.glob("*.csv"):
            stem = p.stem
            clean = re.sub(r"\s*\([^)]*\)", "", stem).strip()
            names.append(clean)
        if names:
            return sorted(set(names))
    return []



def discover_fund_ids(retries: int = 3) -> dict[str, list[str]]:
    """
    Map fund name -> FundID(s) from the MUFAP directory.
    """
    resp = None
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(DIRECTORY_URL, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            break
        except Exception as e:  # noqa: BLE001 - any transport error is retryable
            last_err = e
            if attempt < retries:
                logger.warning(f"Fund directory attempt {attempt}/{retries} failed ({type(e).__name__}); retrying")
                time.sleep(1.0 * attempt)
    if resp is None:
        raise RuntimeError(f"fund directory unreachable after {retries} attempts: {last_err}")
    mapping: dict[str, list[str]] = {}
    for raw_name, fund_id in CARD_RE.findall(resp.text):
        name = html.unescape(re.sub(r"\s+", " ", raw_name)).strip()
        mapping.setdefault(name, []).append(fund_id)
    return mapping


def fetch_fund_history(fund_id: str, retries: int = 2) -> tuple[str, pd.DataFrame]:
    last_err = None
    data = None
    for attempt in range(retries):
        try:
            resp = requests.post(
                DETAIL_URL,
                headers={**HEADERS, "Content-Type": "application/json;charset=utf-8",
                         "Accept": "application/json, text/plain, */*", "Origin": BASE,
                         "Referer": f"{BASE}/FundProfile/FundDetail?FundID={fund_id}"},
                data=json.dumps({"FundID": str(fund_id), "Date": "2026-08-01"}),
                timeout=10,
            )
            resp.raise_for_status()
            outer = resp.json()
            data = json.loads(outer["data"])
            if data.get("Table1"):
                break
            last_err = RuntimeError("empty Table1")
        except Exception as e:
            last_err = e
        time.sleep(1.0 * (attempt + 1))
    if data is None:
        raise last_err

    table1 = data.get("Table1") or []
    category = ""
    table0 = data.get("Table") or []
    if table0:
        category = (table0[0].get("Cat_Desc") or "").strip()

    if not table1:
        return category, pd.DataFrame()

    df = pd.DataFrame(table1)
    df["Date"] = pd.to_datetime(df["entryDate"], errors="coerce")
    df["NAV"] = pd.to_numeric(df["netval"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")[["NAV"]].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return category, df


def run():
    targets = target_fund_names()
    logger.info(f"Target fund universe: {len(targets)} funds")

    logger.info("Discovering FundIDs from MUFAP directory...")
    id_map = discover_fund_ids()
    logger.info(f"Directory returned {len(id_map)} distinct fund names")

    ok, failed = [], []
    for name in targets:
        fund_ids = id_map.get(name)
        if not fund_ids:
            logger.warning(f"  -> No FundID found for '{name}'")
            failed.append(name)
            continue

        for fund_id in fund_ids:
            try:
                category, new_df = fetch_fund_history(fund_id)

                label = f"{name} ({category})" if len(fund_ids) > 1 and category else name
                out_path = OUT_DIR / f"{safe_filename(label)}.csv"
                existing = read_existing_csv(out_path)

                if is_already_current(existing, tz=MARKET_TZ, session_close=SESSION_CLOSE):
                    logger.info(f"  -> '{label}': already up to date ({existing.index.max().date()})")
                    ok.append(label)
                    continue

                trimmed = trim_tail(existing, 10)

                if new_df.empty and trimmed.empty:
                    logger.warning(f"  -> {name} [{fund_id}]: no NAV history returned")
                    failed.append(f"{name} [{fund_id}]")
                    continue

                merged = merge_incremental(trimmed, new_df)
                df, removed = clean_price_series(merged, "NAV", return_threshold=RETURN_THRESHOLD, adjust_stock_splits=True)
                df = clip_last_n_years(df, YEARS)
                df, _ = drop_unclosed_sessions(
                    df, tz=MARKET_TZ, session_close=SESSION_CLOSE
                )
                if df.empty:
                    failed.append(f"{name} [{fund_id}]")
                    continue

                df.to_csv(out_path)
                logger.info(f"  -> Saved '{label}': {len(df)} rows ({df.index.min().date()} -> {df.index.max().date()}), removed {removed} bad rows")
                ok.append(label)
            except Exception as e:
                logger.error(f"  -> FAILED {name} [{fund_id}]: {e}")
                failed.append(f"{name} [{fund_id}]")
            time.sleep(0.3)

    logger.info("=" * 60)
    logger.info(f"MUFAP collection complete: {len(ok)} files saved, {len(failed)} failed")
    if failed:
        logger.warning(f"Failed: {failed}")
    return len(failed)




if __name__ == "__main__":
    # Non-zero exit on any failure so a scheduled run does not report success
    # while some funds quietly went stale.
    raise SystemExit(1 if run() else 0)
