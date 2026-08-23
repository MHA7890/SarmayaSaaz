"""
Collect dated news headlines per asset, for the chart's catalyst markers.

Output: data-new/news/<asset_class>/<ticker>.csv
        columns: Date, Headline, Url, Source, MarketWide

Why this is collected to disk rather than fetched per request
-------------------------------------------------------------
The API can fetch headlines live, and still does for the current day, but two
things make live-only fetching unusable as the chart's source of truth:

  latency  - a useful marker spread needs several windowed queries per asset;
             doing that inside a page request makes the chart wait seconds.
  history  - TradingView's endpoint returns a rolling recent window (BTC came
             back spanning four days), so on a 30D or 90D chart every marker
             bunches against the right edge.

Both sources are also flaky enough to matter: a plain query for
"Systems Limited" PSX returned nothing on one attempt and 73 items on the
next. Collecting to disk on a schedule turns that into a retry-and-move-on
problem instead of an empty chart.

Sources
-------
TradingView headlines  - per-instrument, good for crypto and commodities,
                         patchy for PSX (OGDC 55, LUCK 3, HBL and ENGRO none).
Google News RSS        - keyless, dated, linked, and reaches years back. Issued
                         once plainly and once per quarter-window so markers
                         spread across the chart instead of clustering at
                         today.

Mutual funds
------------
No source carries fund-level headlines: "ABL Income Fund", "Al Meezan Mutual
Fund", "NBP Stock Fund" and "AL Habib Money Market Fund" each return zero
items. News about what *moves* a fund's NAV is abundant, so funds are
collected as a small number of shared feeds instead of one per fund - a rate
pool (SBP policy, T-bill auctions, CPI) and one file per asset management
company. Equity and balanced funds reuse the PSX index feed already collected
for the stock charts. Nine requests cover all 77 funds; one per fund would be
462 and would return nothing.
"""
from __future__ import annotations

import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_new_common import DATA_NEW, get_logger  # noqa: E402

logger = get_logger("collect_news")

OUT_DIR = DATA_NEW / "news"

TV_URL = "https://news-headlines.tradingview.com/v2/headlines"
GOOGLE_URL = "https://news.google.com/rss/search"
GOOGLE_PARAMS = {"hl": "en-PK", "gl": "PK", "ceid": "PK:en"}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.tradingview.com/",
}

# Quarterly windows over the past year. Each windowed query returns its own
# ~100 items, which is what produces markers across the whole chart rather
# than a cluster at the right edge.
BACKFILL_QUARTERS = 4
REQUEST_PAUSE_S = 0.8
RETRIES = 4

TV_COMMODITY_SYMBOLS = {
    "gold": "TVC:GOLD",
    "silver": "TVC:SILVER",
    "copper": "COMEX:HG1!",
    "crude_oil": "TVC:USOIL",
    "natural_gas": "NYMEX:NG1!",
    "wheat": "CBOT:ZW1!",
}

GOOGLE_QUALIFIER = {
    "stock": "PSX",
    "crypto": "crypto price",
    "commodity": "price market",
}

# Drivers of income and money-market NAVs. These are plain (unquoted) queries:
# they are meant to match a topic, not one exact name.
FUND_RATE_QUERIES = (
    "State Bank of Pakistan policy rate",
    "SBP monetary policy decision",
    "Pakistan T-bill auction cut-off yield",
    "Pakistan PIB bond auction",
    "Pakistan inflation CPI",
    "Pakistan mutual fund industry AUM",
)


def _get(url: str, params: dict, *, label: str) -> requests.Response | None:
    """GET with retries. Both upstreams drop connections under sequential load."""
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=25)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001 - any transport error is retryable
            last = e
            if attempt < RETRIES:
                time.sleep(1.5 * attempt)
    logger.debug(f"    {label}: giving up after {RETRIES} attempts ({last})")
    return None


def tradingview_items(symbol: str) -> list[dict]:
    r = _get(TV_URL, {"client": "web", "lang": "en", "symbol": symbol, "streaming": "false"},
             label=f"tv {symbol}")
    if r is None:
        return []
    out = []
    for item in r.json().get("items", []):
        title, published = item.get("title"), item.get("published")
        if not title or not published:
            continue
        # `link` is only set for syndicated partners; TradingView's own wire
        # copy carries none, but `storyPath` is always present.
        url = item.get("link")
        if not url and item.get("storyPath"):
            url = f"https://www.tradingview.com{item['storyPath']}"
        out.append({
            "Date": datetime.fromtimestamp(published, tz=UTC).strftime("%Y-%m-%d"),
            "Headline": title,
            "Url": url,
            "Source": item.get("source"),
            "MarketWide": False,
        })
    return out


def google_items(query: str, needle: str, *, market_wide: bool | None = None) -> list[dict]:
    r = _get(GOOGLE_URL, {"q": query, **GOOGLE_PARAMS}, label=f"google {query[:40]}")
    if r is None:
        return []
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        return []

    out = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        pub = item.findtext("pubDate")
        if not title or not pub:
            continue
        try:
            date = parsedate_to_datetime(pub).astimezone(UTC).strftime("%Y-%m-%d")
        except (TypeError, ValueError):
            continue
        # Market-level stories ("PSX dips 1,109 points") are real catalysts but
        # are not about this asset specifically, so flag rather than drop them.
        out.append({
            "Date": date,
            "Headline": title,
            "Url": item.findtext("link") or None,
            "Source": "Google News",
            "MarketWide": (needle not in title.casefold())
            if market_wide is None
            else market_wide,
        })
    return out


def collect_asset(ticker: str, name: str, asset_class: str) -> pd.DataFrame:
    rows: list[dict] = []

    if asset_class == "commodity":
        symbol = TV_COMMODITY_SYMBOLS.get(ticker)
    elif asset_class == "crypto":
        symbol = f"BINANCE:{ticker}USDT"
    elif asset_class == "stock":
        symbol = f"PSX:{ticker}"
    else:
        symbol = None
    if symbol:
        rows += tradingview_items(symbol)
        time.sleep(REQUEST_PAUSE_S)

    qualifier = GOOGLE_QUALIFIER.get(asset_class)
    if qualifier:
        needle = (name or ticker).casefold()
        base = f'"{name}" {qualifier}' if name else f"{ticker} {qualifier}"
        rows += google_items(base, needle)
        time.sleep(REQUEST_PAUSE_S)

        today = datetime.now(UTC).date()
        for q in range(BACKFILL_QUARTERS):
            end = today - timedelta(days=90 * q)
            start = end - timedelta(days=90)
            windowed = f"{base} after:{start:%Y-%m-%d} before:{end:%Y-%m-%d}"
            rows += google_items(windowed, needle)
            time.sleep(REQUEST_PAUSE_S)

    if not rows:
        return pd.DataFrame(columns=["Date", "Headline", "Url", "Source", "MarketWide"])

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["Date", "Headline"])
    return df.sort_values("Date", ascending=False).reset_index(drop=True)


def collect_fund_feeds() -> tuple[int, int]:
    """Write the shared fund catalyst feeds. Returns (written, failed).

    One rate pool plus one file per asset management company - see the module
    docstring for why funds are not collected individually.
    """
    from backend.services.news_live import FUND_AMCS, FUND_RATE_POOL

    dest = OUT_DIR / "mutual_fund"
    dest.mkdir(parents=True, exist_ok=True)
    written = failed = 0

    # slug -> query, deduped: Al Ameen and UBL share one AMC, as do the three
    # Meezan prefixes.
    jobs: dict[str, tuple[str, list[str]]] = {
        FUND_RATE_POOL: ("rate pool", list(FUND_RATE_QUERIES))
    }
    for _prefix, slug, query in FUND_AMCS:
        jobs.setdefault(f"_amc_{slug}", (slug, [query]))

    for i, (fname, (label, queries)) in enumerate(jobs.items(), 1):
        try:
            rows: list[dict] = []
            for q in queries:
                # Everything here is market-level by construction, so the flag
                # is forced rather than inferred from the headline text.
                rows += google_items(q, "", market_wide=True)
                time.sleep(REQUEST_PAUSE_S)
            if not rows:
                logger.warning(f"  [fund {i}/{len(jobs)}] {label}: no headlines found")
                failed += 1
                continue
            df = pd.DataFrame(rows).drop_duplicates(subset=["Date", "Headline"])
            df = df.sort_values("Date", ascending=False).reset_index(drop=True)
            df.to_csv(dest / f"{fname}.csv", index=False)
            logger.info(
                f"  [fund {i}/{len(jobs)}] {label}: {len(df)} headlines "
                f"({df['Date'].min()} -> {df['Date'].max()})"
            )
            written += 1
        except Exception as e:
            logger.error(f"  [fund {i}/{len(jobs)}] FAILED {label}: {e}")
            failed += 1

    return written, failed


def _universe() -> list[tuple[str, str, str]]:
    """(ticker, name, asset_class) for every forecast asset except mutual funds.

    Read straight from the engines so the news universe cannot drift from the
    assets actually being served.
    """
    from backend.engines import engines

    engines.load_all()
    out = []
    for asset in engines.all_assets():
        ac = asset.asset_class.value
        if ac == "mutual_fund":
            # No source anywhere carries dated headlines for Pakistani
            # open-end funds - not TradingView, not Google News.
            continue
        out.append((asset.ticker, asset.name, ac))
    return out


# Accepted as shorthand on the command line so a whole class can be targeted
# without listing 97 tickers.
_CLASS_ALIASES = {"stock": "stock", "stocks": "stock", "psx": "stock",
                  "crypto": "crypto", "commodity": "commodity", "commodities": "commodity",
                  "mufap": "mutual_fund", "funds": "mutual_fund",
                  "mutual_fund": "mutual_fund", "mutual_funds": "mutual_fund"}


def run(only: list[str] | None = None) -> int:
    universe = _universe()
    # Funds are shared feeds rather than entries in the per-asset universe, so
    # they are selected separately.
    do_funds = True
    if only:
        wanted = {t.casefold() for t in only}
        classes = {_CLASS_ALIASES[w] for w in wanted if w in _CLASS_ALIASES}
        tickers = {w for w in wanted if w not in _CLASS_ALIASES}
        do_funds = "mutual_fund" in classes
        universe = [
            u for u in universe
            if u[2] in classes or u[0].casefold() in tickers
        ]
        if not universe and not do_funds:
            logger.error(f"Nothing matched {only}")
            return 1

    logger.info(f"Collecting news for {len(universe)} assets -> {OUT_DIR}")
    ok, empty, failed = [], [], []

    def _process_one(item):
        i, (ticker, name, asset_class) = item
        try:
            df = collect_asset(ticker, name, asset_class)
            if df.empty:
                return "empty", ticker, f"[{i}/{len(universe)}] {ticker}: no headlines found"
            dest = OUT_DIR / asset_class
            dest.mkdir(parents=True, exist_ok=True)
            df.to_csv(dest / f"{ticker}.csv", index=False)
            return "ok", ticker, f"[{i}/{len(universe)}] {ticker}: {len(df)} headlines ({df['Date'].min()} -> {df['Date'].max()})"
        except Exception as e:
            return "failed", ticker, f"[{i}/{len(universe)}] FAILED {ticker}: {e}"

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_process_one, (i, u)) for i, u in enumerate(universe, 1)]
        for f in as_completed(futures):
            status, ticker, msg = f.result()
            if status == "ok":
                logger.info(f"  {msg}")
                ok.append(ticker)
            elif status == "empty":
                logger.warning(f"  {msg}")
                empty.append(ticker)
            else:
                logger.error(f"  {msg}")
                failed.append(ticker)

    fund_written = fund_failed = 0
    if do_funds:
        logger.info(f"Collecting shared fund feeds -> {OUT_DIR / 'mutual_fund'}")
        fund_written, fund_failed = collect_fund_feeds()

    logger.info("=" * 60)
    logger.info(f"News collection complete: {len(ok)} ok, {len(empty)} empty, {len(failed)} failed")
    if do_funds:
        logger.info(f"Fund feeds: {fund_written} written, {fund_failed} failed")
    if empty:
        logger.warning(f"No headlines: {empty}")
    if failed:
        logger.warning(f"Failed: {failed}")
    # Only hard failures are worth failing the run; an asset the sources simply
    # have nothing on is not a defect the runner can fix.
    return len(failed) + fund_failed


if __name__ == "__main__":
    raise SystemExit(1 if run(sys.argv[1:] or None) else 0)
