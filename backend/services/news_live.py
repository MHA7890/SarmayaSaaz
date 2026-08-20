"""
News catalyst lookups for the price history chart.

Live headlines come from TradingView's public news-headlines endpoint - no
auth, confirmed reachable and genuinely per-instrument (Ethereum-specific
news for ETH, Solana-specific for SOL, real PSX corporate filings for PSX
tickers). Mutual funds have no source anywhere: TradingView has no coverage
for Pakistani open-end funds, and nothing else in this codebase or on disk
carries dated fund-level headlines.

For commodities, live headlines are merged with the local multi-year archive
(data/commodities/{ticker}_news_historical.csv) so older dates on a 1Y chart
still have catalyst markers - TradingView's endpoint only returns a rolling
window of recent headlines.

Best-effort like live_prices.py and mufap_live.py: any failure is caught and
callers get an empty list rather than a broken chart.
"""
from __future__ import annotations

import logging
import threading
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime

import pandas as pd
import requests

from backend.config import settings
from backend.schemas import AssetClass, NewsCatalyst

logger = logging.getLogger(__name__)

NEWS_URL = "https://news-headlines.tradingview.com/v2/headlines"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.tradingview.com/",
}

# Commodity ticker -> TradingView symbol. Confirmed live.
_COMMODITY_SYMBOLS = {
    "gold": "TVC:GOLD",
    "silver": "TVC:SILVER",
    "copper": "COMEX:HG1!",
    "crude_oil": "TVC:USOIL",
    "natural_gas": "NYMEX:NG1!",
    "wheat": "CBOT:ZW1!",
}

_TABLE_TTL_S = 1800.0
_MAX_LIVE_ITEMS = 30


def _symbol(ticker: str, asset_class: AssetClass) -> str | None:
    if asset_class == AssetClass.COMMODITY:
        return _COMMODITY_SYMBOLS.get(ticker)
    if asset_class == AssetClass.CRYPTO:
        return f"BINANCE:{ticker}USDT"
    if asset_class == AssetClass.STOCK:
        return f"PSX:{ticker}"
    return None


_lock = threading.Lock()
_cache: dict[str, tuple[float, list[NewsCatalyst]]] = {}


def _fetch_live(symbol: str, *, market_wide: bool) -> list[NewsCatalyst]:
    resp = requests.get(
        NEWS_URL,
        params={"client": "web", "lang": "en", "symbol": symbol, "streaming": "false"},
        headers=_HEADERS,
        timeout=settings.live_price_timeout_s,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])

    out: list[NewsCatalyst] = []
    for item in items[:_MAX_LIVE_ITEMS]:
        title = item.get("title")
        published = item.get("published")
        if not title or not published:
            continue
        date = datetime.fromtimestamp(published, tz=UTC).strftime("%Y-%m-%d")
        # `link` is only set for syndicated partners; TradingView's own wire
        # copy (Dow Jones, Reuters) carries none, which left roughly half the
        # markers unclickable. `storyPath` is present on every item and
        # resolves to the full story on tradingview.com.
        url = item.get("link")
        if not url and item.get("storyPath"):
            url = f"https://www.tradingview.com{item['storyPath']}"
        out.append(
            NewsCatalyst(
                date=date,
                headline=title,
                url=url,
                source=item.get("source"),
                market_wide=market_wide,
            )
        )
    return out


def _live_catalysts(ticker: str, asset_class: AssetClass) -> list[NewsCatalyst]:
    symbol = _symbol(ticker, asset_class)
    if symbol is None:
        return []

    now = time.monotonic()
    with _lock:
        cached = _cache.get(symbol)
        if cached and now - cached[0] < _TABLE_TTL_S:
            return cached[1]

    try:
        result = _fetch_live(symbol, market_wide=False)
    except Exception as e:
        logger.info("Live news unavailable for %s (%s): %s", symbol, ticker, e)
        with _lock:
            if cached is not None:
                return cached[1]
        return []

    with _lock:
        _cache[symbol] = (now, result)
    return result


_historical_cache: dict[str, list[NewsCatalyst]] = {}


def _historical_commodity_catalysts(ticker: str) -> list[NewsCatalyst]:
    """Multi-year archive, used to extend commodity coverage past what the
    live endpoint's rolling recent-headlines window returns. No URL exists
    for these older entries - they're not clickable, just dated markers."""
    if ticker in _historical_cache:
        return _historical_cache[ticker]

    path = settings.data_dir / "commodities" / f"{ticker}_news_historical.csv"
    if not path.exists():
        _historical_cache[ticker] = []
        return []

    try:
        df = pd.read_csv(path, usecols=["Date", "Headline", "Publisher"])
        df = df.dropna(subset=["Date", "Headline"])
        out = [
            NewsCatalyst(date=str(row.Date), headline=str(row.Headline),
                         source=str(row.Publisher) if pd.notna(row.Publisher) else None)
            for row in df.itertuples()
        ]
    except Exception as e:
        logger.warning("Could not read historical news for %s: %s", ticker, e)
        out = []

    _historical_cache[ticker] = out
    return out


# --- Google News supplement ----------------------------------------------
# Two gaps make TradingView alone insufficient for chart markers:
#
#   coverage - PSX is thin and uneven (OGDC 55 headlines, LUCK 3, HBL and
#              ENGRO none at all);
#   history  - the endpoint returns only a rolling recent window, so BTC came
#              back with headlines spanning two days. On a 30D or 90D chart
#              every marker would bunch against the right edge.
#
# Google News RSS is keyless, returns dated and linked items, and reaches years
# back (HBL 2023-2026), so it is merged in for all three classes.
_GOOGLE_NEWS_URL = "https://news.google.com/rss/search"
_GOOGLE_PARAMS = {"hl": "en-PK", "gl": "PK", "ceid": "PK:en"}
_MAX_GOOGLE_ITEMS = 40

_google_cache: dict[str, tuple[float, list[NewsCatalyst]]] = {}


# Per-class qualifier appended to the quoted asset name. Without it a search
# for "Copper" returns company news and a search for "Cardano" returns
# unrelated results; the qualifier anchors each query to its market.
_GOOGLE_QUALIFIER = {
    AssetClass.STOCK: "PSX",
    AssetClass.CRYPTO: "crypto price",
    AssetClass.COMMODITY: "price market",
}


def _google_catalysts(
    ticker: str, name: str | None, asset_class: AssetClass
) -> list[NewsCatalyst]:
    if not settings.enable_google_news:
        return []

    qualifier = _GOOGLE_QUALIFIER.get(asset_class)
    if qualifier is None:
        return []

    # Quoting the name keeps a search for "Attock Petroleum" off every story
    # that merely says "petroleum".
    query = f'"{name}" {qualifier}' if name else f"{ticker} {qualifier}"

    now = time.monotonic()
    with _lock:
        cached = _google_cache.get(query)
        if cached and now - cached[0] < _TABLE_TTL_S:
            return cached[1]

    try:
        resp = requests.get(
            _GOOGLE_NEWS_URL,
            params={"q": query, **_GOOGLE_PARAMS},
            headers=_HEADERS,
            timeout=settings.live_price_timeout_s,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as e:
        logger.info("Google News unavailable for %s (%s): %s", ticker, query, e)
        with _lock:
            if cached is not None:
                return cached[1]
        return []

    needle = (name or ticker).casefold()
    out: list[NewsCatalyst] = []
    for item in root.findall(".//item")[:_MAX_GOOGLE_ITEMS]:
        title = (item.findtext("title") or "").strip()
        pub = item.findtext("pubDate")
        link = item.findtext("link")
        if not title or not pub:
            continue
        try:
            date = parsedate_to_datetime(pub).astimezone(UTC).strftime("%Y-%m-%d")
        except (TypeError, ValueError):
            continue
        # Google returns market-level stories ("PSX dips 1,109 points")
        # alongside asset-specific ones. Both are genuine catalysts, but only
        # the latter is about *this* asset, so flag the difference rather than
        # dropping either.
        specific = needle in title.casefold() or ticker.casefold() in title.casefold()
        out.append(
            NewsCatalyst(
                date=date,
                headline=title,
                url=link or None,
                source="Google News",
                market_wide=not specific,
            )
        )

    with _lock:
        _google_cache[query] = (now, out)
    return out


# --- chart-sized sampling --------------------------------------------------
# The chart draws one marker per date and can only show about ten legibly, so
# there is no reason to ship every headline the archive holds: a year of BTC
# news is 675 rows across 164 dates, and the default 90D window renders ten of
# them. Trimming here keeps the payload proportional to what a chart can
# actually display.
_MAX_CATALYST_DATES = 60
_MAX_PER_DATE = 2

# The widest range button on the chart is 1Y, so a marker older than that can
# never be drawn. Commodities carry a decade-deep legacy archive; sampling it
# evenly alongside recent news spent 35 of crude oil's 60 date slots on 2014-2019
# headlines that no range setting reaches. Recent dates therefore get the budget
# first, and older ones only fill what is left over.
_VISIBLE_DAYS = 400


def _sample(dates: list[str], budget: int) -> list[str]:
    """Up to `budget` dates spread evenly across the span, newest first.

    Evenly rather than newest-first: taking the most recent would bunch every
    marker against the right edge of the chart, and the spread across the
    window is the whole point of the markers.
    """
    if budget <= 0:
        return []
    if len(dates) <= budget:
        return dates
    if budget == 1:
        return dates[:1]
    step = (len(dates) - 1) / (budget - 1)
    picked: list[str] = []
    for i in range(budget):
        d = dates[round(i * step)]
        # Rounding can land on the same index twice over a short span.
        if not picked or picked[-1] != d:
            picked.append(d)
    return picked


def _trim(
    items: list[NewsCatalyst],
    max_dates: int = _MAX_CATALYST_DATES,
    *,
    include_older: bool = True,
) -> list[NewsCatalyst]:
    """Reduce a newest-first list to a chart-sized sample, newest first.

    Within a date, a headline that links to an article outranks one that does
    not - the legacy commodity archive carries no URLs, and an unclickable
    marker is the one thing these markers are not supposed to be. Among
    clickable ones, asset-specific outranks market-wide so the marker reads as
    being about this asset wherever such a headline exists.
    """
    by_date: dict[str, list[NewsCatalyst]] = {}
    for c in items:
        by_date.setdefault(c.date, []).append(c)

    all_dates = sorted(by_date, reverse=True)
    cutoff = (datetime.now(UTC).date() - timedelta(days=_VISIBLE_DAYS)).isoformat()
    recent = [d for d in all_dates if d >= cutoff]
    older = [d for d in all_dates if d < cutoff]

    # Older dates are kept only when an asset is too thinly covered to fill the
    # budget from recent news - dropping them outright would leave assets like
    # PGLC, whose only headlines predate the window, with no markers at all.
    # Callers holding a reserved slice pass include_older=False: a slot spent on
    # an invisible date is worse than handing the slot back.
    dates = _sample(recent, max_dates)
    if include_older:
        dates += _sample(older, max_dates - len(dates))

    out: list[NewsCatalyst] = []
    for d in dates:
        ranked = sorted(by_date[d], key=lambda c: (c.url is None, c.market_wide))
        out.extend(ranked[:_MAX_PER_DATE])
    return out


def fetch_latest(
    ticker: str,
    asset_class: AssetClass,
    *,
    name: str | None = None,
    group: str | None = None,
) -> list[NewsCatalyst]:
    """Best-effort catalyst list for one asset, newest first. Empty list if
    unavailable - callers should render a chart with no markers, not fail.

    `name` is the company/display name, used to build the Google News query and
    to resolve a fund's AMC; TradingView is keyed on the ticker symbol alone.
    `group` is the MUFAP cluster, which selects a fund's catalyst pools.
    """
    if not settings.enable_news_catalysts:
        return []

    if asset_class == AssetClass.MUTUAL_FUND:
        return _fund_catalysts(name, group)

    live = _live_catalysts(ticker, asset_class)

    merged = live + _archive_catalysts(ticker, asset_class)
    if asset_class == AssetClass.COMMODITY:
        merged += _historical_commodity_catalysts(ticker)
    return _trim(_dedupe(merged))


# --- on-disk archive -------------------------------------------------------
# Written by scripts/collect_news.py and refreshed by the daily job. The
# windowed Google queries that give markers their spread across a chart take
# seconds per asset, which is fine on a schedule and far too slow inside a page
# request - so the request path reads the archive and only tops it up with
# TradingView's live call for today's headlines.
_archive_cache: dict[str, tuple[float, list[NewsCatalyst]]] = {}


def _read_archive(path) -> list[NewsCatalyst]:
    """One catalyst CSV, cached until the file's mtime changes."""
    if not path.exists():
        return []
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return []

    key = str(path)
    with _lock:
        cached = _archive_cache.get(key)
        if cached and cached[0] == mtime:
            return cached[1]

    try:
        df = pd.read_csv(path)
        df = df.dropna(subset=["Date", "Headline"])
        out = [
            NewsCatalyst(
                date=str(row.Date),
                headline=str(row.Headline),
                url=str(row.Url) if pd.notna(row.Url) else None,
                source=str(row.Source) if pd.notna(row.Source) else None,
                market_wide=bool(row.MarketWide),
            )
            for row in df.itertuples()
        ]
    except Exception as e:
        logger.warning("Could not read news archive %s: %s", path.name, e)
        return []

    with _lock:
        _archive_cache[key] = (mtime, out)
    return out


def _archive_catalysts(ticker: str, asset_class: AssetClass) -> list[NewsCatalyst]:
    return _read_archive(
        settings.data_new_dir / "news" / asset_class.value / f"{ticker}.csv"
    )


# --- mutual funds ----------------------------------------------------------
# Open-end funds generate no headlines of their own - "ABL Income Fund",
# "Al Meezan Mutual Fund", "NBP Stock Fund" and "AL Habib Money Market Fund"
# each return zero Google News items. What does exist is news about the things
# that move a fund's NAV, so catalysts come from shared pools keyed to the
# fund's cluster plus its asset management company. Every one is market_wide:
# none of it is about the individual fund, and saying otherwise would be a lie
# the hover box would repeat.

# Fund-name prefix -> (archive slug, Google News query). Longest prefix wins.
# Al Ameen is UBL's Islamic brand and KSE Meezan is Al Meezan's index fund, so
# both resolve to their parent AMC rather than to a house of their own.
FUND_AMCS: tuple[tuple[str, str, str], ...] = (
    ("ABL", "abl", '"ABL Asset Management"'),
    ("AL Habib", "alhabib", '"AL Habib Asset Management"'),
    ("Al Ameen", "ubl", '"UBL Fund Managers"'),
    ("Al Meezan", "meezan", '"Al Meezan Investments"'),
    ("Alfalah", "alfalah", '"Alfalah Asset Management"'),
    ("EFU", "efu", '"EFU Life"'),
    ("Faysal", "faysal", '"Faysal Asset Management"'),
    ("HBL", "hbl", '"HBL Asset Management"'),
    ("KSE Meezan", "meezan", '"Al Meezan Investments"'),
    ("Meezan", "meezan", '"Al Meezan Investments"'),
    ("UBL", "ubl", '"UBL Fund Managers"'),
)

# The rate pool is collected by scripts/collect_news.py; the equity pool is the
# PSX index feed already collected for the stock charts, reused rather than
# fetched twice.
FUND_RATE_POOL = "_pool_rates"

# AMC news is the only fund-family-specific signal that exists, and there is
# very little of it - roughly one item a year per house. Sampled evenly against
# a ~500-item rate pool it lost almost every time (39 of 77 funds showed none),
# so it gets a reserved slice of the date budget instead.
_MAX_AMC_DATES = 6
_FUND_EQUITY_POOL = ("stock", "PSX")

# Cluster -> which pools actually drive it. Balanced funds hold both, so they
# get both.
_FUND_POOLS: dict[str, tuple[str, ...]] = {
    "MoneyMarket": ("rates",),
    "Income": ("rates",),
    "Equity": ("equity",),
    "Balanced": ("rates", "equity"),
}


def amc_for(name: str) -> tuple[str, str] | None:
    """(slug, query) for the AMC behind a fund name, or None if unrecognised."""
    n = (name or "").casefold()
    for prefix, slug, query in sorted(FUND_AMCS, key=lambda x: -len(x[0])):
        if n.startswith(prefix.casefold()):
            return slug, query
    return None


def _fund_catalysts(name: str | None, group: str | None) -> list[NewsCatalyst]:
    """Trimmed catalysts for one fund, newest first.

    Trimmed here rather than by the caller because the AMC feed and the pools
    have to be budgeted separately - see _MAX_AMC_DATES.
    """
    news_dir = settings.data_new_dir / "news"
    pool: list[NewsCatalyst] = []

    # An unknown cluster falls back to rates: 58 of 77 funds are income or
    # money market, so it is the safer default of the two.
    for source in _FUND_POOLS.get(group or "", ("rates",)):
        if source == "equity":
            cls, ticker = _FUND_EQUITY_POOL
            pool += _read_archive(news_dir / cls / f"{ticker}.csv")
        else:
            pool += _read_archive(news_dir / "mutual_fund" / f"{FUND_RATE_POOL}.csv")

    amc = amc_for(name or "")
    amc_items = (
        _read_archive(news_dir / "mutual_fund" / f"_amc_{amc[0]}.csv") if amc else []
    )

    picked = _trim(_dedupe(amc_items), max_dates=_MAX_AMC_DATES, include_older=False)
    used = len({c.date for c in picked})
    picked += _trim(_dedupe(pool), max_dates=_MAX_CATALYST_DATES - used)

    # Copies, so flagging market_wide cannot mutate another asset's cached rows.
    out = [c.model_copy(update={"market_wide": True}) for c in picked]
    return sorted(out, key=lambda c: c.date, reverse=True)


def _dedupe(items: list[NewsCatalyst]) -> list[NewsCatalyst]:
    """Newest first, one entry per (date, headline)."""
    seen: set[tuple[str, str]] = set()
    out: list[NewsCatalyst] = []
    for c in sorted(items, key=lambda c: c.date, reverse=True):
        key = (c.date, c.headline)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
