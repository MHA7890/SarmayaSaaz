"""
Live NAV lookups for MUFAP mutual funds.

MUFAP publishes every fund's latest NAV on one server-rendered page - no JS,
no auth required:

    https://www.mufap.com.pk/Industry/IndustryStatDaily?tab=3

One fetch covers the whole ~500-row industry table, so unlike the per-ticker
yfinance lookups in live_prices.py this caches the whole parsed table rather
than one entry per fund. NAV is published once per trading day, so the cache
TTL is long: there is nothing to gain by refetching more often, and doing so
just adds load against a Cloudflare-fronted site that responds slowly (or
times out) to rapid consecutive requests without a full browser header set.

Best-effort like live_prices.py: any failure - blocked request, timeout,
unexpected page structure - is caught and callers fall back to the stored
dataset.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from datetime import datetime

import requests

from backend.config import settings

logger = logging.getLogger(__name__)

NAV_URL = "https://www.mufap.com.pk/Industry/IndustryStatDaily?tab=3"

# MUFAP is Cloudflare-fronted and has been observed to time out or reset a
# connection that looks non-browser-like (missing Accept/Referer, or a burst
# of requests). A full header set is what got a reliable 200 in testing.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.mufap.com.pk/",
}

# Table TTL is long on purpose - NAV is published once per trading day.
_TABLE_TTL_S = 1800.0

_ROW_RE = re.compile(r'<tr data-filter="\d+"[^>]*>(.*?)</tr>', re.S)
_CELL_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def _clean(cell: str) -> str:
    return re.sub(r"\s+", " ", _TAG_RE.sub("", cell)).strip()


def _parse(html: str) -> dict[str, dict[str, tuple[float, str]]]:
    """Fund name -> {category -> (NAV, ISO validity date)}. Columns per the
    page's <thead>: Sector, AMC, Fund, Category, Inception Date, Offer,
    Repurchase, NAV, Validity Date, ...

    Voluntary Pension Scheme umbrellas (e.g. "ABL Pension Fund") publish one
    row per sub-fund - VPS-Debt, VPS-Equity, VPS-Money Market - all under the
    identical Fund name, distinguished only by Category. Keeping category as
    a second key is what lets fetch_latest() avoid silently picking the wrong
    sub-fund's NAV for those ~28 umbrella names.
    """
    out: dict[str, dict[str, tuple[float, str]]] = {}
    for row in _ROW_RE.findall(html):
        cells = _CELL_RE.findall(row)
        if len(cells) < 9:
            continue
        fund = _clean(cells[2])
        category = _clean(cells[3])
        try:
            nav = float(_clean(cells[7]))
        except ValueError:
            continue
        try:
            as_of = datetime.strptime(_clean(cells[8]), "%b %d, %Y").strftime("%Y-%m-%d")
        except ValueError:
            continue
        if fund and nav > 0:
            out.setdefault(fund, {})[category] = (nav, as_of)
    return out


_lock = threading.Lock()
_cached_table: dict[str, dict[str, tuple[float, str]]] | None = None
_cached_at: float = 0.0


def _table() -> dict[str, dict[str, tuple[float, str]]]:
    global _cached_table, _cached_at
    now = time.monotonic()
    with _lock:
        if _cached_table is not None and now - _cached_at < _TABLE_TTL_S:
            return _cached_table

    try:
        resp = requests.get(NAV_URL, headers=_HEADERS, timeout=settings.live_price_timeout_s)
        resp.raise_for_status()
        parsed = _parse(resp.text)
        if not parsed:
            raise ValueError("NAV table parsed to zero rows - page structure may have changed")
    except Exception as e:
        logger.info("MUFAP live NAV table unavailable: %s", e)
        with _lock:
            if _cached_table is not None:
                # Serve the stale-but-real table rather than nothing while a
                # transient failure clears up.
                return _cached_table
        return {}

    with _lock:
        _cached_table = parsed
        _cached_at = now
    return parsed


def fetch_latest(fund: str, category: str | None = None) -> tuple[float, str] | None:
    """
    Best-effort (nav, as_of) for one fund name. None if unavailable.

    `category` should be the category the caller already resolved for this
    ticker from the stored dataset (MUFAPEngine tracks one per fund). Most
    fund names have exactly one live row and category is unused; for the
    umbrella names with several sub-funds sharing one name, a category match
    is required - an ambiguous fund with no (or no matching) category hint
    returns None rather than guessing, so the caller falls back to stored.
    """
    if not settings.enable_live_prices:
        return None
    entries = _table().get(fund)
    if not entries:
        return None
    if len(entries) == 1:
        return next(iter(entries.values()))
    if category:
        for cat, value in entries.items():
            if cat.casefold() == category.casefold():
                return value
    return None
