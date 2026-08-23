"""
Live price lookups.

Best-effort fetch of the freshest available close per ticker, via yfinance.
Never raises and never blocks a request past `live_price_timeout_s` - a slow
or unreachable upstream degrades to the stored dataset's last row rather than
hanging the API. Results are cached briefly so a burst of requests for the
same ticker doesn't repeat outbound calls.

Mutual funds have no live source: MUFAP publishes NAV only on its own site,
which blocks scraping (403), and there is no Yahoo Finance coverage for
Pakistani open-end funds. That asset class is intentionally absent from the
symbol map below and always resolves to None.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from backend.config import settings
from backend.schemas import AssetClass

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="live-price")

# How long a failed/timed-out fetch is remembered before being retried.
# Deliberately much shorter than the success TTL.
_NEGATIVE_TTL_S = 20.0

# ticker -> Yahoo Finance futures symbol. Copied from the mapping already
# used for live feature fetching in src/commodities/live_data_pipeline.py.
_COMMODITY_SYMBOLS = {
    "gold": "GC=F",
    "silver": "SI=F",
    "copper": "HG=F",
    "crude_oil": "CL=F",
    "natural_gas": "NG=F",
    "wheat": "ZW=F",
}


def _yahoo_symbol(ticker: str, asset_class: AssetClass) -> str | None:
    # Primary price data comes strictly from stored daily exports (TradingView, PSX, Binance, MUFAP)
    # to guarantee 100% consistency across the ribbon, market tables, and forecast models.
    return None



@dataclass(frozen=True)
class LiveQuote:
    price: float
    as_of: str
    change_pct: float | None


_cache: dict[str, tuple[float, LiveQuote | None]] = {}
_cache_lock = threading.Lock()


def _fetch(symbol: str) -> LiveQuote | None:
    import yfinance as yf  # local import - keeps this an optional runtime dep

    history = yf.Ticker(symbol).history(period="5d")
    if history.empty:
        return None
    close = history["Close"].dropna()
    if close.empty:
        return None

    price = float(close.iloc[-1])
    as_of = close.index[-1].strftime("%Y-%m-%d")
    change_pct = None
    if len(close) >= 2 and close.iloc[-2]:
        change_pct = float((close.iloc[-1] / close.iloc[-2] - 1) * 100)
    return LiveQuote(price=price, as_of=as_of, change_pct=change_pct)


def fetch_latest(ticker: str, asset_class: AssetClass) -> LiveQuote | None:
    """Best-effort live quote. Returns None on any failure or unsupported
    asset class - callers should fall back to the stored dataset."""
    if not settings.enable_live_prices:
        return None
    symbol = _yahoo_symbol(ticker, asset_class)
    if symbol is None:
        return None

    now = time.monotonic()
    with _cache_lock:
        cached = _cache.get(symbol)
        if cached:
            age = now - cached[0]
            # A failure (None) is cached only briefly - a slow/contended
            # request shouldn't poison a symbol for the full TTL and make
            # every request for the next 5 minutes look "stored".
            ttl = settings.live_price_cache_ttl_s if cached[1] is not None else _NEGATIVE_TTL_S
            if age < ttl:
                return cached[1]

    try:
        future = _executor.submit(_fetch, symbol)
        result = future.result(timeout=settings.live_price_timeout_s)
    except Exception as e:
        logger.info("Live price unavailable for %s (%s): %s", symbol, ticker, e)
        result = None

    with _cache_lock:
        _cache[symbol] = (now, result)
    return result
