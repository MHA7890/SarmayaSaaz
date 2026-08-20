"""
The Engine contract.

Four asset classes, four very different pipelines, one interface. Anything the
routers need must be expressible through this - so adding a fifth engine never
means touching a router.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path

from backend.schemas import (
    Action,
    Asset,
    AssetClass,
    Direction,
    Forecast,
    Quote,
)

logger = logging.getLogger(__name__)


class EngineError(RuntimeError):
    """Engine could not satisfy the request."""


class AssetNotFound(EngineError):
    """Ticker is not served by this engine."""


class Engine(ABC):
    """Base class carrying the behaviour every engine shares."""

    asset_class: AssetClass
    name: str
    currency: str = "USD"
    unit: str = "USD"

    def __init__(self) -> None:
        self._assets: dict[str, Asset] = {}

    # -- data freshness ---------------------------------------------------
    def reload_data(self) -> int:
        """
        Drop every cached dataframe so the next request re-reads from disk.

        Engines memoise their CSV loads with @lru_cache, which is right for
        serving but means a long-running process keeps returning the file it
        read first. The daily collectors rewrite those CSVs underneath it, so
        without this the API would go on quoting yesterday's close until the
        next restart - the files would be current and the dashboard would not.

        Returns the number of caches cleared. Model artifacts are deliberately
        untouched: they only change on a retrain, not a data refresh.
        """
        cleared = 0
        for name in dir(type(self)):
            attr = getattr(type(self), name, None)
            if hasattr(attr, "cache_clear"):
                attr.cache_clear()
                cleared += 1
        self._assets = self._build_catalog()
        return cleared

    # -- required ---------------------------------------------------------
    @abstractmethod
    def _build_catalog(self) -> dict[str, Asset]:
        """Enumerate every asset this engine can serve, keyed by ticker."""

    @abstractmethod
    def forecast(self, ticker: str, *, live: bool = True) -> Forecast:
        """
        Produce a multi-horizon forecast.

        `live=False` skips any live price/news fetch and uses stored data
        only - set by snapshot.py, which calls this for the whole ~300-asset
        universe and would otherwise turn one rebuild into hundreds of
        outbound network calls.
        """

    @abstractmethod
    def quote(self, ticker: str) -> Quote:
        """Current price only - must stay cheap enough for list views."""

    # -- shared -----------------------------------------------------------
    @property
    def assets(self) -> dict[str, Asset]:
        if not self._assets:
            self._assets = self._build_catalog()
        return self._assets

    def list_assets(self) -> list[Asset]:
        return sorted(self.assets.values(), key=lambda a: a.ticker)

    def has(self, ticker: str) -> bool:
        return self.resolve(ticker) is not None

    def resolve(self, ticker: str) -> str | None:
        """Case-insensitive ticker lookup; returns the canonical key."""
        if ticker in self.assets:
            return ticker
        lowered = ticker.casefold()
        for key in self.assets:
            if key.casefold() == lowered:
                return key
        return None

    def require(self, ticker: str) -> Asset:
        key = self.resolve(ticker)
        if key is None:
            raise AssetNotFound(f"'{ticker}' is not served by the {self.name} engine")
        return self.assets[key]

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def direction_of(return_pct: float, *, deadband: float = 0.5) -> Direction:
        """
        Classify a percentage move. The deadband keeps near-zero forecasts from
        being dressed up as directional calls.
        """
        if return_pct > deadband:
            return Direction.UP
        if return_pct < -deadband:
            return Direction.DOWN
        return Direction.NEUTRAL

    @staticmethod
    def action_of(return_pct: float) -> Action:
        if return_pct > 8:
            return Action.STRONG_BUY
        if return_pct > 2:
            return Action.BUY
        if return_pct < -8:
            return Action.STRONG_SELL
        if return_pct < -2:
            return Action.SELL
        return Action.HOLD

    @staticmethod
    def load_json(path: Path) -> dict:
        if not path.exists():
            logger.warning("Missing registry file: %s", path)
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Unreadable registry %s: %s", path, e)
            return {}

    @staticmethod
    def now_iso() -> str:
        return datetime.now(UTC).isoformat(timespec="seconds")
