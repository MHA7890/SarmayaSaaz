"""
Engine aggregation.

Holds one instance of each engine and resolves a ticker to whichever one owns
it. An engine that fails to initialise is recorded as offline rather than
taking the whole API down - a missing MUFAP export should not stop anyone from
pricing Bitcoin.
"""
from __future__ import annotations

import logging
import traceback

from backend.engines.base import AssetNotFound, Engine, EngineError
from backend.schemas import Asset, AssetClass, EngineStatus, Forecast, Quote

logger = logging.getLogger(__name__)

__all__ = ["Engine", "EngineError", "AssetNotFound", "EngineRegistry", "engines"]


class EngineRegistry:
    """Owns the four engines and routes by ticker or asset class."""

    def __init__(self) -> None:
        self._engines: dict[AssetClass, Engine] = {}
        self._failures: dict[str, str] = {}

    def load_all(self) -> None:
        from backend.engines.commodities import CommoditiesEngine
        from backend.engines.crypto import CryptoEngine
        from backend.engines.mufap import MUFAPEngine
        from backend.engines.stocks import StocksEngine

        for cls in (CommoditiesEngine, CryptoEngine, StocksEngine, MUFAPEngine):
            name = getattr(cls, "name", cls.__name__)
            try:
                engine = cls()
                # Force catalog construction now so a broken engine fails at
                # startup rather than on a user's first request.
                _ = engine.assets
                self._engines[engine.asset_class] = engine
                logger.info("Engine online: %s (%d assets)", name, len(engine.assets))
            except Exception as e:
                self._failures[name] = f"{type(e).__name__}: {e}"
                logger.error("Engine failed: %s -> %s", name, e)
                logger.debug(traceback.format_exc())

    # -- access -----------------------------------------------------------
    @property
    def online(self) -> dict[AssetClass, Engine]:
        return self._engines

    def get(self, asset_class: AssetClass) -> Engine:
        engine = self._engines.get(asset_class)
        if engine is None:
            raise EngineError(f"The {asset_class} engine is offline")
        return engine

    def reload_data(self) -> dict[str, int]:
        """Clear cached dataframes in every online engine. See Engine.reload_data."""
        out: dict[str, int] = {}
        for ac, engine in self._engines.items():
            try:
                out[ac.value] = engine.reload_data()
            except Exception as e:  # noqa: BLE001 - one bad engine must not block the rest
                logger.error("reload_data failed for %s: %s", ac.value, e)
                out[ac.value] = -1
        return out

    def all_assets(self) -> list[Asset]:
        out: list[Asset] = []
        for engine in self._engines.values():
            out.extend(engine.list_assets())
        return out

    def find(self, ticker: str, asset_class: AssetClass | None = None) -> tuple[Engine, Asset]:
        """
        Resolve a ticker to its owning engine.

        Short symbols are checked before MUFAP's long fund names so that an
        exact symbol match always wins.
        """
        if asset_class is not None:
            engine = self.get(asset_class)
            return engine, engine.require(ticker)

        order = [
            AssetClass.CRYPTO, AssetClass.COMMODITY,
            AssetClass.STOCK, AssetClass.MUTUAL_FUND,
        ]
        for ac in order:
            engine = self._engines.get(ac)
            if engine and engine.has(ticker):
                return engine, engine.require(ticker)
        raise AssetNotFound(f"No engine serves '{ticker}'")

    def forecast(
        self, ticker: str, asset_class: AssetClass | None = None, *, live: bool = True
    ) -> Forecast:
        engine, asset = self.find(ticker, asset_class)
        return engine.forecast(asset.ticker, live=live)

    def quote(self, ticker: str, asset_class: AssetClass | None = None) -> Quote:
        engine, asset = self.find(ticker, asset_class)
        return engine.quote(asset.ticker)

    # -- introspection ----------------------------------------------------
    def status(self) -> list[EngineStatus]:
        out = [
            EngineStatus(name=e.name, online=True, asset_count=len(e.assets))
            for e in self._engines.values()
        ]
        out.extend(
            EngineStatus(name=name, online=False, asset_count=0, detail=detail)
            for name, detail in self._failures.items()
        )
        return sorted(out, key=lambda s: s.name)

    @property
    def asset_count(self) -> int:
        return sum(len(e.assets) for e in self._engines.values())


engines = EngineRegistry()
