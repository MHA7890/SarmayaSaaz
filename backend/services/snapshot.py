"""
Forecast snapshot.

Top Movers, the market table and the model leaderboard all need results across
the whole universe. Computing 329 assets x 7 horizons on demand takes minutes,
so it is computed once and cached to disk.

The snapshot records when it was built and which assets failed, so the UI can
show its age honestly instead of implying the numbers are live.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.engines import engines
from backend.schemas import Mover, Quote

logger = logging.getLogger(__name__)

SNAPSHOT_PATH: Path = settings.results_dir / "snapshot.json"


class SnapshotStore:
    """Thread-safe cache of whole-universe forecast results."""

    def __init__(self) -> None:
        self._data: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._building = False
        self._mtime: float | None = None

    def _refresh_if_stale(self) -> None:
        """
        Pick up a snapshot rebuilt by another process.

        scripts/build_snapshot.py writes the file directly, so without this the
        API would keep serving whatever it read at startup and silently ignore
        a rebuild until restarted.
        """
        if self._building or not SNAPSHOT_PATH.exists():
            return
        try:
            mtime = SNAPSHOT_PATH.stat().st_mtime
        except OSError:
            return
        if self._mtime is not None and mtime <= self._mtime:
            return
        self.load()

    # -- state ------------------------------------------------------------
    @property
    def available(self) -> bool:
        self._refresh_if_stale()
        return self._data is not None

    @property
    def building(self) -> bool:
        return self._building

    def meta(self) -> dict[str, Any]:
        self._refresh_if_stale()
        if not self._data:
            return {"available": False, "building": self._building}
        return {
            "available": True,
            "building": self._building,
            "generated_at": self._data.get("generated_at"),
            "asset_count": len(self._data.get("assets", {})),
            "failed": len(self._data.get("failures", {})),
        }

    # -- persistence ------------------------------------------------------
    def load(self) -> bool:
        if not SNAPSHOT_PATH.exists():
            return False
        try:
            with open(SNAPSHOT_PATH, encoding="utf-8") as f:
                self._data = json.load(f)
            self._mtime = SNAPSHOT_PATH.stat().st_mtime
            logger.info(
                "Loaded forecast snapshot (%s, %d assets)",
                self._data.get("generated_at"), len(self._data.get("assets", {})),
            )
            return True
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not read snapshot: %s", e)
            return False

    def save(self) -> None:
        if not self._data:
            return
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = SNAPSHOT_PATH.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f)
        tmp.replace(SNAPSHOT_PATH)

    # -- building ---------------------------------------------------------
    def build(self, *, persist: bool = True, progress: bool = False) -> dict[str, Any]:
        with self._lock:
            if self._building:
                raise RuntimeError("A snapshot build is already running")
            self._building = True

        try:
            assets: dict[str, Any] = {}
            failures: dict[str, str] = {}
            catalog = engines.all_assets()
            total = len(catalog)

            for i, asset in enumerate(catalog, 1):
                if progress and i % 25 == 0:
                    logger.info("  snapshot %d/%d", i, total)
                try:
                    engine = engines.get(asset.asset_class)
                    # live=False - a full-universe build would otherwise turn
                    # into hundreds of outbound price/news calls per rebuild.
                    forecast = engine.forecast(asset.ticker, live=False)
                    assets[self._key(asset.asset_class, asset.ticker)] = {
                        "ticker": forecast.ticker,
                        "name": forecast.name,
                        "asset_class": str(forecast.asset_class),
                        "group": forecast.group,
                        "currency": forecast.currency,
                        "unit": forecast.unit,
                        "current_price": forecast.current_price,
                        "change_24h_pct": forecast.change_24h_pct,
                        "as_of": forecast.as_of,
                        "action": str(forecast.action),
                        "horizons": [
                            {
                                "horizon_days": h.horizon_days,
                                "predicted_return_pct": h.predicted_return_pct,
                                "projected_price": h.projected_price,
                                "direction": str(h.direction),
                                "confidence_pct": h.confidence_pct,
                                "primary_model": h.primary_model,
                            }
                            for h in forecast.horizons
                        ],
                    }
                except Exception as e:
                    failures[f"{asset.asset_class}:{asset.ticker}"] = f"{type(e).__name__}: {e}"

            self._data = {
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "assets": assets,
                "failures": failures,
            }
            if persist:
                self.save()
            logger.info("Snapshot built: %d assets, %d failures", len(assets), len(failures))
            return self.meta()
        finally:
            self._building = False

    @staticmethod
    def _key(asset_class: str, ticker: str) -> str:
        return f"{asset_class}:{ticker}"

    # -- queries ----------------------------------------------------------
    def _rows(self, asset_class: str | None = None) -> list[dict]:
        self._refresh_if_stale()
        if not self._data:
            return []
        rows = list(self._data["assets"].values())
        if asset_class:
            rows = [r for r in rows if r["asset_class"] == asset_class]
        return rows

    def quotes(self, asset_class: str | None = None) -> list[Quote]:
        return [
            Quote(
                ticker=r["ticker"], name=r["name"], asset_class=r["asset_class"],
                group=r["group"], currency=r["currency"], unit=r["unit"],
                current_price=r["current_price"], change_24h_pct=r["change_24h_pct"],
            )
            for r in self._rows(asset_class)
        ]

    def movers(
        self, horizon_days: int, *, asset_class: str | None = None, limit: int = 10
    ) -> tuple[list[Mover], list[Mover]]:
        candidates: list[Mover] = []
        for r in self._rows(asset_class):
            match = next(
                (h for h in r["horizons"] if h["horizon_days"] == horizon_days), None
            )
            if match is None:
                continue
            candidates.append(
                Mover(
                    ticker=r["ticker"], name=r["name"], asset_class=r["asset_class"],
                    group=r["group"], currency=r["currency"],
                    current_price=r["current_price"],
                    projected_price=match["projected_price"],
                    predicted_return_pct=match["predicted_return_pct"],
                    horizon_days=horizon_days,
                    primary_model=match["primary_model"],
                    confidence_pct=match["confidence_pct"],
                )
            )

        ranked = sorted(candidates, key=lambda m: m.predicted_return_pct, reverse=True)
        gainers = [m for m in ranked if m.predicted_return_pct > 0][:limit]
        losers = [m for m in reversed(ranked) if m.predicted_return_pct < 0][:limit]
        return gainers, losers

    @property
    def generated_at(self) -> str | None:
        return self._data.get("generated_at") if self._data else None


snapshot = SnapshotStore()
