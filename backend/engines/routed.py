"""
Shared base for engines that route through a winning-model table.

MUFAP and PSX were built the same way: group the universe (super-cluster or
sector), train XGBoost/LightGBM/LSTM per group per horizon, then record the
single winner in ensemble_weights.json as {group: {"60d": {Winning_Model, MAE,
Path}}}. Both consumed a single feature row through an identical tabular LSTM.

Unlike the commodity and crypto engines these do not ensemble - one model owns
each cell - so the confidence band comes from that model's held-out MAE, which
both engines record as a fraction of price.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np

from backend.engines.base import Engine, EngineError
from backend.ml.registry import ArtifactError, registry
from backend.schemas import HorizonForecast, ModelContribution

logger = logging.getLogger(__name__)


class RoutedEngine(Engine):
    """Engine whose predictions are routed to one winning model per group/horizon."""

    models_dir: Path
    routing: dict

    def _scaler(self, group: str):
        path = self.models_dir / f"{group}_scaler.pkl"
        if not path.exists():
            raise EngineError(f"Scaler missing for group '{group}'")
        return joblib.load(path)

    def _route(self, group: str, horizon: int) -> tuple[str, Path, float | None] | None:
        entry = self.routing.get(group, {}).get(f"{horizon}d")
        if not entry:
            return None
        name = entry.get("Winning_Model")
        rel = entry.get("Path")
        if not name or not rel:
            return None
        return name, self.models_dir / rel, entry.get("MAE")

    def _forecast_horizons(
        self,
        group: str,
        X_scaled: np.ndarray,
        current_price: float,
        horizons: tuple[int, ...],
        warnings_out: list[str],
        *,
        round_to: int = 4,
    ) -> list[HorizonForecast]:
        """Run every horizon for one asset through its routed model."""
        out: list[HorizonForecast] = []

        for h in horizons:
            route = self._route(group, h)
            if route is None:
                warnings_out.append(f"{h}d: no model registered for '{group}'")
                continue
            model_name, path, mae = route

            family = "tabular" if model_name.upper() == "LSTM" else "commodity"
            try:
                handle = registry.get(path, model_name=model_name, family=family)
                value = handle.predict(X_scaled)
            except ArtifactError as e:
                warnings_out.append(f"{h}d/{model_name}: {e}")
                continue

            if not np.isfinite(value):
                warnings_out.append(f"{h}d/{model_name} produced a non finite value")
                continue

            projected = current_price * (1 + value)
            lower = upper = None
            if mae is not None and np.isfinite(mae):
                # MAE is recorded as a fraction of price for both engines.
                lower = round(projected * (1 - float(mae)), round_to)
                upper = round(projected * (1 + float(mae)), round_to)

            out.append(
                HorizonForecast(
                    horizon_days=h,
                    predicted_return_pct=round(value * 100, 2),
                    projected_price=round(projected, round_to),
                    direction=self.direction_of(value * 100),
                    lower_bound=lower,
                    upper_bound=upper,
                    confidence_pct=(
                        round(max(0.0, 1.0 - float(mae)) * 100, 2)
                        if mae is not None and np.isfinite(mae) else None
                    ),
                    primary_model=model_name,
                    contributions=[
                        ModelContribution(
                            model_name=model_name,
                            predicted_return_pct=round(value * 100, 4),
                            weight=1.0,
                        )
                    ],
                )
            )
        return out
