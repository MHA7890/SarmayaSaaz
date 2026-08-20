"""
Model performance endpoints backing the leaderboard and heatmap.

Metric coverage is uneven across engines and this module does not paper over
that. Commodities recorded full held-out metrics per asset per model. Crypto
recorded per-cluster win rates only. MUFAP and PSX recorded just the winning
model and its MAE. Fields with no recorded value are returned as null so the
UI can render "not measured" instead of a fabricated score.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Query

from backend.config import settings
from backend.engines import engines
from backend.schemas import AssetClass, Leaderboard, LeaderboardEntry, ModelScore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/models", tags=["Models"])

COMMODITY_MODELS = [
    "XGBoost", "LightGBM", "CatBoost", "RandomForest",
    "LSTM", "GRU", "Transformer", "N-BEATS", "TFT",
]
CRYPTO_MODELS = [
    "XGBoost", "LightGBM", "CatBoost", "RandomForest",
    "LSTM", "GRU", "Transformer", "NBEATS_Lite", "TFT_Lite",
]


@lru_cache(maxsize=16)
def _commodity_metrics(horizon: int) -> dict:
    merged: dict[str, dict] = {}
    for kind in ("ml", "dl"):
        path = settings.results_dir / "commodities" / f"stage2_{kind}_metrics_{horizon}d.json"
        if not path.exists():
            continue
        try:
            with open(path, encoding="utf-8") as f:
                for asset, models in json.load(f).items():
                    merged.setdefault(asset, {}).update(models)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Unreadable metrics %s: %s", path, e)
    return merged


@router.get("/available")
async def available() -> dict:
    """Which architectures exist per asset class, and what metrics were recorded."""
    return {
        "commodity": {
            "models": COMMODITY_MODELS,
            "metrics": ["directional_accuracy", "mae", "rmse", "r2"],
            "granularity": "per asset",
        },
        "crypto": {
            "models": CRYPTO_MODELS,
            "metrics": ["win_rate"],
            "granularity": "per cluster",
        },
        "stock": {
            "models": ["XGBoost", "LightGBM", "LSTM"],
            "metrics": ["mae"],
            "granularity": "per sector, winner only",
        },
        "mutual_fund": {
            "models": ["XGBoost", "LightGBM", "LSTM"],
            "metrics": ["mae"],
            "granularity": "per super-cluster, winner only",
        },
    }


@router.get("/leaderboard", response_model=Leaderboard)
async def leaderboard(
    horizon_days: int = Query(60),
    asset_class: AssetClass = Query(AssetClass.COMMODITY),
    metric: str = Query("directional_accuracy"),
) -> Leaderboard:
    """
    Per-asset model scores at one horizon.

    Only the commodity engine recorded metrics at asset granularity, so other
    classes return their coarser data rather than an interpolation.
    """
    if horizon_days not in settings.horizons and horizon_days != 1:
        raise HTTPException(
            status_code=422,
            detail=f"horizon_days must be one of {[1, *settings.horizons]}",
        )

    entries: list[LeaderboardEntry] = []

    if asset_class == AssetClass.COMMODITY:
        engine = engines.online.get(AssetClass.COMMODITY)
        metrics = _commodity_metrics(horizon_days)
        for ticker, models in metrics.items():
            asset = engine.assets.get(ticker) if engine else None
            scores = [
                ModelScore(
                    model_name=name,
                    directional_accuracy=m.get("Dir_Acc"),
                    mae=m.get("MAE"),
                    rmse=m.get("RMSE"),
                    r2=m.get("R2"),
                )
                for name, m in models.items()
            ]
            best = max(
                (s for s in scores if s.directional_accuracy is not None),
                key=lambda s: s.directional_accuracy,
                default=None,
            )
            entries.append(
                LeaderboardEntry(
                    ticker=ticker,
                    name=asset.name if asset else ticker,
                    asset_class=AssetClass.COMMODITY,
                    group="Commodity",
                    horizon_days=horizon_days,
                    scores=sorted(scores, key=lambda s: s.model_name),
                    best_model=best.model_name if best else None,
                )
            )
        models_axis = COMMODITY_MODELS

    elif asset_class == AssetClass.CRYPTO:
        weights = {}
        path = settings.results_dir / "crypto" / "ensemble_weights.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                weights = json.load(f)
        engine = engines.online.get(AssetClass.CRYPTO)
        for cluster_key, horizons in weights.items():
            cell = horizons.get(f"{horizon_days}d", {})
            if not cell:
                continue
            # Win rate is a ranking signal, not an accuracy percentage; it is
            # reported in its own field rather than dressed up as one.
            scores = [
                ModelScore(model_name=name, directional_accuracy=None, mae=None, win_rate=rate)
                for name, rate in cell.items()
            ]
            best = max(cell.items(), key=lambda kv: kv[1])[0] if cell else None
            cluster_num = cluster_key.removeprefix("Cluster_")
            cluster_name = (
                next((a.group for a in engine.list_assets()
                      if str(engine._clusters.get(a.ticker)) == cluster_num), cluster_key)
                if engine else cluster_key
            )
            members = (
                [a.ticker for a in engine.list_assets()
                 if str(engine._clusters.get(a.ticker)) == cluster_num]
                if engine else []
            )
            unit = "asset" if len(members) == 1 else "assets"
            entries.append(
                LeaderboardEntry(
                    ticker=cluster_name,
                    name=f"{cluster_name} ({len(members)} {unit})",
                    asset_class=AssetClass.CRYPTO,
                    group=cluster_name,
                    horizon_days=horizon_days,
                    scores=scores,
                    best_model=best,
                )
            )
        models_axis = CRYPTO_MODELS

    else:
        source = (
            settings.stocks_models / "ensemble_weights.json"
            if asset_class == AssetClass.STOCK
            else settings.results_dir / "mufap" / "ensemble_weights.json"
        )
        routing = {}
        if source.exists():
            with open(source, encoding="utf-8") as f:
                routing = json.load(f)
        for group, horizons in routing.items():
            cell = horizons.get(f"{horizon_days}d")
            if not cell:
                continue
            entries.append(
                LeaderboardEntry(
                    ticker=group,
                    name=group.replace("_", " "),
                    asset_class=asset_class,
                    group=group,
                    horizon_days=horizon_days,
                    scores=[
                        ModelScore(
                            model_name=cell.get("Winning_Model", "unknown"),
                            mae=cell.get("MAE"),
                        )
                    ],
                    best_model=cell.get("Winning_Model"),
                )
            )
        models_axis = ["XGBoost", "LightGBM", "LSTM"]

    return Leaderboard(
        horizon_days=horizon_days,
        metric=metric,
        models=models_axis,
        entries=sorted(entries, key=lambda e: e.ticker),
    )
