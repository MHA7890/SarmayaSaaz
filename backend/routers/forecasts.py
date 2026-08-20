"""Forecast endpoints - the core prediction surface."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Query
from starlette.concurrency import run_in_threadpool

from backend.engines import engines
from backend.schemas import AssetClass, Forecast

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/forecast", tags=["Forecast"])


@router.get("/{ticker}", response_model=Forecast)
async def forecast(
    ticker: str,
    asset_class: AssetClass | None = Query(
        None, description="Disambiguate when a symbol exists in more than one class"
    ),
) -> Forecast:
    """
    Full multi-horizon forecast for one asset.

    Returns every horizon the engine could compute. Horizons whose models were
    unavailable are omitted and explained in `warnings` rather than being
    filled with a substitute value.
    """
    return await run_in_threadpool(engines.forecast, ticker, asset_class)
