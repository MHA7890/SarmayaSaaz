"""Market table and Top Movers - both served from the cached snapshot."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.config import settings
from backend.schemas import AssetClass, MarketResponse, MoversResponse
from backend.services.snapshot import snapshot

router = APIRouter(prefix="/api", tags=["Market"])


def _require_snapshot() -> None:
    if not snapshot.available:
        raise HTTPException(
            status_code=503,
            detail=(
                "No forecast snapshot available yet. Build one with "
                "`uv run python scripts/build_snapshot.py` or POST /api/snapshot/rebuild."
            ),
        )


@router.get("/market", response_model=MarketResponse)
async def market(
    asset_class: AssetClass | None = Query(None),
) -> MarketResponse:
    """Current quotes across the universe, optionally filtered by class."""
    _require_snapshot()
    quotes = snapshot.quotes(str(asset_class) if asset_class else None)
    return MarketResponse(
        count=len(quotes),
        generated_at=snapshot.generated_at or "",
        quotes=sorted(quotes, key=lambda q: q.ticker),
    )


@router.get("/movers", response_model=MoversResponse)
async def movers(
    horizon_days: int = Query(30, description="Forecast horizon to rank by"),
    asset_class: AssetClass | None = Query(None),
    limit: int = Query(10, ge=1, le=50),
) -> MoversResponse:
    """
    Highest predicted gainers and losers at a given horizon.

    Ranked on model output, not on realised price movement.
    """
    _require_snapshot()

    if horizon_days not in settings.horizons:
        raise HTTPException(
            status_code=422,
            detail=f"horizon_days must be one of {list(settings.horizons)}",
        )

    gainers, losers = snapshot.movers(
        horizon_days,
        asset_class=str(asset_class) if asset_class else None,
        limit=limit,
    )
    return MoversResponse(
        horizon_days=horizon_days,
        generated_at=snapshot.generated_at or "",
        gainers=gainers,
        losers=losers,
    )
