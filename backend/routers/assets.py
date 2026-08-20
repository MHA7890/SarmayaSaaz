"""Asset catalog endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Query

from backend.engines import engines
from backend.schemas import Asset, AssetClass, AssetList

router = APIRouter(prefix="/api/assets", tags=["Assets"])


@router.get("", response_model=AssetList)
async def list_assets(
    asset_class: AssetClass | None = Query(None, description="Restrict to one asset class"),
    search: str | None = Query(None, description="Case-insensitive ticker or name match"),
    limit: int = Query(500, ge=1, le=2000),
) -> AssetList:
    """Every asset the loaded engines can serve."""
    if asset_class is not None:
        items = engines.get(asset_class).list_assets()
    else:
        items = engines.all_assets()

    if search:
        needle = search.casefold()
        items = [
            a for a in items
            if needle in a.ticker.casefold() or needle in a.name.casefold()
        ]

    return AssetList(count=len(items), assets=items[:limit])


@router.get("/{ticker}", response_model=Asset)
async def get_asset(ticker: str, asset_class: AssetClass | None = None) -> Asset:
    _engine, asset = engines.find(ticker, asset_class)
    return asset
