"""System, health and snapshot-control endpoints."""
from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException

from backend.config import settings
from backend.engines import engines
from backend.ml.registry import registry
from backend.schemas import Health, SystemStats
from backend.services.snapshot import snapshot

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["System"])


@router.get("/health", response_model=Health)
async def health() -> Health:
    statuses = engines.status()
    any_online = any(s.online for s in statuses)
    return Health(
        status="ok" if any_online else "degraded",
        version=settings.version,
        engines=statuses,
        model_cache=registry.stats(),
    )


@router.get("/stats", response_model=SystemStats)
async def stats() -> SystemStats:
    artifacts = sum(1 for _ in settings.models_dir.rglob("*") if _.is_file())
    return SystemStats(
        total_assets=engines.asset_count,
        engines_online=len(engines.online),
        engines_total=4,
        artifacts_on_disk=artifacts,
        horizons=list(settings.horizons),
    )


@router.get("/snapshot")
async def snapshot_meta() -> dict:
    """Age and coverage of the cached whole-universe forecast."""
    return snapshot.meta()


@router.post("/snapshot/rebuild", status_code=202)
async def rebuild(background: BackgroundTasks) -> dict:
    """
    Recompute forecasts for every asset.

    Runs in the background because a full pass takes minutes. Poll
    GET /api/snapshot to observe completion.
    """
    if snapshot.building:
        raise HTTPException(status_code=409, detail="A snapshot build is already running")
    background.add_task(snapshot.build, persist=True, progress=True)
    return {"status": "accepted", "detail": "Snapshot rebuild started"}


@router.post("/data/reload", status_code=200)
async def reload_data() -> dict:
    """
    Re-read the on-disk datasets without restarting the process.

    The engines memoise their CSV loads, so once this process has served an
    asset it keeps returning the file it read first. The daily collectors
    rewrite those files underneath it, which means a long-running API goes on
    quoting a stale close while the data on disk is current. scripts/
    daily_update.py calls this after a refresh so the dashboard reflects the
    new data immediately.

    Model artifacts are left alone - they change on a retrain, not a refresh.
    """
    cleared = engines.reload_data()
    reloaded = snapshot.load()
    logger.info("Data reload: caches cleared %s, snapshot reloaded=%s", cleared, reloaded)
    return {
        "status": "ok",
        "caches_cleared": cleared,
        "snapshot_reloaded": reloaded,
        "snapshot": snapshot.meta(),
    }


@router.get("/data/freshness")
async def data_freshness() -> dict:
    """Check dataset freshness across all asset classes against expected market closed date."""
    from backend.services.auto_update import check_data_freshness, get_sync_status
    status = get_sync_status()
    status["freshness"] = check_data_freshness()
    return status


@router.get("/system/status")
async def system_status() -> dict:
    """Returns current sync status, progress, and dataset freshness."""
    from backend.services.auto_update import check_data_freshness, get_sync_status
    status = get_sync_status()
    status["freshness"] = check_data_freshness()
    return status


@router.post("/system/sync-status")
async def update_sync_status(payload: dict) -> dict:
    """Endpoint for background scripts to update current sync state & progress."""
    from backend.services.auto_update import set_sync_status
    set_sync_status(
        is_syncing=payload.get("is_syncing", False),
        step=payload.get("step", "Updating data..."),
        progress=payload.get("progress", 0),
    )
    return {"status": "ok"}


@router.post("/data/sync")
async def sync_data(background: BackgroundTasks) -> dict:
    """
    Check dataset freshness and trigger data procurement if stale.
    """
    from backend.services.auto_update import check_data_freshness, procure_fresh_data
    freshness = check_data_freshness()
    stale = [ac for ac, info in freshness.items() if info["is_stale"]]
    if not stale:
        return {"status": "ok", "detail": "Data is already up to date", "freshness": freshness}
    
    background.add_task(procure_fresh_data, stale)
    return {
        "status": "accepted",
        "detail": f"Procurement started for stale asset classes: {stale}",
        "freshness": freshness,
    }

