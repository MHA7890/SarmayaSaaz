"""
SarmayaSaaz API.

    uv run uvicorn backend.main:app --reload --port 8000

Engines are constructed once at startup and shared. No module in this package
mutates the process working directory - every path resolves from
backend.config.settings.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.engines import AssetNotFound, EngineError, engines
from backend.routers import assets, forecasts, market, models, system
from backend.services.snapshot import snapshot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sarmayasaaz")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s v%s", settings.app_name, settings.version)
    engines.load_all()

    online = len(engines.online)
    logger.info("%d/4 engines online, %d assets total", online, engines.asset_count)
    if online == 0:
        logger.error("No engine loaded - every prediction endpoint will return 503")

    if snapshot.load():
        logger.info("Snapshot ready: %s", snapshot.meta())
    else:
        logger.warning(
            "No forecast snapshot on disk. /api/market and /api/movers will be "
            "empty until one is built (uv run python scripts/build_snapshot.py)"
        )

    yield
    logger.info("Shutting down")


app = FastAPI(
    title=f"{settings.app_name} API",
    description=(
        "Multi-asset quantitative forecasting across Pakistani and global markets: "
        "commodities, cryptocurrencies, PSX equities and MUFAP mutual funds."
    ),
    version=settings.version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _no_store(request: Request, call_next):
    """
    Every response here is either derived from data-new/ (refreshed by the
    collector scripts) or the forecast snapshot (refreshed by
    build_snapshot.py) - never something a client should hold onto past this
    request. Without this, a browser can serve a stale forecast (wrong price,
    stale chart) from its HTTP cache even after the backend data changes.
    """
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return response


# --- Error handling ---------------------------------------------------------
@app.exception_handler(AssetNotFound)
async def _asset_not_found(request: Request, exc: AssetNotFound):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(EngineError)
async def _engine_error(request: Request, exc: EngineError):
    # The engine is reachable but could not answer - not the caller's fault.
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# --- Routers ----------------------------------------------------------------
app.include_router(system.router)
app.include_router(assets.router)
app.include_router(forecasts.router)
app.include_router(market.router)
app.include_router(models.router)


@app.get("/", tags=["System"], include_in_schema=False)
async def root():
    return {
        "name": settings.app_name,
        "version": settings.version,
        "docs": "/docs",
        "health": "/api/health",
    }
