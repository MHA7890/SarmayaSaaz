"""
The API contract.

One response shape across all four engines so the frontend has a single set of
types to render, regardless of whether it is showing a Karachi mutual fund or
a crude oil future.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class AssetClass(StrEnum):
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    MUTUAL_FUND = "mutual_fund"
    STOCK = "stock"


class Direction(StrEnum):
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


class Action(StrEnum):
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------


class Asset(BaseModel):
    ticker: str
    name: str
    asset_class: AssetClass
    group: str = Field(description="Cluster or sector the model routes this asset to")
    currency: str
    unit: str = Field(description="Display unit, e.g. 'USD/oz' or 'PKR/unit'")


class AssetList(BaseModel):
    count: int
    assets: list[Asset]


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


class ModelContribution(BaseModel):
    """One model's vote inside an ensemble, with the weight it carried."""

    model_name: str
    predicted_return_pct: float
    weight: float


class HorizonForecast(BaseModel):
    horizon_days: int
    predicted_return_pct: float
    projected_price: float
    direction: Direction

    # Bounds are derived from each model's held-out validation MAE. When the
    # metrics needed to compute them are unavailable these stay null rather
    # than being filled with an invented percentage band.
    lower_bound: float | None = None
    upper_bound: float | None = None
    confidence_pct: float | None = None

    primary_model: str
    contributions: list[ModelContribution] = []


class MarketDriver(BaseModel):
    """A SHAP attribution for the top-weighted tree model."""

    feature: str
    impact: float
    direction: Direction
    headline: str | None = None
    url: str | None = None


class PricePoint(BaseModel):
    date: str
    price: float


class NewsCatalyst(BaseModel):
    date: str = Field(description="ISO date the headline was published")
    headline: str
    url: str | None = None
    source: str | None = None
    market_wide: bool = Field(
        default=False,
        description="True when the headline isn't specific to this ticker (crypto-wide news)",
    )


class Forecast(BaseModel):
    ticker: str
    name: str
    asset_class: AssetClass
    group: str
    currency: str
    unit: str

    current_price: float
    change_24h_pct: float | None = None
    as_of: str = Field(description="ISO date of the most recent observation used")
    price_source: str = Field(description="'live' or 'stored' - never silently mixed")
    source_url: str | None = Field(
        default=None, description="Public page for the upstream feed this data-new series was collected from"
    )

    action: Action
    sentiment_score: float | None = None

    horizons: list[HorizonForecast]
    history: list[PricePoint] = []
    drivers: list[MarketDriver] = []
    catalysts: list[NewsCatalyst] = []

    warnings: list[str] = Field(
        default=[],
        description="Degradations the caller should surface, e.g. models dropped",
    )


# ---------------------------------------------------------------------------
# Movers / market
# ---------------------------------------------------------------------------


class Mover(BaseModel):
    ticker: str
    name: str
    asset_class: AssetClass
    group: str
    currency: str
    current_price: float
    projected_price: float
    predicted_return_pct: float
    horizon_days: int
    primary_model: str
    confidence_pct: float | None = None


class MoversResponse(BaseModel):
    horizon_days: int
    generated_at: str
    gainers: list[Mover]
    losers: list[Mover]


class Quote(BaseModel):
    ticker: str
    name: str
    asset_class: AssetClass
    group: str
    currency: str
    unit: str
    current_price: float
    change_24h_pct: float | None = None


class MarketResponse(BaseModel):
    count: int
    generated_at: str
    quotes: list[Quote]


# ---------------------------------------------------------------------------
# Model performance
# ---------------------------------------------------------------------------


class ModelScore(BaseModel):
    model_name: str
    directional_accuracy: float | None = None
    mae: float | None = None
    rmse: float | None = None
    r2: float | None = None
    win_rate: float | None = Field(
        default=None,
        description="Crypto only: validated win rate for the model's cluster, not a per-asset accuracy",
    )


class LeaderboardEntry(BaseModel):
    ticker: str
    name: str
    asset_class: AssetClass
    group: str
    horizon_days: int
    scores: list[ModelScore]
    best_model: str | None = None


class Leaderboard(BaseModel):
    horizon_days: int
    metric: str
    models: list[str]
    entries: list[LeaderboardEntry]


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------


class EngineStatus(BaseModel):
    name: str
    online: bool
    asset_count: int
    detail: str | None = None


class Health(BaseModel):
    status: str
    version: str
    engines: list[EngineStatus]
    model_cache: dict


class SystemStats(BaseModel):
    total_assets: int
    engines_online: int
    engines_total: int
    artifacts_on_disk: int
    horizons: list[int]
