"""
Central configuration.

Every path is absolute and derived from PROJECT_ROOT. Nothing in this codebase
calls os.chdir() - that was the root cause of the previous integration's
path fragility, where import order silently determined whether a relative
path resolved.
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Identity ---
    app_name: str = "SarmayaSaaz"
    version: str = "1.0.0"

    # --- Roots ---
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    # Freshly re-collected, single-source, null/outlier-cleaned OHLCV - used
    # for chart display and displayed prices. The engineered feature sets
    # under data_dir (which drive the actual forecasts) are untouched.
    data_new_dir: Path = PROJECT_ROOT / "data-new"
    # Engineered feature sets built from data-new/ by
    # training-scripts-new/*/01_feature_engineering.py. These are the exact
    # frames the models in models/ were trained on, so any engine serving a
    # retrained artifact must read its MODEL INPUTS from here - feeding it the
    # older data_dir frames would be a train/serve skew (same column names and
    # order, different underlying source data).
    #
    # All four classes are migrated. MUFAP was the last: it served
    # data/mufap_clustered/ against pre-swap artifacts until that export
    # froze at 2026-08-07, so its forecasts aged a day every day while the
    # collectors kept data-ready/mufap/ current.
    data_ready_dir: Path = PROJECT_ROOT / "data-ready"
    models_dir: Path = PROJECT_ROOT / "models"
    results_dir: Path = PROJECT_ROOT / "results"

    # --- Engine artifact locations ---
    commodities_models: Path = PROJECT_ROOT / "models" / "commodities" / "models_production"
    crypto_models: Path = PROJECT_ROOT / "models" / "crypto"
    mufap_models: Path = PROJECT_ROOT / "models" / "mufap"
    stocks_models: Path = PROJECT_ROOT / "models" / "stocks"

    # --- Inference behaviour ---
    # Number of deserialized models held in memory. Artifacts are small
    # (DL hidden sizes of 32-64), so this trades a few hundred MB of RAM for
    # ~800ms -> ~5ms on repeat predictions.
    model_cache_size: int = 128

    # Sequence length used by commodity DL models (locked at training time).
    commodity_seq_length: int = 10
    commodity_hidden_size: int = 32
    # Sequence length used by crypto DL models.
    crypto_seq_length: int = 30

    horizons: tuple[int, ...] = (7, 14, 28, 42, 60, 90, 120)

    # --- Live market data ---
    # When True, single-asset forecasts try a live yfinance quote before
    # falling back to the stored dataset's last row. Left False: yfinance
    # quotes a *different instrument* per asset class than the one data-new/
    # is collected from (COMEX futures vs OANDA spot for gold's GC=F, Yahoo's
    # own PSX feed vs PSX's own DPS for stocks, ...), so turning this on
    # reintroduces exactly the cross-source price discrepancies data-new/ was
    # built to eliminate. Re-enable only once live_prices.py quotes each
    # asset class from the same source its history now comes from.
    enable_live_prices: bool = False
    # yfinance pays a one-time session/cookie bootstrap cost with Yahoo on the
    # first call (observed up to ~15-20s); later calls on a warm connection
    # return in under a second. The timeout has to absorb that cold start.
    live_price_timeout_s: float = 20.0
    # How long a fetched live quote is reused before asking Yahoo again.
    live_price_cache_ttl_s: float = 300.0

    # --- News catalysts ---
    # Deliberately independent of enable_live_prices. That flag is off because
    # yfinance quotes a *different instrument* per asset class than data-new/
    # is collected from, which would put a price on screen that disagrees with
    # its own chart. News has nothing to do with price sourcing - a headline is
    # a headline whichever feed the candles came from - so gating the chart's
    # catalyst markers behind the price flag only hid a working feature.
    enable_news_catalysts: bool = True
    # TradingView alone is not enough for chart markers: its PSX coverage is
    # patchy (OGDC 55 headlines, LUCK 3, HBL and ENGRO none) and its endpoint
    # returns only a rolling recent window, so BTC came back spanning two days
    # - every marker would bunch at the right edge of a 90D chart. Google News
    # RSS is merged in for all three classes. Set False for TradingView alone.
    enable_google_news: bool = True

    # --- API ---
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # --- Third-party keys (read from .env; never logged) ---
    news_api_key: str = ""
    eia_api_key: str = ""
    nasdaq_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
