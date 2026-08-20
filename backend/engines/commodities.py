"""
Commodities engine - 6 assets, 9 models each, per-asset (not clustered).

Ensembling is weighted by held-out directional accuracy, and the confidence
band is the accuracy-weighted validation MAE. Because commodity MAE is recorded
in absolute price units, those bounds are genuine error bars rather than a
cosmetic percentage spread.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from backend.config import settings
from backend.engines.base import Engine, EngineError
from backend.ml.registry import ArtifactError, registry
from backend.schemas import (
    Asset,
    AssetClass,
    Forecast,
    HorizonForecast,
    MarketDriver,
    ModelContribution,
    PricePoint,
    Quote,
)
from backend.services import live_prices, news_live

logger = logging.getLogger(__name__)

# ticker -> (display name, unit)
CATALOG = {
    "gold": ("Gold", "USD/oz"),
    "silver": ("Silver", "USD/oz"),
    "copper": ("Copper", "USD/lb"),
    "crude_oil": ("WTI Crude Oil", "USD/bbl"),
    "natural_gas": ("Natural Gas", "USD/MMBtu"),
    "wheat": ("Wheat", "USD/bu"),
}

# ticker -> public TradingView symbol page for the exact feed data-new/commodities-data
# was collected from (mirrors the SYMBOLS map in scripts/collect_commodities_tv.py).
SOURCE_URLS = {
    "gold": "https://www.tradingview.com/symbols/OANDA-XAUUSD/",
    "silver": "https://www.tradingview.com/symbols/TVC-SILVER/",
    "copper": "https://www.tradingview.com/symbols/COMEX-HG1!/",
    "crude_oil": "https://www.tradingview.com/symbols/TVC-USOIL/",
    "natural_gas": "https://www.tradingview.com/symbols/NYMEX-NG1!/",
    "wheat": "https://www.tradingview.com/symbols/CBOT-ZW1!/",
}

ARTIFACT_SUFFIX = {
    "XGBoost": ("xgboost", "json"),
    "LightGBM": ("lightgbm", "txt"),
    "CatBoost": ("catboost", "cbm"),
    "RandomForest": ("randomforest", "pkl"),
    "LSTM": ("lstm", "pt"),
    "GRU": ("gru", "pt"),
    "Transformer": ("transformer", "pt"),
    "N-BEATS": ("nbeats", "pt"),
    "TFT": ("tft", "pt"),
}

TREE_MODELS = {"XGBoost", "LightGBM", "CatBoost", "RandomForest"}

# Columns that are targets or raw prices - never model inputs.
DROP_COLS = {
    "Target_Close_Next", "Target_Return_Next", "Target_Direction",
    "Open", "High", "Low", "Close", "Adj Close", "Volume",
}


class CommoditiesEngine(Engine):
    asset_class = AssetClass.COMMODITY
    name = "Commodities"

    def __init__(self, top_n: int = 3):
        super().__init__()
        self.top_n = top_n
        self._metrics: dict[int, dict] = {}
        for h in settings.horizons:
            merged: dict[str, dict] = {}
            for kind in ("ml", "dl"):
                data = self.load_json(
                    settings.results_dir / "commodities" / f"stage2_{kind}_metrics_{h}d.json"
                )
                for commodity, models in data.items():
                    merged.setdefault(commodity, {}).update(models)
            self._metrics[h] = merged
        logger.info("Commodities engine ready: %d assets", len(CATALOG))

    # -- catalog ----------------------------------------------------------
    def _build_catalog(self) -> dict[str, Asset]:
        return {
            ticker: Asset(
                ticker=ticker,
                name=name,
                asset_class=self.asset_class,
                group="Commodity",
                currency="USD",
                unit=unit,
            )
            for ticker, (name, unit) in CATALOG.items()
        }

    # -- data -------------------------------------------------------------
    @lru_cache(maxsize=8)
    def _frame(self, ticker: str) -> pd.DataFrame:
        """
        Model-input frame: the engineered features the serving artifacts were
        actually trained on.

        These come from data-ready/commodities (built by
        training-scripts-new/commodities/01_feature_engineering.py out of the
        re-collected data-new/ OHLCV). The older data/commodities frames carry
        the same 36 feature columns in the same order, so feeding them would
        raise nothing at all - it would just silently score a model against
        features computed from a different price source. That is precisely the
        train/serve skew the re-collection existed to remove, so the fallback
        below is a hard error rather than a quiet substitution.
        """
        path = settings.data_ready_dir / "commodities" / f"{ticker}.csv"
        if not path.exists():
            raise EngineError(
                f"Engineered dataset missing for {ticker} at {path}. "
                "Run training-scripts-new/commodities/01_feature_engineering.py."
            )
        df = pd.read_csv(path, index_col="Date", parse_dates=True).ffill()
        return df

    @lru_cache(maxsize=8)
    def _chart_frame(self, ticker: str) -> pd.DataFrame:
        """Display price series - the freshly re-collected TradingView OHLCV
        under data-new/, which is what the chart and quoted price should
        reflect. The engineered feature frame above (_frame) still drives the
        model inputs unchanged. Falls back to the feature frame's own Close
        if a ticker somehow has no data-new export yet."""
        path = settings.data_new_dir / "commodities-data" / f"{ticker}.csv"
        if path.exists():
            return pd.read_csv(path, index_col="Date", parse_dates=True)
        return self._frame(ticker)

    def _features(self, df: pd.DataFrame) -> list[str]:
        return [
            c for c in df.columns
            if c not in DROP_COLS and not c.startswith("Target_")
        ]

    def _scaler(self, ticker: str, horizon: int):
        for name in (f"{ticker}_scaler_{horizon}d.pkl", f"{ticker}_scaler.pkl"):
            path = settings.commodities_models / name
            if path.exists():
                return joblib.load(path)
        raise EngineError(f"No scaler for {ticker} @ {horizon}d")

    def _ranked_models(self, ticker: str, horizon: int) -> list[tuple[str, float, float]]:
        """Top-N models for this asset/horizon, ordered by directional accuracy."""
        models = self._metrics.get(horizon, {}).get(ticker, {})
        ranked = sorted(
            ((name, m.get("Dir_Acc", 0.0), m.get("MAE", 0.0)) for name, m in models.items()),
            key=lambda t: t[1],
            reverse=True,
        )
        return ranked[: self.top_n]

    # -- quotes -----------------------------------------------------------
    def quote(self, ticker: str) -> Quote:
        asset = self.require(ticker)
        close = self._chart_frame(asset.ticker)["Close"]
        change = None
        if len(close) >= 2 and close.iloc[-2]:
            change = float((close.iloc[-1] / close.iloc[-2] - 1) * 100)
        return Quote(
            ticker=asset.ticker,
            name=asset.name,
            asset_class=self.asset_class,
            group=asset.group,
            currency=asset.currency,
            unit=asset.unit,
            current_price=round(float(close.iloc[-1]), 4),
            change_24h_pct=round(change, 2) if change is not None else None,
        )

    # -- forecast ---------------------------------------------------------
    def forecast(self, ticker: str, *, live: bool = True) -> Forecast:
        asset = self.require(ticker)
        df = self._frame(asset.ticker)
        features = self._features(df)
        close = self._chart_frame(asset.ticker)["Close"]

        live_quote = live_prices.fetch_latest(asset.ticker, self.asset_class) if live else None
        if live_quote is not None:
            current_price = live_quote.price
            as_of = live_quote.as_of
            price_source = "live"
            change_24h_pct = live_quote.change_pct
        else:
            current_price = float(close.iloc[-1])
            as_of = close.index[-1].strftime("%Y-%m-%d")
            price_source = "stored"
            change_24h_pct = self.quote(asset.ticker).change_24h_pct

        warnings_out: list[str] = []
        horizons: list[HorizonForecast] = []
        drivers: list[MarketDriver] = []

        for h in settings.horizons:
            ranked = self._ranked_models(asset.ticker, h)
            if not ranked:
                continue

            try:
                scaler = self._scaler(asset.ticker, h)
            except EngineError as e:
                warnings_out.append(str(e))
                continue

            raw = df[features].to_numpy(dtype=np.float64)
            scaled = scaler.transform(raw)
            X_flat = scaled[-1:]
            X_seq = scaled[-settings.commodity_seq_length:][None, ...]
            if X_seq.shape[1] < settings.commodity_seq_length:
                warnings_out.append(f"{h}d: insufficient history for sequence models")
                continue

            contributions: list[ModelContribution] = []
            weighted_return = 0.0
            weighted_mae = 0.0
            total_weight = 0.0

            for model_name, accuracy, mae in ranked:
                stem, ext = ARTIFACT_SUFFIX[model_name]
                path = settings.commodities_models / f"{asset.ticker}_{stem}_{h}d.{ext}"
                try:
                    handle = registry.get(path, model_name=model_name, family="commodity")
                    value = handle.predict(X_flat if handle.kind == "tree" else X_seq)
                except ArtifactError as e:
                    warnings_out.append(f"{h}d/{model_name} unavailable: {e}")
                    continue

                if not np.isfinite(value):
                    warnings_out.append(f"{h}d/{model_name} produced a non finite value")
                    continue

                weight = max(accuracy, 0.0)
                weighted_return += value * weight
                weighted_mae += mae * weight
                total_weight += weight
                contributions.append(
                    ModelContribution(
                        model_name=model_name,
                        predicted_return_pct=round(value * 100, 4),
                        weight=round(weight, 2),
                    )
                )

                if not drivers and model_name in TREE_MODELS:
                    drivers = self._shap_drivers(handle, X_flat, features, asset.ticker)

            if total_weight <= 0:
                warnings_out.append(f"{h}d: every candidate model failed")
                continue

            mean_return = weighted_return / total_weight
            projected = current_price * (1 + mean_return)
            mae_band = weighted_mae / total_weight

            horizons.append(
                HorizonForecast(
                    horizon_days=h,
                    predicted_return_pct=round(mean_return * 100, 2),
                    projected_price=round(projected, 4),
                    direction=self.direction_of(mean_return * 100),
                    lower_bound=round(projected - mae_band, 4),
                    upper_bound=round(projected + mae_band, 4),
                    confidence_pct=round(total_weight / len(contributions), 2),
                    primary_model=contributions[0].model_name,
                    contributions=contributions,
                )
            )

        if not horizons:
            raise EngineError(f"No horizon could be forecast for {asset.ticker}")

        headline_return = next(
            (hz.predicted_return_pct for hz in horizons if hz.horizon_days == 30),
            horizons[len(horizons) // 2].predicted_return_pct,
        )

        history = [
            PricePoint(date=idx.strftime("%Y-%m-%d"), price=round(float(v), 4))
            for idx, v in close.tail(400).items()
        ]
        catalysts = news_live.fetch_latest(asset.ticker, self.asset_class) if live else []

        return Forecast(
            ticker=asset.ticker,
            name=asset.name,
            asset_class=self.asset_class,
            group=asset.group,
            currency=asset.currency,
            unit=asset.unit,
            current_price=round(current_price, 4),
            change_24h_pct=round(change_24h_pct, 2) if change_24h_pct is not None else None,
            as_of=as_of,
            price_source=price_source,
            source_url=SOURCE_URLS.get(asset.ticker),
            action=self.action_of(headline_return),
            sentiment_score=self._sentiment(df),
            horizons=horizons,
            history=history,
            drivers=drivers,
            catalysts=catalysts,
            warnings=warnings_out,
        )

    # -- explainability ---------------------------------------------------
    def _shap_drivers(self, handle, X_flat, features, ticker) -> list[MarketDriver]:
        try:
            import shap

            explainer = shap.TreeExplainer(handle.raw)
            values = np.asarray(explainer.shap_values(X_flat[..., : handle.input_width]))
            if values.ndim > 1:
                values = values[0]

            headline, url = self._latest_news(ticker)
            pairs = sorted(
                zip(features[: handle.input_width], values, strict=False),
                key=lambda t: abs(t[1]),
                reverse=True,
            )

            out = []
            for feature, impact in pairs[:5]:
                is_sentiment = any(
                    tag in feature for tag in ("Sentiment", "Daily_NSS", "News_Volume",
                                               "Daily_Bullish", "Daily_Bearish")
                )
                out.append(
                    MarketDriver(
                        feature=feature,
                        impact=round(float(impact), 6),
                        direction=self.direction_of(float(impact), deadband=0.0),
                        headline=headline if is_sentiment else None,
                        url=url if is_sentiment else None,
                    )
                )
            return out
        except Exception as e:  # SHAP is best-effort; never fail a forecast for it
            logger.debug("SHAP unavailable for %s: %s", ticker, e)
            return []

    def _latest_news(self, ticker: str) -> tuple[str | None, str | None]:
        path = settings.data_dir / "commodities" / f"{ticker}_news_live.csv"
        if not path.exists():
            return None, None
        try:
            news = pd.read_csv(path)
            if news.empty:
                return None, None
            row = news.iloc[-1]
            return row.get("Headline"), row.get("URL")
        except Exception:
            return None, None

    def _sentiment(self, df: pd.DataFrame) -> float | None:
        for col in ("Daily_NSS", "Sentiment_EMA_7d"):
            if col in df.columns:
                value = df[col].iloc[-1]
                if pd.notna(value):
                    return round(float(value), 4)
        return None
