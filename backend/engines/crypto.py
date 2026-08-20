"""
Crypto engine - assets routed through 4 K-Means clusters.

Two deliberate departures from the original stage6_inference.py:

1. No synthetic noise. That script added np.random.normal(0, 0.03) to every
   model output to manufacture ensemble spread for the dashboard. A +/-3%
   random perturbation on a financial forecast is fabrication, so the band here
   is the true min/max across the models that actually voted.

2. The cluster map is computed once at startup rather than per request. It is
   still derived exactly as training did (same features, same random_state) so
   assets keep routing to the models trained on them.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from backend.config import settings
from backend.engines.base import Engine, EngineError
from backend.ml.registry import ArtifactError, registry
from backend.schemas import (
    Asset,
    AssetClass,
    Forecast,
    HorizonForecast,
    ModelContribution,
    PricePoint,
    Quote,
)
from backend.services import live_prices, news_live

logger = logging.getLogger(__name__)

TREE_MODELS = {"XGBoost", "LightGBM", "CatBoost", "RandomForest"}

NAMES = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana", "BNB": "BNB",
    "ADA": "Cardano", "AVAX": "Avalanche", "NEAR": "NEAR Protocol",
    "SUI": "Sui", "APT": "Aptos", "TRX": "TRON", "LINK": "Chainlink",
    "AAVE": "Aave", "UNI": "Uniswap", "CRV": "Curve DAO", "LDO": "Lido DAO",
    "PENDLE": "Pendle", "SNX": "Synthetix", "FET": "Artificial Superintelligence",
    "RNDR": "Render", "FIL": "Filecoin", "GRT": "The Graph", "INJ": "Injective",
    "IMX": "Immutable", "WIF": "dogwifhat",
}

# Minimum validated win-rate for a model to join the ensemble. Below this the
# model was worse than the decision floor during training.
WIN_RATE_FLOOR = 0.4

# Descriptive labels for the real pinned clusters (results/crypto/cluster_map.json),
# derived from actual served membership - not from docs/crypto_project_tracker.md,
# whose thematic names ("Macro Majors", "DeFi Blue Chips", ...) don't match what's
# actually in each numbered cluster today. Cluster 2 is BTC-adjacent code only:
# its sole member, UNI, is itself withheld as mislabeled data (see _quarantine),
# so it never actually serves a user, but keeps a name for completeness.
CLUSTER_NAMES = {
    0: "High Volatility Alts",
    1: "Bitcoin",
    2: "Uniswap",
    3: "Established Alts",
}

SENTIMENT_COLS = (
    "Sentiment_Score", "News_Volume", "Sentiment_EMA_7d", "Sentiment_EMA_30d",
    "FearGreed_Score", "FearGreed_EMA_7d", "FearGreed_EMA_30d",
)


class CryptoEngine(Engine):
    asset_class = AssetClass.CRYPTO
    name = "Crypto"
    currency = "USD"

    def __init__(self):
        super().__init__()
        self.weights = self.load_json(settings.results_dir / "crypto" / "ensemble_weights.json")
        self.sentiment = self.load_json(settings.results_dir / "crypto" / "live_sentiment.json")
        self._clusters = self._compute_clusters()
        self.withheld = self._quarantine()
        logger.info(
            "Crypto engine ready: %d assets across %d clusters (%d withheld)",
            len(self._clusters) - len(self.withheld),
            len(set(self._clusters.values())),
            len(self.withheld),
        )

    # -- clustering -------------------------------------------------------
    def _compute_clusters(self) -> dict[str, int]:
        """
        Load the cluster map the models were actually trained under.

        This must never be recomputed at request time. K-Means cluster labels
        are arbitrary: an identical partition can come back numbered
        differently between runs, and the partition itself shifts as new price
        data arrives. Recomputing produced exactly that - BTC/GRT/IMX moved
        from cluster 1 to 2 and swapped places with UNI, so Bitcoin was scored
        by models trained on a single sub-$10 token. Its MACD_Hist, which is
        denominated in absolute price, z-scored to ~251,000 and the ensemble
        returned five-digit percentage moves.
        """
        pinned = self.load_json(settings.results_dir / "crypto" / "cluster_map.json")
        mapping = pinned.get("clusters") if pinned else None
        if mapping:
            logger.info("Loaded pinned cluster map (%d assets)", len(mapping))
            return {str(k): int(v) for k, v in mapping.items()}

        logger.error(
            "cluster_map.json is missing - falling back to recomputing K-Means. "
            "Assets may be routed to models that were never trained on them. "
            "Regenerate the map from results/crypto/final_predictions.csv."
        )
        return self._recompute_clusters()

    def _recompute_clusters(self) -> dict[str, int]:
        """Last-resort reconstruction. See _compute_clusters for why this is unsafe."""
        directory = settings.data_ready_dir / "crypto"
        rows = []
        for path in sorted(directory.glob("*.csv")):
            ticker = path.stem
            try:
                df = pd.read_csv(path, index_col="Date")
            except Exception as e:
                logger.warning("Skipping %s: %s", ticker, e)
                continue
            rows.append({
                "Ticker": ticker,
                "Vol": df["BTC_Volatility_30d"].mean() if "BTC_Volatility_30d" in df else 0.0,
                "Ret": df["Target_7d"].mean() if "Target_7d" in df else 0.0,
            })
        if not rows:
            raise EngineError("No processed crypto datasets found")

        stats = pd.DataFrame(rows).set_index("Ticker")
        labels = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(
            StandardScaler().fit_transform(stats)
        )
        return dict(zip(stats.index, (int(x) for x in labels), strict=True))

    # -- data quality -----------------------------------------------------
    def _quarantine(self) -> dict[str, str]:
        """
        Exclude assets whose stored price history cannot be trusted.

        UNI is the known case: data/crypto/UNI.csv ends 2025-04-11 while the
        rest of the universe runs to 2026-08-14, peaks at $0.60, and begins
        2020-05-07 - months before Uniswap launched. It is a mislabeled
        download, not Uniswap. Rather than special-casing one ticker, anything
        whose history ends more than STALE_DAYS behind the universe median is
        withheld, because a forecast on the wrong asset is worse than none.
        """
        STALE_DAYS = 90
        last_dates: dict[str, pd.Timestamp] = {}
        for ticker in self._clusters:
            path = settings.data_dir / "crypto" / f"{ticker}.csv"
            if not path.exists():
                continue
            try:
                idx = pd.read_csv(path, usecols=["Date"], parse_dates=["Date"])["Date"]
                if len(idx):
                    last_dates[ticker] = idx.max()
            except Exception as e:
                logger.warning("Cannot read price history for %s: %s", ticker, e)

        if not last_dates:
            return {}

        reference = pd.Series(last_dates).median()
        held = {}
        for ticker, last in last_dates.items():
            lag = (reference - last).days
            if lag > STALE_DAYS:
                held[ticker] = (
                    f"price history ends {last.date()}, {lag} days behind the "
                    f"universe - dataset is stale or mislabeled"
                )
        for ticker, reason in held.items():
            logger.warning("Withholding %s: %s", ticker, reason)
        return held

    def _build_catalog(self) -> dict[str, Asset]:
        return {
            ticker: Asset(
                ticker=ticker,
                name=NAMES.get(ticker, ticker),
                asset_class=self.asset_class,
                group=CLUSTER_NAMES.get(cluster, f"Cluster {cluster}"),
                currency="USD",
                unit="USD/token",
            )
            for ticker, cluster in self._clusters.items()
            if ticker not in self.withheld
        }

    # -- data -------------------------------------------------------------
    @lru_cache(maxsize=32)
    def _features_frame(self, ticker: str) -> pd.DataFrame:
        """Model input features - the engineered frame built from data-new/ by
        training-scripts-new/crypto/01_feature_engineering.py, which is what
        the models in models/crypto/ were retrained on. See
        config.data_ready_dir for why this must not fall back to the legacy
        data_dir/crypto_processed frame."""
        path = settings.data_ready_dir / "crypto" / f"{ticker}.csv"
        if not path.exists():
            raise EngineError(f"Processed dataset missing for {ticker}")
        return pd.read_csv(path, index_col="Date")

    @lru_cache(maxsize=32)
    def _price_frame(self, ticker: str) -> pd.DataFrame:
        path = settings.data_dir / "crypto" / f"{ticker}.csv"
        if not path.exists():
            raise EngineError(f"Price history missing for {ticker}")
        return pd.read_csv(path, index_col="Date", parse_dates=True)

    @lru_cache(maxsize=32)
    def _chart_frame(self, ticker: str) -> pd.DataFrame:
        """Display price series - the freshly re-collected Binance OHLCV under
        data-new/, which is what the chart and quoted price should reflect.
        Falls back to the (feature-pipeline) stored price file if a ticker
        somehow has no data-new export yet."""
        path = settings.data_new_dir / "crypto-data" / f"{ticker}.csv"
        if path.exists():
            return pd.read_csv(path, index_col="Date", parse_dates=True)
        return self._price_frame(ticker)

    def _apply_live_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Overwrite today's row with the freshest FinBERT scores, if present."""
        if not self.sentiment:
            return df
        df = df.copy()
        for col in SENTIMENT_COLS:
            if col in df.columns and col in self.sentiment:
                df.loc[df.index[-1], col] = self.sentiment[col]
        return df

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
            current_price=round(float(close.iloc[-1]), 6),
            change_24h_pct=round(change, 2) if change is not None else None,
        )

    # -- forecast ---------------------------------------------------------
    def forecast(self, ticker: str, *, live: bool = True) -> Forecast:
        asset = self.require(ticker)
        cluster = self._clusters[asset.ticker]
        cluster_key = f"Cluster_{cluster}"

        df = self._apply_live_sentiment(self._features_frame(asset.ticker))
        feature_cols = [c for c in df.columns if not c.startswith("Target_") and c != "Date"]

        price_df = self._chart_frame(asset.ticker)
        close = price_df["Close"]

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

        for h in settings.horizons:
            model_weights = self.weights.get(cluster_key, {}).get(f"{h}d", {})
            if not model_weights:
                continue

            scaler_path = settings.crypto_models / f"{cluster_key}_{h}d_scaler.pkl"
            if not scaler_path.exists():
                warnings_out.append(f"{h}d: scaler missing, skipped")
                continue
            scaler = joblib.load(scaler_path)

            scaled = scaler.transform(df[feature_cols].to_numpy(dtype=np.float64))
            window = scaled[-settings.crypto_seq_length:]
            if len(window) < settings.crypto_seq_length:
                warnings_out.append(f"{h}d: needs {settings.crypto_seq_length} rows of history")
                continue
            X_seq = window[None, ...]
            X_flat = window[-1:]

            contributions: list[ModelContribution] = []
            returns: list[float] = []
            weighted, total_weight = 0.0, 0.0

            for model_name, win_rate in model_weights.items():
                if win_rate <= WIN_RATE_FLOOR:
                    continue
                ext = "pkl" if model_name in TREE_MODELS else "pt"
                path = settings.crypto_models / f"{cluster_key}_{h}d_{model_name}.{ext}"
                try:
                    handle = registry.get(path, model_name=model_name, family="crypto")
                    value = handle.predict(X_flat if handle.kind == "tree" else X_seq)
                except ArtifactError as e:
                    warnings_out.append(f"{h}d/{model_name}: {e}")
                    continue
                if not np.isfinite(value):
                    continue

                returns.append(value)
                weighted += value * win_rate
                total_weight += win_rate
                contributions.append(
                    ModelContribution(
                        model_name=model_name,
                        predicted_return_pct=round(value * 100, 4),
                        weight=round(float(win_rate), 4),
                    )
                )

            if total_weight <= 0 or not returns:
                warnings_out.append(f"{h}d: no model cleared the {WIN_RATE_FLOOR} win rate floor")
                continue

            mean_return = weighted / total_weight
            projected = current_price * (1 + mean_return)
            contributions.sort(key=lambda c: c.weight, reverse=True)

            horizons.append(
                HorizonForecast(
                    horizon_days=h,
                    predicted_return_pct=round(mean_return * 100, 2),
                    projected_price=round(projected, 6),
                    direction=self.direction_of(mean_return * 100),
                    # Real ensemble dispersion - the spread of models that voted.
                    lower_bound=round(current_price * (1 + min(returns)), 6),
                    upper_bound=round(current_price * (1 + max(returns)), 6),
                    confidence_pct=round(total_weight / len(contributions) * 100, 2),
                    primary_model=contributions[0].model_name,
                    contributions=contributions,
                )
            )

        if not horizons:
            raise EngineError(f"No horizon could be forecast for {asset.ticker}")

        headline = horizons[len(horizons) // 2].predicted_return_pct
        history = [
            PricePoint(date=idx.strftime("%Y-%m-%d"), price=round(float(v), 6))
            for idx, v in close.tail(400).items()
        ]
        catalysts = news_live.fetch_latest(asset.ticker, self.asset_class) if live else []

        return Forecast(
            ticker=asset.ticker,
            name=asset.name,
            asset_class=self.asset_class,
            group=asset.group,
            currency="USD",
            unit=asset.unit,
            current_price=round(current_price, 6),
            change_24h_pct=round(change_24h_pct, 2) if change_24h_pct is not None else None,
            as_of=as_of,
            price_source=price_source,
            action=self.action_of(headline),
            sentiment_score=self.sentiment.get("Sentiment_Score"),
            horizons=horizons,
            history=history,
            drivers=[],
            catalysts=catalysts,
            warnings=warnings_out,
        )
