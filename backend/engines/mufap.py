"""
MUFAP engine - Pakistani mutual funds, 5 super-clusters.

NAV is deliberately absent from the engineered feature set (Stage 2 dropped it
to prevent leakage), so the absolute rupee level has to come from the raw
MUFAP export. That file is 94MB, so it is read once at startup with only the
four columns needed, then discarded.

Model inputs come from data-ready/mufap/, the same frames
training-scripts-new/mufap/ built and trained on. Before the swap this engine
read data/mufap_clustered/, which stopped at 2026-08-07 and never advanced -
so every fund's forecast aged by a day, every day, no matter how well
collection ran.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import settings
from backend.engines.base import Engine, EngineError
from backend.engines.routed import RoutedEngine
from backend.schemas import (
    Asset,
    AssetClass,
    Forecast,
    PricePoint,
    Quote,
)
from backend.services import mufap_live, news_live

logger = logging.getLogger(__name__)

# The retrained run produced four clusters; the pre-swap set also carried a
# "Commodity" cluster that no fund actually mapped to. Read from disk rather
# than hardcoding, so this can never drift from the data again.
def _clusters(root: Path) -> tuple[str, ...]:
    if not root.exists():
        return ()
    return tuple(sorted(p.name for p in root.iterdir() if p.is_dir()))


def classify(category: str) -> str:
    """Map a MUFAP category string onto a super-cluster."""
    c = (category or "").casefold()
    if any(k in c for k in ("equity", "index", "etf")):
        return "Equity"
    if "money market" in c:
        return "MoneyMarket"
    if any(k in c for k in ("income", "fixed", "debt", "sovereign")):
        return "Income"
    if any(k in c for k in ("commodit", "gold")):
        return "Balanced"
    return "Balanced"


class MUFAPEngine(RoutedEngine):
    asset_class = AssetClass.MUTUAL_FUND
    name = "MUFAP"
    currency = "PKR"
    unit = "PKR/unit"

    def __init__(self):
        Engine.__init__(self)
        self.models_dir = settings.mufap_models
        self.routing = self.load_json(settings.results_dir / "mufap" / "ensemble_weights.json")
        self._nav: dict[str, float] = {}
        self._category: dict[str, str] = {}
        self._file_for: dict[str, tuple[str, str]] = {}  # ticker -> (cluster, filename)
        self._date_col: dict[str, str] = {}
        self._load_reference()
        self._chart_exact, self._chart_variants = self._index_chart_files()
        self._drop_navless()
        logger.info("MUFAP engine ready: %d funds", len(self._file_for))

    # -- reference data ---------------------------------------------------
    def _load_reference(self) -> None:
        """Build fund -> (category, latest NAV) from the raw export."""
        raw = settings.data_dir / "mufap" / "raw" / "MUFAP_Historical_NAV.csv"
        if raw.exists():
            try:
                df = pd.read_csv(
                    raw,
                    usecols=["Fund", "Category", "NAV", "Validity_Date"],
                    dtype={"Fund": "string", "Category": "string"},
                    low_memory=False,
                )
                df = df.dropna(subset=["Fund"])
                df["NAV"] = pd.to_numeric(df["NAV"], errors="coerce")
                df["Validity_Date"] = pd.to_datetime(df["Validity_Date"], errors="coerce")
                df = df.dropna(subset=["NAV", "Validity_Date"])
                df = df[df["NAV"] > 0].sort_values("Validity_Date")

                latest = df.groupby("Fund", observed=True).last()
                self._nav = {str(k): float(v) for k, v in latest["NAV"].items()}
                self._category = {
                    str(k): str(v) for k, v in latest["Category"].items() if pd.notna(v)
                }
                logger.info("Loaded NAV reference for %d funds", len(self._nav))
            except Exception as e:
                logger.warning("Could not read MUFAP raw export: %s", e)
        else:
            logger.warning("MUFAP raw export not found at %s", raw)

        # Discover which funds are servable, and screen out datasets that are
        # not.
        #
        # The pre-swap export under data/mufap_clustered/ carried two defects -
        # 20 files with a header and zero rows, and 11 that lost their date
        # column's name in a to_csv round trip. data-ready/ has neither, but
        # the screen stays: it is two lines of I/O per file, and it is the
        # difference between a startup warning and a 503 on someone's first
        # click if a future regeneration reintroduces them.
        empty, unnamed = 0, 0
        root = settings.data_ready_dir / "mufap"
        for cluster in _clusters(root):
            directory = root / cluster
            if not directory.exists():
                continue
            for path in directory.glob("*.csv"):
                try:
                    with open(path, encoding="utf-8") as fh:
                        header = fh.readline()
                        first_row = fh.readline()
                except OSError:
                    continue

                if not header or not first_row.strip():
                    empty += 1
                    continue

                date_col = header.split(",")[0].strip()
                if not date_col:
                    date_col = "__index__"
                    unnamed += 1

                self._file_for[path.stem] = (cluster, path.name)
                self._date_col[path.stem] = date_col

        if empty or unnamed:
            logger.warning(
                "MUFAP datasets: %d empty (excluded), %d with an unnamed date column "
                "(recovered positionally)", empty, unnamed,
            )

    def _drop_navless(self) -> None:
        """Remove funds with a feature frame but no NAV anywhere.

        A forecast is a percentage return applied to a rupee NAV, so a fund
        with no NAV cannot produce one - it would reach the user as a 503 on
        click. data-ready/ is built from the clustered export and carries a
        few funds (ETFs, newly listed plans) that neither data-new/mufap-data
        nor the raw export quotes, so they are screened out here instead.
        """
        dropped = [
            fund for fund in self._file_for
            if self._chart_path(fund) is None and not self._nav.get(fund)
        ]
        for fund in dropped:
            del self._file_for[fund]
            self._date_col.pop(fund, None)
        if dropped:
            logger.warning(
                "MUFAP: %d fund(s) excluded, no NAV in data-new or the raw export: %s",
                len(dropped), ", ".join(sorted(dropped)),
            )

    _VARIANT_RE = re.compile(r"^(.*) \(([^()]+)\)$")

    def _index_chart_files(self) -> tuple[dict[str, Path], dict[str, list[tuple[str, Path]]]]:
        """
        Map each fund to its freshly re-collected NAV history under
        data-new/mufap-data/.

        Most funds are a plain "<Fund>.csv". A couple dozen pension/VPS
        umbrellas publish 2-4 NAV series under one fund name distinguished
        only by category (VPS-Debt / VPS-Equity / VPS-Money Market, ...) - see
        collect_mufap_funds.py - so those are written as
        "<Fund> (<Category>).csv" and resolved by category at lookup time
        rather than picking one arbitrarily.
        """
        chart_dir = settings.data_new_dir / "mufap-data"
        exact: dict[str, Path] = {}
        variants: dict[str, list[tuple[str, Path]]] = {}
        if not chart_dir.exists():
            return exact, variants

        for path in chart_dir.glob("*.csv"):
            stem = path.stem
            match = self._VARIANT_RE.match(stem)
            if match and match.group(1) in self._file_for:
                variants.setdefault(match.group(1), []).append((match.group(2), path))
            elif stem in self._file_for:
                exact[stem] = path
        return exact, variants

    def _chart_path(self, fund: str) -> Path | None:
        if fund in self._chart_exact:
            return self._chart_exact[fund]
        candidates = self._chart_variants.get(fund)
        if not candidates:
            return None
        category = (self._category.get(fund) or "").casefold()
        for cat, path in candidates:
            if cat.casefold() == category:
                return path
        # No confirmed category match (fund missing from the raw NAV export,
        # or its category text differs slightly) - any sub-series is still a
        # real, accurate NAV history for this fund, so pick deterministically
        # rather than falling back to the old synthetic reconstruction.
        return sorted(candidates)[0][1]

    @lru_cache(maxsize=64)
    def _chart_series(self, fund: str) -> pd.Series | None:
        """Display NAV series - the freshly re-collected MUFAP export under
        data-new/, which is what the chart and quoted NAV should reflect."""
        path = self._chart_path(fund)
        if path is None:
            return None
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        nav = df["NAV"].dropna()
        return nav if not nav.empty else None

    def _build_catalog(self) -> dict[str, Asset]:
        return {
            fund: Asset(
                ticker=fund,
                name=fund,
                asset_class=self.asset_class,
                group=cluster,
                currency="PKR",
                unit="PKR/unit",
            )
            for fund, (cluster, _) in self._file_for.items()
        }

    @lru_cache(maxsize=64)
    def _frame(self, fund: str) -> pd.DataFrame:
        """Model inputs. Column order here is the feature order the scaler was
        fitted on - training takes every non-Target column in file order, and
        _feature_matrix reproduces that, so the two must not be reordered
        independently."""
        cluster, filename = self._file_for[fund]
        path = settings.data_ready_dir / "mufap" / cluster / filename
        # Files whose date column lost its name are indexed by position 0.
        index_col = 0 if self._date_col.get(fund) == "__index__" else "Date"
        df = pd.read_csv(path, index_col=index_col, parse_dates=True)
        if df.empty:
            raise EngineError(f"'{fund}' has no rows in its clustered dataset")
        return df

    def _feature_matrix(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        drop = {"Fund"} | {f"Target_{h}d" for h in settings.horizons}
        cols = [
            c for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])
        ]
        return df.iloc[-1:][cols].to_numpy(dtype=np.float64), cols

    def _current_nav(self, fund: str) -> float:
        """Latest NAV, preferring the freshly re-collected chart series over
        the (potentially category-mixed, see collect_mufap_funds.py) raw
        export groupby."""
        series = self._chart_series(fund)
        if series is not None and not series.empty:
            return float(series.iloc[-1])
        nav = self._nav.get(fund)
        if nav and np.isfinite(nav) and nav > 0:
            return float(nav)
        raise EngineError(
            f"No NAV on record for '{fund}'. Neither data-new/mufap-data nor "
            "the raw MUFAP export has a value."
        )

    # -- interface --------------------------------------------------------
    def quote(self, ticker: str) -> Quote:
        asset = self.require(ticker)
        series = self._chart_series(asset.ticker)
        if series is not None and not series.empty:
            nav = float(series.iloc[-1])
            change = None
            if len(series) >= 2 and series.iloc[-2]:
                change = round(float((series.iloc[-1] / series.iloc[-2] - 1) * 100), 2)
        else:
            df = self._frame(asset.ticker)
            nav = self._current_nav(asset.ticker)
            change = None
            if "Daily_Return" in df.columns and pd.notna(df["Daily_Return"].iloc[-1]):
                change = round(float(df["Daily_Return"].iloc[-1]) * 100, 2)
        return Quote(
            ticker=asset.ticker,
            name=asset.name,
            asset_class=self.asset_class,
            group=asset.group,
            currency="PKR",
            unit="PKR/unit",
            current_price=round(nav, 4),
            change_24h_pct=change,
        )

    def forecast(self, ticker: str, *, live: bool = True) -> Forecast:
        asset = self.require(ticker)
        cluster = asset.group
        df = self._frame(asset.ticker)

        live_nav = (
            mufap_live.fetch_latest(asset.ticker, self._category.get(asset.ticker))
            if live else None
        )
        if live_nav is not None:
            nav, as_of = live_nav
            price_source = "live"
            # MUFAP's live page is a same-day snapshot with no prior-day NAV
            # alongside it, so there is nothing to diff a day-over-day change
            # from - left null rather than comparing against the (possibly
            # many days older) stored NAV and mislabeling a multi-day drift
            # as a 24h change.
            change_24h_pct = None
        else:
            chart_series = self._chart_series(asset.ticker)
            if chart_series is not None and not chart_series.empty:
                nav = float(chart_series.iloc[-1])
                as_of = chart_series.index[-1].strftime("%Y-%m-%d")
            else:
                nav = self._current_nav(asset.ticker)
                as_of = df.index[-1].strftime("%Y-%m-%d")
            price_source = "stored"
            change_24h_pct = self.quote(asset.ticker).change_24h_pct

        X_raw, _cols = self._feature_matrix(df)
        scaler = self._scaler(cluster)
        X_scaled = scaler.transform(X_raw)

        warnings_out: list[str] = []
        horizons = self._forecast_horizons(
            cluster, X_scaled, nav, settings.horizons, warnings_out
        )
        if not horizons:
            raise EngineError(f"No horizon could be forecast for '{asset.ticker}'")

        # Chart history - the freshly re-collected true NAV series where
        # available, falling back to a Daily_Return reconstruction from the
        # feature frame only for funds data-new couldn't match.
        history: list[PricePoint] = []
        chart_series = self._chart_series(asset.ticker)
        if chart_series is not None:
            history = [
                PricePoint(date=idx.strftime("%Y-%m-%d"), price=round(float(v), 4))
                for idx, v in chart_series.tail(400).items()
            ]
        elif "Daily_Return" in df.columns:
            tail = df["Daily_Return"].tail(400).fillna(0.0)
            factors = (1.0 + tail).iloc[::-1].cumprod().iloc[::-1]
            for idx, factor in zip(tail.index, factors, strict=True):
                if factor:
                    history.append(
                        PricePoint(date=idx.strftime("%Y-%m-%d"),
                                   price=round(nav / float(factor), 4))
                    )

        headline = horizons[len(horizons) // 2].predicted_return_pct
        # Funds have no headlines of their own; these are cluster-level rate
        # and market catalysts, flagged market_wide by news_live.
        catalysts = news_live.fetch_latest(
            asset.ticker, self.asset_class, name=asset.name, group=cluster
        )
        return Forecast(
            ticker=asset.ticker,
            name=asset.name,
            asset_class=self.asset_class,
            group=cluster,
            currency="PKR",
            unit="PKR/unit",
            current_price=round(nav, 4),
            change_24h_pct=change_24h_pct,
            as_of=as_of,
            price_source=price_source,
            action=self.action_of(headline),
            horizons=horizons,
            history=history,
            catalysts=catalysts,
            warnings=warnings_out,
        )
