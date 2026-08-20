"""
Commodities feature engineering: data-new/commodities-data/*.csv (accurate
TradingView OHLCV) -> data-ready/commodities/*.csv.

Reproduces src/commodities/stage2_feature_engineering.py (19 technical
features + Return/Log_Return/lags), stage3_macro_merge.py (per-commodity
macro spreadsheet merge) and stage4_sentiment_features.py (daily sentiment
aggregation with exponential decay + multi-span EMA), against the new price
data.

Macro spreadsheets and FinBERT-scored news are reused as-is from data/ -
neither is OHLCV, so neither was part of what data-new/ was collected to
fix, and the news is already scored (re-running FinBERT over it would just
reproduce the same scores at real compute cost for no benefit).

Run:
    uv run python training-scripts-new/commodities/01_feature_engineering.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import ta

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "training-scripts-new"))
from common.progress import StageProgress  # noqa: E402

RAW_DIR = ROOT / "data-new" / "commodities-data"
OLD_DIR = ROOT / "data" / "commodities"
OUT_DIR = ROOT / "data-ready" / "commodities"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [7, 14, 28, 42, 60, 90, 120]
DECAY_FACTOR = 0.95

MACRO_FILES = {
    "gold": "gold-macro.csv",
    "silver": "silver-macro.csv",
    "copper": "copper-macro.xlsx",
    "crude_oil": "crude-macro.xlsx",
    "natural_gas": "natural gas-macro.csv",
    "wheat": "wheat-macro.csv",
}
EXTRA_MACRO = {
    "copper": [("china_pmi.csv", ["China_PMI"])],
    "crude_oil": [("opec_production.csv", ["OPEC_Production"])],
}
RENAME_MAP = {"USD_Index_DXY": "USD_Index", "cpi_inflation_yoy": "CPI_Inflation", "date": "Date"}
DROP_MACRO_COLS = {"Copper_Price", "Crude_Oil_Price", "ingestion_time_utc", "cpi_index", "CPI_Index"}


# --------------------------------------------------------------------------
# Stage 2 equivalent: technical features
# --------------------------------------------------------------------------
def technical_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    df["Return"] = close.pct_change()
    df["Log_Return"] = np.log(close / close.shift(1))
    for lag in [1, 2, 3, 5, 10]:
        df[f"Ret_Lag_{lag}"] = df["Return"].shift(lag)

    df["Daily_Range_Pct"] = (df["High"] - df["Low"]) / close
    for w in [5, 10, 20, 50]:
        df[f"Close_to_MA_{w}"] = close / close.rolling(window=w).mean()
    for w in [10, 20]:
        df[f"Close_to_EMA_{w}"] = close / close.ewm(span=w, adjust=False).mean()

    df["Rolling_Std_10"] = df["Return"].rolling(window=10).std()
    df["Rolling_Std_20"] = df["Return"].rolling(window=20).std()
    df["RSI_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    df["MACD_Pct"] = ta.trend.MACD(close=close).macd_diff() / close

    vol_ma_10 = df["Volume"].rolling(window=10).mean()
    df["Volume_to_MA_10"] = df["Volume"] / (vol_ma_10 + 1e-9)

    # kept temporarily for Sentiment_x_Trend below, dropped before saving
    df["_Close_dev_SMA50"] = close / close.rolling(window=50).mean() - 1.0

    for h in HORIZONS:
        df[f"Target_Close_{h}d"] = close.shift(-h)
        df[f"Target_Return_{h}d"] = ((df[f"Target_Close_{h}d"] - close) / close).clip(lower=-0.8, upper=1.5)

    df = df.dropna(subset=[c for c in df.columns if not c.startswith("Target_")])
    return df


# --------------------------------------------------------------------------
# Stage 3 equivalent: macro merge
# --------------------------------------------------------------------------
def load_macro(name: str) -> pd.DataFrame | None:
    filename = MACRO_FILES.get(name)
    if not filename:
        return None
    path = OLD_DIR / filename
    if not path.exists():
        print(f"  ! macro file missing for {name}: {path}")
        return None

    if filename.endswith(".xlsx"):
        macro = pd.read_excel(path)
    else:
        macro = pd.read_csv(path)
    macro = macro.rename(columns=RENAME_MAP)
    macro = macro.drop(columns=[c for c in DROP_MACRO_COLS if c in macro.columns], errors="ignore")
    macro["Date"] = pd.to_datetime(macro["Date"], errors="coerce", format="mixed", dayfirst=False)
    macro = macro.dropna(subset=["Date"]).set_index("Date").sort_index()
    macro = macro[~macro.index.duplicated(keep="last")]

    for extra_file, cols in EXTRA_MACRO.get(name, []):
        extra_path = OLD_DIR / extra_file
        if not extra_path.exists():
            continue
        extra = pd.read_csv(extra_path, parse_dates=["Date"]).set_index("Date").sort_index()
        macro = macro.join(extra[cols], how="outer")

    return macro


def apply_macro(df: pd.DataFrame, macro: pd.DataFrame | None) -> pd.DataFrame:
    if macro is None or macro.empty:
        return df
    macro_cols = list(macro.columns)
    df = df.join(macro, how="left")
    df[macro_cols] = df[macro_cols].ffill()
    df = df.dropna(subset=macro_cols)  # no backfill - avoids look-ahead bias, matches original
    return df


# --------------------------------------------------------------------------
# Stage 4 equivalent: sentiment aggregation (news already FinBERT-scored)
# --------------------------------------------------------------------------
def load_scored_news(name: str) -> pd.DataFrame:
    frames = []
    for suffix in ("news_historical_scored.csv", "news_live_scored.csv"):
        path = OLD_DIR / f"{name}_{suffix}"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame(columns=["Date", "Weighted_Bullish", "Weighted_Bearish", "NSS"])
    news = pd.concat(frames, ignore_index=True)
    news["Date"] = pd.to_datetime(news["Date"], errors="coerce").dt.normalize()
    return news.dropna(subset=["Date"])


def daily_sentiment(news: pd.DataFrame, date_index: pd.DatetimeIndex) -> pd.DataFrame:
    if news.empty:
        cols = ["News_Volume", "Daily_Bullish", "Daily_Bearish", "Daily_NSS",
                "Sentiment_EMA_7d", "Sentiment_Spike", "Sentiment_EMA_30d", "Sentiment_EMA_60d"]
        return pd.DataFrame(0.0, index=date_index, columns=cols)

    daily = news.groupby("Date").agg(
        News_Volume=("NSS", "count"),
        Daily_Bullish=("Weighted_Bullish", "mean"),
        Daily_Bearish=("Weighted_Bearish", "mean"),
        Daily_NSS=("NSS", "mean"),
    )
    full_range = pd.date_range(min(date_index.min(), daily.index.min()), max(date_index.max(), daily.index.max()), freq="D")
    daily = daily.reindex(full_range)

    # Exponential decay on no-news days (stage4_sentiment_features.py), not flat ffill.
    has_news = daily["News_Volume"].notna()
    daily["News_Volume"] = daily["News_Volume"].fillna(0.0)
    for col in ["Daily_Bullish", "Daily_Bearish", "Daily_NSS"]:
        values = daily[col].to_numpy(dtype=np.float64)
        last_valid = np.nan
        days_since = 0
        out = np.empty_like(values)
        for i, (v, present) in enumerate(zip(values, has_news.to_numpy(), strict=True)):
            if present and not np.isnan(v):
                last_valid = v
                days_since = 0
                out[i] = v
            elif np.isnan(last_valid):
                out[i] = 0.0
            else:
                days_since += 1
                out[i] = last_valid * (DECAY_FACTOR**days_since)
        daily[col] = out

    daily["Sentiment_EMA_7d"] = daily["Daily_NSS"].ewm(span=7, adjust=False).mean()
    daily["Sentiment_Spike"] = daily["Daily_NSS"] - daily["Sentiment_EMA_7d"]
    daily["Sentiment_EMA_30d"] = daily["Daily_NSS"].ewm(span=30, adjust=False).mean()
    daily["Sentiment_EMA_60d"] = daily["Daily_NSS"].ewm(span=60, adjust=False).mean()
    return daily


def apply_sentiment(df: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    daily = daily_sentiment(news, df.index)
    cols = ["News_Volume", "Daily_Bullish", "Daily_Bearish", "Daily_NSS",
            "Sentiment_EMA_7d", "Sentiment_Spike", "Sentiment_EMA_30d", "Sentiment_EMA_60d"]
    df = df.join(daily[cols], how="left")
    df[cols] = df[cols].fillna(0.0)
    df["Sentiment_x_Trend"] = df["Daily_NSS"] * df["_Close_dev_SMA50"]
    df = df.drop(columns=["_Close_dev_SMA50"])
    return df


def run():
    names = sorted(p.stem for p in RAW_DIR.glob("*.csv"))
    print(f"Commodities feature engineering: {len(names)} assets from {RAW_DIR}")

    progress = StageProgress("Commodities feature engineering", len(names))
    ok, failed, skipped = [], [], []
    for name in names:
        try:
            df = pd.read_csv(RAW_DIR / f"{name}.csv", index_col="Date", parse_dates=True).sort_index()
            df = technical_features(df)

            macro = load_macro(name)
            df = apply_macro(df, macro)

            news = load_scored_news(name)
            df = apply_sentiment(df, news)

            df = df.replace([np.inf, -np.inf], np.nan)
            # OHLCV + Target_* stay in the file (the engine reads Close directly from
            # it, same as data/commodities/*.csv) - DROP_COLS in 02_train_all.py is
            # what excludes them from the model's feature vector, not this step.
            non_feature = {"Open", "High", "Low", "Close", "Volume"} | {c for c in df.columns if c.startswith("Target_")}
            feature_cols = [c for c in df.columns if c not in non_feature]
            df = df.dropna(subset=feature_cols)

            if len(df) < 200:
                progress.step(f"{name}: SKIPPED ({len(df)} rows after cleaning)")
                skipped.append(name)
                continue

            df.to_csv(OUT_DIR / f"{name}.csv")
            progress.step(f"{name}: {len(df)} rows, {df.shape[1]} columns")
            ok.append(name)
        except Exception as e:
            progress.step(f"{name}: FAILED ({e})")
            failed.append(name)
    progress.close()

    print(f"Done: {len(ok)} ok, {len(skipped)} skipped (thin history), {len(failed)} FAILED")
    if skipped:
        print(f"Skipped (thin history): {skipped}")
    if failed:
        print(f"FAILED: {failed}")
    return len(failed)


if __name__ == "__main__":
    # Exit non-zero only on real errors, not on assets deliberately skipped
    # for thin history. This used to exit 0 for everything, so a missing
    # optional dependency (openpyxl, for the copper/crude .xlsx macro inputs)
    # silently froze those assets' model inputs while the runner reported
    # success. The two must stay distinct: a short-history fund is expected and
    # benign, and failing the nightly run for it would train everyone to
    # ignore failures.
    raise SystemExit(1 if run() else 0)
