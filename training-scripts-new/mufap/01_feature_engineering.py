"""
MUFAP feature engineering: data-new/mufap-data/*.csv (accurate NAV history,
one file per fund - umbrella pension funds split by sub-category, see
scripts/collect_mufap_funds.py) -> data-ready/mufap/<cluster>/*.csv.

Reproduces src/mufap/stage2_feature_engineering.py (return/momentum/
volatility features from NAV) and stage3_macro_clustering.py (category ->
cluster classification + PAK/PKR/GC macro blend), against the new data.

Category source: data-new/mufap-data filenames already carry the category
for split umbrella funds ("<Fund> (<Category>).csv"); every other fund's
category comes from data/mufap/raw/MUFAP_Historical_NAV.csv (categorical
metadata, not price data - not part of what data-new/ was collected to fix).

One deliberate fix vs. the original: classify() here matches the *production*
backend/engines/mufap.py::classify() (which treats "sovereign" as Income) -
the original training-time stage3_macro_clustering.py omitted that keyword,
which was a real train/serve skew. Fixing it here removes that skew instead
of reproducing it.

NAV itself is dropped from the feature set (anti-leakage, matching the
original exactly) - Daily_Return and its lags carry the relative dynamics;
the absolute NAV level for display/current-price still comes from
data-new/mufap-data directly, same as production.

Run:
    uv run python training-scripts-new/mufap/01_feature_engineering.py
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import ta
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "training-scripts-new"))
from common.progress import StageProgress  # noqa: E402

RAW_DIR = ROOT / "data-new" / "mufap-data"
OUT_DIR = ROOT / "data-ready" / "mufap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [7, 14, 28, 42, 60, 90, 120]
CLUSTERS = ("Equity", "MoneyMarket", "Income", "Commodity", "Balanced")
VARIANT_RE = re.compile(r"^(.*) \(([^()]+)\)$")


def classify(category: str) -> str:
    """Matches backend/engines/mufap.py::classify() exactly."""
    c = (category or "").casefold()
    if any(k in c for k in ("equity", "index", "etf")):
        return "Equity"
    if "money market" in c:
        return "MoneyMarket"
    if any(k in c for k in ("income", "fixed", "debt", "sovereign")):
        return "Income"
    if any(k in c for k in ("commodit", "gold")):
        return "Commodity"
    return "Balanced"


def load_category_lookup() -> dict[str, str]:
    raw = ROOT / "data" / "mufap" / "raw" / "MUFAP_Historical_NAV.csv"
    if not raw.exists():
        print(f"  ! {raw} not found - funds without a filename category will default to Balanced")
        return {}
    df = pd.read_csv(raw, usecols=["Fund", "Category"], dtype={"Fund": "string", "Category": "string"},
                      low_memory=False)
    df = df.dropna(subset=["Fund", "Category"])
    latest = df.drop_duplicates(subset=["Fund"], keep="last")
    return dict(zip(latest["Fund"].astype(str), latest["Category"].astype(str), strict=True))


def fund_category(filename_stem: str, lookup: dict[str, str]) -> str:
    match = VARIANT_RE.match(filename_stem)
    if match:
        return match.group(2)  # umbrella variant - category is right in the filename
    return lookup.get(filename_stem, "")


def nav_features(df: pd.DataFrame) -> pd.DataFrame:
    nav = df["NAV"]
    df["Daily_Return"] = nav.pct_change()
    df["Log_Return"] = np.log(nav / nav.shift(1))
    for lag in [1, 2, 3, 5, 10]:
        df[f"Ret_Lag_{lag}"] = df["Log_Return"].shift(lag)

    df["NAV_to_SMA_20"] = nav / nav.rolling(20).mean()
    df["NAV_to_SMA_50"] = nav / nav.rolling(50).mean()
    df["NAV_to_SMA_200"] = nav / nav.rolling(200).mean()
    df["NAV_to_EMA_20"] = nav / nav.ewm(span=20, adjust=False).mean()

    df["RSI_14"] = ta.momentum.RSIIndicator(close=nav, window=14).rsi()
    df["MACD_Pct"] = ta.trend.MACD(close=nav).macd_diff() / nav
    df["Rolling_Std_10"] = df["Log_Return"].rolling(10).std()
    df["Rolling_Std_30"] = df["Log_Return"].rolling(30).std()

    for h in HORIZONS:
        df[f"Target_{h}d"] = (nav.shift(-h) - nav) / nav

    df = df.drop(columns=["NAV"], errors="ignore")  # anti-leakage, matches stage2 exactly
    feature_cols = [c for c in df.columns if not c.startswith("Target_")]
    df = df.dropna(subset=["NAV_to_SMA_200", "Rolling_Std_30", "RSI_14", "MACD_Pct", "Ret_Lag_10"])
    return df


def macro_proxy(ticker: str, prefix: str, start: str, end: str) -> pd.DataFrame:
    print(f"  Fetching {ticker} for {prefix}_* macro proxy ...")
    raw = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    close = raw["Close"]
    out = pd.DataFrame(index=raw.index)
    out[f"{prefix}_Log_Return"] = np.log(close / close.shift(1))
    out[f"{prefix}_SMA_200_Ratio"] = close / ta.trend.sma_indicator(close, window=200)
    out[f"{prefix}_Volatility_30d"] = out[f"{prefix}_Log_Return"].rolling(30).std()
    out.index = pd.to_datetime(out.index).normalize()
    return out


def run():
    files = sorted(RAW_DIR.glob("*.csv"))
    print(f"MUFAP feature engineering: {len(files)} fund files from {RAW_DIR}")

    category_lookup = load_category_lookup()
    start_date = "2015-01-01"
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    pak = macro_proxy("PAK", "PAK", start_date, end_date)
    pkr = macro_proxy("PKR=X", "PKR", start_date, end_date)
    gold = macro_proxy("GC=F", "GC", start_date, end_date)

    cluster_counts = {c: 0 for c in CLUSTERS}
    progress = StageProgress("MUFAP feature engineering", len(files))
    ok, failed, skipped = [], [], []
    for path in files:
        stem = path.stem
        try:
            category = fund_category(stem, category_lookup)
            cluster = classify(category)

            df = pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()
            df = nav_features(df)

            macro_cols: list[str] = []
            if cluster in ("Equity", "Balanced"):
                df = df.join(pak, how="left")
                macro_cols += list(pak.columns)
            if cluster in ("MoneyMarket", "Income", "Balanced"):
                df = df.join(pkr, how="left")
                macro_cols += list(pkr.columns)
            if cluster in ("Commodity",):
                df = df.join(gold, how="left")
                macro_cols += list(gold.columns)

            if macro_cols:
                df = df.ffill()
                df = df.dropna(subset=macro_cols)

            if len(df) < 200:
                progress.step(f"{stem}: SKIPPED ({len(df)} rows after cleaning)")
                skipped.append(stem)
                continue

            cluster_dir = OUT_DIR / cluster
            cluster_dir.mkdir(parents=True, exist_ok=True)
            safe_name = "".join(c for c in stem if c not in '<>:"/\\|?*')
            df.to_csv(cluster_dir / f"{safe_name}.csv")
            cluster_counts[cluster] += 1
            progress.step(f"{stem} [{cluster}]: {len(df)} rows, {df.shape[1]} columns")
            ok.append(stem)
        except Exception as e:
            progress.step(f"{stem}: FAILED ({e})")
            failed.append(stem)
    progress.close()

    print(f"Done: {len(ok)} ok, {len(skipped)} skipped (thin history), {len(failed)} FAILED")
    print(f"Cluster distribution: {cluster_counts}")
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
