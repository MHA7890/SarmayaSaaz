"""
PSX feature engineering: data-new/psx-data/*.csv (accurate PSX DPS OHLCV) ->
data-ready/psx/<sector>/*.csv (feature set + sector macro, ready to train).

Reproduces src/stocks/stage2_feature_engineering.py (technical indicators)
and stage3_macro_clustering.py (static 7-sector grouping + PKR/oil/NASDAQ
macro proxies), against the new data.

One deliberate difference from the original: data-new/psx-data has no
"Adj Close" (PSX's own DPS site doesn't publish a split/dividend-adjusted
series the way yfinance did) - every formula here uses `Close` directly,
matching what backend/engines/stocks.py already falls back to when
"Adj Close" is absent.

Run:
    uv run python training-scripts-new/psx/01_feature_engineering.py
"""
from __future__ import annotations

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
sys.path.insert(0, str(ROOT / "scripts"))
from common.progress import StageProgress  # noqa: E402


RAW_DIR = ROOT / "data-new" / "psx-data"
OUT_DIR = ROOT / "data-ready" / "psx"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [7, 14, 28, 42, 60, 90, 120]

SECTOR_MAP = {
    "Financials": ["ABL", "AHCL", "AICL", "AKBL", "BAFL", "BAHL", "BOP", "FABL",
                   "HBL", "HMB", "MCB", "MEBL", "NBP", "PSX", "SCBPL", "UBL"],
    "Energy_Power": ["APL", "ATRL", "CNERGY", "HUBC", "KAPCO", "KEL", "MARI",
                     "OGDC", "POL", "PPL", "PSO", "SNGP", "SSGC"],
    "Cement_Construction": ["BWCL", "CHCC", "DGKC", "FCCL", "KOHC", "LUCK", "MLCF",
                            "PIOC", "POWER", "THALL", "ISL", "INIL"],
    "Tech_Telecom": ["AIRLINK", "NETSOL", "PTC", "SYS", "TRG"],
    "Pharmaceuticals": ["ABOT", "AGP", "CPHL", "GLAXO", "HALEON", "HINOON", "IBFL",
                        "SEARL", "SHFA"],
    "Fertilizers_Chemicals": ["COLG", "EFERT", "ENGROH", "FATIMA", "FFC", "GHGL",
                              "LCI", "LOTCHEM", "PABC", "EPCL"],
    "Consumer_Autos": ["ATLH", "BNWM", "FFL", "GADT", "GAL", "GHNI", "HCAR", "HUMNL",
                       "ILP", "INDU", "JDWS", "JVDC", "KTML", "MEHT", "MTL", "MUREB",
                       "NATF", "NESTLE", "NML", "PAEL", "PAKT", "PGLC", "PIBTL",
                       "PKGS", "PSEL", "RMPL", "SAZEW", "SRVI", "SSOM", "TGL",
                       "UNITY", "UPFL", "YOUW"],
}
TICKER_TO_SECTOR = {t: s for s, ts in SECTOR_MAP.items() for t in ts}


def technical_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]

    sma20 = ta.trend.sma_indicator(close, window=20)
    sma50 = ta.trend.sma_indicator(close, window=50)
    sma200 = ta.trend.sma_indicator(close, window=200)
    df["Close_to_SMA_20"] = close / sma20
    df["Close_to_SMA_50"] = close / sma50
    df["Close_to_SMA_200"] = close / sma200

    df["RSI_14"] = ta.momentum.rsi(close, window=14)
    df["MACD_Diff"] = ta.trend.MACD(close).macd_diff()

    for d in [1, 3, 7, 14]:
        df[f"Log_Return_{d}d"] = np.log(close / close.shift(d))

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_Width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / sma20

    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], close, window=14).average_true_range()
    df["ATR_Normalized"] = atr / close

    df["Volatility_30d"] = df["Log_Return_1d"].rolling(30).std()

    sma_vol20 = df["Volume"].rolling(20).mean()
    df["Volume_Surge"] = df["Volume"] / sma_vol20.replace(0, 1)

    obv = ta.volume.OnBalanceVolumeIndicator(close, df["Volume"]).on_balance_volume()
    obv_ma20 = obv.rolling(20).mean().replace(0, 1)
    df["OBV_Momentum"] = obv / obv_ma20

    for h in HORIZONS:
        df[f"Target_{h}d"] = (close.shift(-h) - close) / close

    df = df.drop(columns=["Open", "High", "Low", "Close", "Volume"], errors="ignore")
    feature_cols = [c for c in df.columns if not c.startswith("Target_")]
    df = df.dropna(subset=feature_cols)
    return df


TV_MACRO_SYMBOLS = {
    "PKR=X": "FX_IDC:USDPKR",
    "CL=F": "TVC:USOIL",
    "^IXIC": "NASDAQ:IXIC",
}


def macro_proxy(ticker: str, prefix: str, start: str, end: str) -> pd.DataFrame:
    print(f"  Fetching {ticker} for {prefix}_* macro proxy ...")
    tv_symbol = TV_MACRO_SYMBOLS.get(ticker)
    if tv_symbol:
        try:
            from tradingview_fetch import fetch_bars
            df_tv = fetch_bars(tv_symbol, n_bars=3000)
            if not df_tv.empty:
                close = df_tv["Close"]
                out = pd.DataFrame(index=df_tv.index)
                out[f"{prefix}_Log_Return"] = np.log(close / close.shift(1))
                out[f"{prefix}_SMA_200_Ratio"] = close / ta.trend.sma_indicator(close, window=200)
                out[f"{prefix}_Volatility_30d"] = out[f"{prefix}_Log_Return"].rolling(30).std()
                out.index = pd.to_datetime(out.index).normalize()
                return out
        except Exception as e:
            print(f"  TradingView fetch for {tv_symbol} failed ({e}); falling back to yfinance {ticker}...")

    import yfinance as yf
    raw = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    close = raw["Close"].squeeze() if isinstance(raw["Close"], pd.DataFrame) else raw["Close"]
    out = pd.DataFrame(index=raw.index)
    out[f"{prefix}_Log_Return"] = np.log(close / close.shift(1))
    out[f"{prefix}_SMA_200_Ratio"] = close / ta.trend.sma_indicator(close, window=200)
    out[f"{prefix}_Volatility_30d"] = out[f"{prefix}_Log_Return"].rolling(30).std()
    out.index = pd.to_datetime(out.index).normalize()
    return out



def run():
    tickers = sorted(p.stem for p in RAW_DIR.glob("*.csv"))
    print(f"PSX feature engineering: {len(tickers)} tickers from {RAW_DIR}")

    start_date = "2014-01-01"
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    pkr = macro_proxy("PKR=X", "Currency", start_date, end_date)
    oil = macro_proxy("CL=F", "Oil", start_date, end_date)
    nasdaq = macro_proxy("^IXIC", "Tech", start_date, end_date)

    progress = StageProgress("PSX feature engineering", len(tickers))
    ok, failed, skipped = [], [], []
    for ticker in tickers:
        sector = TICKER_TO_SECTOR.get(ticker)
        if sector is None:
            progress.step(f"{ticker}: SKIPPED (no sector mapping)")
            skipped.append(ticker)
            continue
        try:
            df = pd.read_csv(RAW_DIR / f"{ticker}.csv", index_col="Date", parse_dates=True).sort_index()
            df = technical_features(df)

            df = df.join(pkr, how="left")
            if sector == "Energy_Power":
                df = df.join(oil, how="left")
            elif sector == "Tech_Telecom":
                df = df.join(nasdaq, how="left")
            df = df.ffill()
            df = df.dropna(subset=["Currency_SMA_200_Ratio"])

            if len(df) < 200:
                progress.step(f"{ticker}: SKIPPED ({len(df)} rows after cleaning)")
                skipped.append(ticker)
                continue

            sector_dir = OUT_DIR / sector
            sector_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(sector_dir / f"{ticker}.csv")
            progress.step(f"{ticker} [{sector}]: {len(df)} rows, {df.shape[1]} columns")
            ok.append(ticker)
        except Exception as e:
            progress.step(f"{ticker}: FAILED ({e})")
            failed.append(ticker)
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
