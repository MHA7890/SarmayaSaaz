"""
Crypto feature engineering: data-new/crypto-data/*.csv (accurate Binance
OHLCV) -> data-ready/crypto/*.csv (full feature set, ready for training).

Reproduces src/crypto/stage1_data_collection.py (technical + SMC features),
stage2_target_engineering.py (targets + swing-distance ratios),
stage3_macro_engineering.py (Fear&Greed / S&P500 / BTC-vol macro block) and
stage3b_sentiment_engineering.py (FinBERT news sentiment) against the new,
accurate price data. Macro/sentiment source data is the same cached data the
original pipeline used (data/crypto_raw/*) - it isn't OHLCV, so it wasn't
part of what data-new/ was collected to fix; only its FinBERT scoring is
(re)computed here from headlines already present on disk.

Run:
    uv run python training-scripts-new/crypto/01_feature_engineering.py
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
from common.progress import StageProgress  # noqa: E402

RAW_DIR = ROOT / "data-new" / "crypto-data"
OUT_DIR = ROOT / "data-ready" / "crypto"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = ROOT / "data-ready" / "crypto" / "_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CRYPTO_RAW = ROOT / "data" / "crypto_raw"
HORIZONS = [7, 14, 28, 42, 60, 90, 120]
SENTIMENT_CACHE = CACHE_DIR / "sentiment_daily.csv"


# --------------------------------------------------------------------------
# Stage 1 equivalent: technical + SMC features from raw OHLCV
# --------------------------------------------------------------------------
def technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    for p in [1, 7, 14, 30, 60, 90, 120]:
        df[f"Log_Return_{p}d"] = np.log(df["Close"] / df["Close"].shift(p))

    df["RSI_14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    df["MACD_Hist"] = ta.trend.MACD(df["Close"]).macd_diff()

    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
    df["ATR_14_Pct"] = atr / df["Close"]

    bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["BB_Width"] = bb.bollinger_wband()

    df["Volatility_30d"] = df["Log_Return_1d"].rolling(30).std() * np.sqrt(365)

    for w in [20, 50, 200]:
        sma = ta.trend.SMAIndicator(df["Close"], window=w).sma_indicator()
        df[f"Close_to_SMA{w}"] = (df["Close"] - sma) / sma

    df["Volume_Change_1d"] = df["Volume"].pct_change()
    return df


def smc_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Bullish_FVG"] = (df["Low"] > df["High"].shift(2)) & (df["Close"].shift(1) > df["Open"].shift(1))
    df["Bearish_FVG"] = (df["High"] < df["Low"].shift(2)) & (df["Close"].shift(1) < df["Open"].shift(1))

    df["Swing_High"] = (
        (df["High"].shift(3) > df["High"].shift(4))
        & (df["High"].shift(3) > df["High"].shift(5))
        & (df["High"].shift(3) > df["High"].shift(6))
        & (df["High"].shift(3) > df["High"].shift(2))
        & (df["High"].shift(3) > df["High"].shift(1))
        & (df["High"].shift(3) > df["High"])
    )
    df["Swing_Low"] = (
        (df["Low"].shift(3) < df["Low"].shift(4))
        & (df["Low"].shift(3) < df["Low"].shift(5))
        & (df["Low"].shift(3) < df["Low"].shift(6))
        & (df["Low"].shift(3) < df["Low"].shift(2))
        & (df["Low"].shift(3) < df["Low"].shift(1))
        & (df["Low"].shift(3) < df["Low"])
    )

    df["Last_Swing_High_Price"] = np.where(df["Swing_High"], df["High"].shift(3), np.nan)
    df["Last_Swing_High_Price"] = df["Last_Swing_High_Price"].ffill()
    df["Last_Swing_Low_Price"] = np.where(df["Swing_Low"], df["Low"].shift(3), np.nan)
    df["Last_Swing_Low_Price"] = df["Last_Swing_Low_Price"].ffill()

    df["Bullish_BOS"] = (df["Close"] > df["Last_Swing_High_Price"]) & (
        df["Close"].shift(1) <= df["Last_Swing_High_Price"]
    )
    df["Bearish_BOS"] = (df["Close"] < df["Last_Swing_Low_Price"]) & (
        df["Close"].shift(1) >= df["Last_Swing_Low_Price"]
    )
    return df


# --------------------------------------------------------------------------
# Stage 2 equivalent: targets + swing-distance ratios
# --------------------------------------------------------------------------
def targets_and_ratios(df: pd.DataFrame) -> pd.DataFrame:
    for h in HORIZONS:
        df[f"Target_{h}d"] = (df["Close"].shift(-h) - df["Close"]) / df["Close"]

    safe_high = df["Last_Swing_High_Price"].replace(0, np.nan)
    safe_low = df["Last_Swing_Low_Price"].replace(0, np.nan)
    df["Dist_to_Swing_High"] = (df["Close"] - safe_high) / safe_high
    df["Dist_to_Swing_Low"] = (df["Close"] - safe_low) / safe_low

    # "Close" is dropped alongside the other raw prices, matching the original
    # pipeline's explicit rule (docs/crypto_project_tracker.md, Phase 2:
    # "purged all absolute dollar values (Close, Volume)").
    #
    # This matters more here than anywhere else in the project because crypto
    # models are trained POOLED across a cluster, and a single cluster spans
    # price scales of up to ~4x10^8 (e.g. MKR at $1,813 beside a memecoin at
    # $0.000004). Under a shared StandardScaler an absolute Close is both
    # non-stationary - BTC at $30k in training vs $64k at inference is out of
    # distribution - and an implicit asset-identity label: it alone tells the
    # model which coin a row belongs to, letting it memorise per-asset levels
    # instead of learning the transferable structure clustering exists to
    # capture. Every price-derived signal the models need is already present
    # in stationary ratio form (Close_to_SMA*, Dist_to_Swing_*, Log_Return_*).
    df = df.drop(columns=["Open", "High", "Low", "Close", "Volume",
                          "Last_Swing_High_Price", "Last_Swing_Low_Price"],
                 errors="ignore")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


# --------------------------------------------------------------------------
# Stage 3 equivalent: macro block (Fear&Greed / S&P500 / BTC-vol)
# --------------------------------------------------------------------------
def load_fear_greed() -> pd.Series:
    path = CRYPTO_RAW / "fear_greed_index.csv"
    if not path.exists():
        print(f"  ! {path} not found - Fear_Greed_Score will be filled with the neutral value (50)")
        return pd.Series(dtype=float)
    fg = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")["FearGreed_Raw"].astype(float)
    fg.index = fg.index.normalize()
    return fg.rename("Fear_Greed_Score")


def fetch_sp500_return() -> pd.Series:
    print("  Fetching ^GSPC (S&P 500) for SP500_Return_7d ...")
    sp = yf.download("^GSPC", period="10y", interval="1d", progress=False)
    if isinstance(sp.columns, pd.MultiIndex):
        sp.columns = [c[0] for c in sp.columns]
    ret = np.log(sp["Close"] / sp["Close"].shift(7))
    ret.index = pd.to_datetime(ret.index).normalize()
    return ret.rename("SP500_Return_7d")


def compute_btc_volatility() -> pd.Series:
    btc_path = RAW_DIR / "BTC.csv"
    btc = pd.read_csv(btc_path, index_col="Date", parse_dates=True)
    log_ret = np.log(btc["Close"] / btc["Close"].shift(1))
    vol = log_ret.rolling(30).std() * np.sqrt(365)
    return vol.rename("BTC_Volatility_30d")


def apply_macro(df: pd.DataFrame, fear_greed: pd.Series, sp500: pd.Series, btc_vol: pd.Series) -> pd.DataFrame:
    df = df.join(fear_greed, how="left")
    df = df.join(sp500, how="left")
    df = df.join(btc_vol, how="left")
    df["SP500_Return_7d"] = df["SP500_Return_7d"].ffill()
    df["BTC_Volatility_30d"] = df["BTC_Volatility_30d"].ffill()
    df["Fear_Greed_Score"] = df["Fear_Greed_Score"].fillna(50.0)
    return df


# --------------------------------------------------------------------------
# Stage 3b equivalent: FinBERT news sentiment (scored once, cached, then
# merged into every asset)
# --------------------------------------------------------------------------
def score_sentiment_daily() -> pd.DataFrame:
    if SENTIMENT_CACHE.exists():
        print(f"  Using cached sentiment: {SENTIMENT_CACHE}")
        return pd.read_csv(SENTIMENT_CACHE, index_col="Date", parse_dates=True)

    news_path = CRYPTO_RAW / "coindesk_news.csv"
    if not news_path.exists():
        print(f"  ! {news_path} not found - sentiment columns will be zero-filled")
        return pd.DataFrame(columns=["Sentiment_Score", "News_Volume"])

    print("  Scoring crypto news headlines with FinBERT (one-time, cached after)...")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    news = pd.read_csv(news_path)
    news = news.dropna(subset=["Headline", "Date"])
    news["Date"] = pd.to_datetime(news["Date"], errors="coerce").dt.normalize()
    news = news.dropna(subset=["Date"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device).eval()

    headlines = news["Headline"].astype(str).tolist()
    batch_size = 128
    scores = np.empty(len(headlines), dtype=np.float32)

    from tqdm import tqdm

    with torch.no_grad():
        for i in tqdm(range(0, len(headlines), batch_size), desc="  FinBERT batches", unit="batch"):
            batch = headlines[i : i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            # FinBERT label order: 0=positive, 1=negative, 2=neutral
            scores[i : i + len(batch)] = probs[:, 0] - probs[:, 1]

    news["Score"] = scores
    daily = news.groupby("Date").agg(Sentiment_Score=("Score", "mean"), News_Volume=("Score", "count"))
    daily.index.name = "Date"
    daily.to_csv(SENTIMENT_CACHE)
    print(f"  Cached sentiment to {SENTIMENT_CACHE}")
    return daily


def apply_sentiment(df: pd.DataFrame, fear_greed_raw: pd.Series, sentiment_daily: pd.DataFrame) -> pd.DataFrame:
    full_range = pd.date_range(
        min(df.index.min(), sentiment_daily.index.min() if len(sentiment_daily) else df.index.min()),
        max(df.index.max(), sentiment_daily.index.max() if len(sentiment_daily) else df.index.max()),
        freq="D",
    )
    sent = sentiment_daily.reindex(full_range)
    sent[["Sentiment_Score", "News_Volume"]] = sent[["Sentiment_Score", "News_Volume"]].ffill().fillna(0.0)
    sent["Sentiment_EMA_7d"] = sent["Sentiment_Score"].ewm(span=7, adjust=False).mean()
    sent["Sentiment_EMA_30d"] = sent["Sentiment_Score"].ewm(span=30, adjust=False).mean()

    fg = fear_greed_raw.reindex(full_range).ffill().fillna(50.0)
    fg_norm = (fg - 50.0) / 50.0
    sent["FearGreed_Score"] = fg_norm
    sent["FearGreed_EMA_7d"] = fg_norm.ewm(span=7, adjust=False).mean()
    sent["FearGreed_EMA_30d"] = fg_norm.ewm(span=30, adjust=False).mean()

    cols = ["Sentiment_Score", "News_Volume", "Sentiment_EMA_7d", "Sentiment_EMA_30d",
            "FearGreed_Score", "FearGreed_EMA_7d", "FearGreed_EMA_30d"]
    df = df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")
    df = df.join(sent[cols], how="left")
    df[cols] = df[cols].ffill().fillna(0.0)
    return df


def run():
    tickers = sorted(p.stem for p in RAW_DIR.glob("*.csv"))
    print(f"Crypto feature engineering: {len(tickers)} tickers from {RAW_DIR}")

    print("Loading shared macro/sentiment sources (computed once, reused per ticker)...")
    fear_greed_raw = load_fear_greed()
    sp500 = fetch_sp500_return()
    btc_vol = compute_btc_volatility()
    sentiment_daily = score_sentiment_daily()

    progress = StageProgress("Crypto feature engineering", len(tickers))
    ok, failed, skipped = [], [], []
    for ticker in tickers:
        try:
            df = pd.read_csv(RAW_DIR / f"{ticker}.csv", index_col="Date", parse_dates=True)
            df = df.sort_index()
            df = technical_indicators(df)
            df = smc_features(df)
            df = targets_and_ratios(df)
            df = apply_macro(df, fear_greed_raw, sp500, btc_vol)
            df = apply_sentiment(df, fear_greed_raw, sentiment_daily)

            feature_cols = [c for c in df.columns if not c.startswith("Target_")]
            df = df.dropna(subset=feature_cols)

            if len(df) < 200:
                progress.step(f"{ticker}: SKIPPED ({len(df)} rows after cleaning)")
                skipped.append(ticker)
                continue

            df.to_csv(OUT_DIR / f"{ticker}.csv")
            progress.step(f"{ticker}: {len(df)} rows, {df.shape[1]} columns")
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
