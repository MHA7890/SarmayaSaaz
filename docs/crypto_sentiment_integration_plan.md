# Crypto Multi-Horizon Deep Sentiment Integration Plan

## Executive Summary
This document outlines the architectural strategy and step-by-step execution plan to upgrade the Crypto Forecasting Engine. We will transition from a rudimentary "post-processing kill-switch" to a **Deep Feature Integration** approach. By historically scoring years of crypto news with FinBERT and injecting it directly into the OHLCV datasets, the XGBoost and Deep Learning models will mathematically learn the exact correlations between global market sentiment and multi-horizon price action.

---

## 1. The Architectural Advantage (Crypto vs Commodities)
When we integrated sentiment into Commodities, we had to build complex "Weekend Rollover" logic because commodities don't trade on Saturdays and Sundays. 

**Crypto is a 24/7/365 market.** 
This is a massive mathematical advantage. We do not need to roll weekend news to Monday. A news event that happens on Saturday at 2:00 PM will directly impact the Saturday 11:59 PM Daily Close. We can map `Sentiment_Score` perfectly, 1-to-1, to the daily OHLCV rows.

---

## 2. Step-by-Step Execution Guide

### Phase 3.5a: Historical News Acquisition
**Goal:** Gather at least 3-4 years of daily cryptocurrency headlines to train the models on historical market cycles (Bull runs, FTX collapse, LUNA crash).
*   **Action:** Create `src/crypto/stage3a_historical_news.py`
*   **Execution:** We will leverage the HuggingFace `datasets` library to pull a massive open-source dataset (e.g., `ElKulako/cryptonews` or similar robust financial/crypto text datasets).
*   **Processing:** The script will filter the raw dataset for English headlines and strictly format it into a massive pandas DataFrame with two columns: `Date` (YYYY-MM-DD) and `Headline`.

### Phase 3.5b: FinBERT Historical Scoring & Engineering
**Goal:** Score the historical dataset using Institutional NLP and engineer temporal sentiment features.
*   **Action:** Create `src/crypto/stage3b_sentiment_engineering.py`
*   **Execution:**
    1.  Load `ProsusAI/finbert` onto the RTX 6000 Ada GPU.
    2.  Run batched inference across all historical headlines, calculating the Net Sentiment Score (`P(Bullish) - P(Bearish)`).
    3.  Group the data by `Date` to calculate the daily volume-weighted `Sentiment_Score` and `News_Volume`.
    4.  **Feature Engineering:** Calculate `Sentiment_EMA_7d` and `Sentiment_EMA_30d` to capture the "momentum" of the news cycle.
    5.  **Merge:** Iterate through all 29 files in `data/crypto_processed/`. Perform a Left Join on `Date` to inject the new sentiment features directly alongside the MACD, RSI, and BTC Volatility.

### Phase 4: The Great Retraining (Master Engine)
**Goal:** Force the 252 models to learn the mathematical weight of Sentiment.
*   **Action:** Run `src/crypto/stage4_master_training.py`
*   **Execution:** Because the dataset now has new columns (`Sentiment_Score`, `News_Volume`, etc.), the previously saved `.pt` and `.pkl` models are invalid (they expect 27 features, but will now receive ~31). 
*   **Impact:** The models will automatically adjust their internal weights. For example, XGBoost might learn that if `Target_7d` is usually positive when RSI > 50, a `Sentiment_Score` of `-0.80` (Extreme Panic) overrides the RSI and forces a downward prediction.

### Phase 5: Live API News Collector
**Goal:** Create a lightweight, bulletproof pipeline to fetch *today's* news for the live web server inference.
*   **Action:** Rewrite `src/crypto/stage5_sentiment_pipeline.py`
*   **Execution:** 
    1. Scrap the live Cointelegraph RSS feed (or similar reliable API).
    2. Score the last 24 hours of headlines with FinBERT.
    3. Calculate today's exact `Sentiment_Score` and `News_Volume`.
    4. Export this directly to `results/crypto/live_sentiment.json` for the Inference engine to consume.

### Phase 6: Deep Feature Inference Update
**Goal:** Feed the live sentiment into the live prediction matrix.
*   **Action:** Update `src/crypto/stage6_inference.py`
*   **Execution:** 
    1. Load `live_sentiment.json`.
    2. When pulling the live OHLCV and Technicals from `yfinance` to construct `X_today`, the script will append the `Sentiment_Score` to the exact end of the feature array.
    3. The array (now 31 features wide) is passed into the physically saved XGBoost and PyTorch models.
    4. The models output a purely mathematical target price—no heuristic "Kill Switches" required, because the models themselves *already know* how to react to the panic.

### Phase 7: Dashboard Transparency Upgrades
**Goal:** Prove to the user that the AI is using the news.
*   **Action:** Update `src/crypto/crypto_dashboard.py`
*   **Execution:** Re-add the FinBERT Gauge Chart to the UI, but this time label it correctly as a *Mathematical Input Feature* rather than an arbitrary override. If SHAP integration is desired, we can visually display exactly how much the `Sentiment_Score` altered the final dollar prediction.

---

## 3. Required Deliverables
1. `src/crypto/stage3a_historical_news.py`
2. `src/crypto/stage3b_sentiment_engineering.py`
3. Updated `src/crypto/stage5_sentiment_pipeline.py`
4. Updated `src/crypto/stage6_inference.py`
5. Updated `src/crypto/crypto_dashboard.py`
