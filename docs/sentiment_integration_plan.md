# Comprehensive Multi-Horizon Sentiment Integration Strategy & Execution Guide

Integrating sentiment into a 432-model multi-horizon forecasting system is the final major upgrade for FundForge. Because unstructured text can introduce fatal flaws (like look-ahead bias and entity-inversion), this process requires strict engineering.

This document serves as both the architectural strategy and the **Step-by-Step Execution Guide** for completing this integration end-to-end.

---

## Part 1: The Architectural Strategy & Edge Cases

Before writing code, we must architect around the four critical pitfalls of NLP in finance:

1.  **The "Entity Inversion" Problem:** A headline reading *"US Dollar Surges"* is scored as **Bullish** by NLP models, but a strong Dollar is **Bearish** for Gold. We must explicitly invert the sentiment score (`score * -1`) for negatively correlated macro news before feeding it to the commodity model.
2.  **Time-of-Publication Look-Ahead Bias:** News published after 4:00 PM EST cannot be used to predict today's closing price. We must strictly shift after-hours news to the **next trading day's** row.
3.  **Handling "No News" Days:** Days with 0 articles cannot be scored `0.0` (which implies "Neutral"). We must **forward-fill** the last known sentiment score (with a slight time-decay) because market psychology carries over until new information breaks.
4.  **Horizon-Aware Engineering:** A daily news spike matters for a 1-day forecast but is noise for a 120-day forecast. We must engineer short-term spikes (e.g., `Sentiment_1d`) and long-term EMAs (e.g., `Sentiment_EMA_60d`).

---

## Part 2: End-to-End Execution Guide

We will execute this integration in **seven sequential steps**, utilizing a **Hybrid Data Strategy** to avoid the crippling costs of deep historical APIs.

### Step 1: Historical Data Collection (For Training Only)
**Goal:** Gather ~10 years of financial headlines to teach the models historical correlations without paying thousands of dollars for API access.
*   **Action:** Create the script `stage4_historical_scraper.py`.
*   **Dataset Targets:** We will programmatically download two specific, highly vetted open-source datasets via the HuggingFace `datasets` library:
    1.  `"zeroshot/twitter-financial-news-sentiment"` (For fast-moving retail sentiment).
    2.  `"financial_phrasebank"` (For institutional news).
    *(Alternative: Download the "Daily Financial News for 6000+ Stocks" dataset from Kaggle which contains 2M+ articles from 2009-2020).*
*   **Code Implementation Logic:**
    1.  **Load Data:** Use `pandas` to load the massive offline CSV/Parquet file into memory.
    2.  **Strict Regex Filtering:** Create a dictionary of regex keywords per commodity to ensure zero cross-contamination.
        *   `Gold: r'\b(gold|xau|bullion|fomc|fed rate)\b'`
        *   `Oil: r'\b(crude|oil|wti|brent|opec|eia)\b'`
        *   `Wheat: r'\b(wheat|usda|crop|grain)\b'`
    3.  **Extraction:** Apply the regex filter. Extract the `date`, `publisher` (if available), and `headline` columns.
*   **Output:** Six clean, raw text CSVs (e.g., `data/gold_news_historical.csv`) containing thousands of rows of exact, filtered headlines.

### Step 2: Live API Setup (For Dashboard Inference Only)
**Goal:** Fetch up-to-the-minute news so the live Streamlit dashboard can predict tomorrow's price.
*   **Action:** Create the script `stage4_live_api_collector.py`.
*   **The Tool:** We will use **NewsAPI** (`newsapi.org`). It allows 100 free requests per day, and can look back 30 days on the free tier (which is all the dashboard needs).
*   **Code Implementation Logic:**
    1.  **API Connection:** `import requests` and pass the `API_KEY` via headers: `{"X-Api-Key": "YOUR_KEY"}`.
    2.  **The GET Request:** 
        ```python
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": "(crude OR oil OR WTI) AND (OPEC OR supply OR prices)",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 100
        }
        response = requests.get(url, params=params)
        ```
    3.  **JSON Parsing:** Loop through `response.json()['articles']`.
    4.  **Data Extraction:** Extract `article['publishedAt']` (string), `article['title']` (headline), and `article['source']['name']` (publisher).
    5.  **Timezone Standardization:** Use `datetime.fromisoformat()` to convert the `publishedAt` string into a strict UTC timezone object.
*   **Execution:** This script will be called inside `app.py` or a cron job. It appends today's news to `data/live_news.csv` every time the dashboard is refreshed.

### Step 3: NLP Scoring (FinBERT)
**Goal:** Convert text headlines into numerical probabilities.
*   **Action:** Create `stage4_sentiment_scoring.py`.
*   **Strategy:** Load the `ProsusAI/finbert` model via HuggingFace `transformers`. For every headline in both historical and live CSVs, we run inference to extract `P(Bullish)` and `P(Bearish)`.
*   **Calculation:** Calculate Net Sentiment Score: `NSS = P(Bullish) - P(Bearish)`.
*   **Correction:** Apply the "Entity Inversion" logic here (e.g., if the article is about the Dollar Index, multiply the NSS by -1 for Gold/Silver).
*   **Output:** Scored numerical datasets.

### Step 4: Horizon-Aware Feature Engineering
**Goal:** Aggregate daily scores and create temporal features for all 8 horizons.
*   **Action:** Create `stage4_sentiment_features.py`.
*   **Strategy:** 
    1.  Align all news by strict trading hours (shift >4:00 PM news to tomorrow).
    2.  Group by `Date` and calculate the volume-weighted average `Daily_NSS` and `News_Volume`.
    3.  Forward-fill missing days (with a 0.95 daily decay factor).
    4.  Calculate Short-Term features: `Sentiment_Spike` (Daily - 7d avg).
    5.  Calculate Long-Term features: `Sentiment_EMA_30d`, `Sentiment_EMA_60d`.
    6.  Calculate Interaction feature: `Sentiment_x_Trend` (`Daily_NSS * Close_to_MA_50`).
*   **Output:** Six final sentiment feature tables (e.g., `gold_sentiment_features.csv`).

### Step 5: Pipeline Merging
**Goal:** Safely inject the new features into the existing ML pipeline.
*   **Action:** Update `stage3_macro_merge.py`.
*   **Strategy:** Perform a strict pandas `merge(how='left', on='Date')` combining the existing `data/gold.csv` with `gold_sentiment_features.csv`. 
*   **Output:** The final, enriched `data/*.csv` files, containing OHLCV, Technicals, Macro, AND Sentiment.

### Step 6: The "Champion vs. Challenger" Retraining Loop
**Goal:** Retrain all 432 models with the new data while aggressively preventing overfitting.
*   **Action:** Run existing `create_ml_nb*.py` and `create_dl_nb*.py` scripts.
*   **Crucial Hyperparameter Tweaks:** 
    *   **Tree Models (XGB/LGBM):** Increase L2 Regularization (`reg_lambda`) and slightly lower `max_depth` to force general trend learning over spike memorization.
    *   **Deep Learning (LSTM/TFT):** Increase `Dropout` by 0.05 so networks don't over-rely on the sentiment nodes.
*   **Benchmarking (The Golden Rule):** We compare the new Challenger models against the current Champions. If sentiment *degrades* the performance of the 120d horizons (because it acted as noise), we delete the sentiment-aware 120d models and **keep the Macro-Only Champions**. We do not force sentiment where it doesn't belong.

### Step 7: UI & Explainability Integration
**Goal:** Show the user exactly why the AI made its decision.
*   **Action:** Update `app.py` and `ensemble_inference.py`.
*   **Strategy:** Delete the hardcoded heuristic rules currently driving the "Primary Market Drivers" UI. Replace them with a live `shap.TreeExplainer`, which outputs the exact top 3 features driving the forecast. 
*   **Result:** The UI will dynamically say things like: *"Bullish Driver: Sentiment_Spike (+0.8)"*. Furthermore, because we extract and save the `URL` in Step 2, if SHAP detects that a specific news event drove the sentiment, the dashboard will dynamically display the exact headline as a **clickable hyperlink** so the user can read the source article. This provides 100% mathematical and contextual transparency to the end user.

---

## Part 3: Advanced Alpha Generation (Hedge-Fund Grade Enhancements)

If the goal is to make FundForge a truly elite, predictive engine, a basic "Net Sentiment" score is not enough. To squeeze the absolute maximum predictive performance (alpha) out of the news, we will implement these three advanced architectural upgrades during **Step 3 (Feature Engineering)** and **Step 4 (Scoring)**:

### 1. Asymmetric Sentiment Separation (Fear > Greed)
**The Flaw:** Currently, the plan calculates a Net Score (`NSS = Bullish - Bearish`). This assumes a +0.8 Bullish news day is the exact opposite of a -0.8 Bearish news day. However, financial markets suffer from *Loss Aversion*—they react 3x faster and more violently to bad news (fear) than good news (greed). 
**The Upgrade:** Instead of collapsing them into one number, we will feed them into the ML models as **two separate columns**: `Daily_Bullish_Score` and `Daily_Bearish_Score`. This allows XGBoost and LSTM to mathematically learn the asymmetry (e.g., learning that a 0.8 Bear score triggers a -5% crash, but a 0.8 Bull score only triggers a +2% rally).

### 2. Topic-Based Decomposition (Zero-Shot Classification)
**The Flaw:** Averaging all news into one "Daily Score" destroys nuance. If there is hyper-bullish news about an Oil supply shortage, but hyper-bearish news about a global recession, the average score is `0.0` (Neutral). The model learns nothing.
**The Upgrade:** Before scoring with FinBERT, we will run the headlines through a Zero-Shot Classifier (like `facebook/bart-large-mnli`) to tag the *Topic*. We will create three separate sentiment columns per commodity:
*   `Sentiment_Supply` (e.g., OPEC cuts, mine closures)
*   `Sentiment_Macro` (e.g., interest rates, inflation)
*   `Sentiment_Geopolitics` (e.g., wars, trade tariffs)
By separating these, the AI can learn that a supply shock overrides macro weakness in the short term.

### 3. Source Credibility Weighting
**The Flaw:** A headline from a random crypto blog is treated with the exact same mathematical weight as a breaking headline from Bloomberg or Reuters.
**The Upgrade:** We will apply a `Source_Weight` multiplier to the FinBERT scores. 
*   Tier 1 (Reuters, Bloomberg, WSJ, Fed Press Releases) = `1.5x` weight.
*   Tier 2 (CNBC, Yahoo Finance, generic news) = `1.0x` weight.
*   Tier 3 (Unknown/Blogs) = `0.5x` weight.
This ensures the model pays maximum attention to market-moving sources and ignores clickbait.

---

## Part 4: Visual Data Alignment Guide (Translating News to Math)

Combining synchronous data (OHLCV) with asynchronous data (News) is one of the hardest parts of this pipeline. OHLCV data is perfectly neat—one row per trading day. News data is chaotic—you might have 5 articles on Monday, 0 on Tuesday, and 12 on Sunday when the markets are closed.

Machine learning models require a strict, flat table: 1 row = 1 day. Here is exactly how we convert the chaos of news into perfectly aligned columns.

### Step 1: What We Start With
Imagine we are looking at Crude Oil. We have two completely separate datasets.

**Dataset 1: Existing `crude_oil.csv` (Perfectly structured)**
| Date | Close | RSI_14 | USD_Index | EIA_Inventory |
| :--- | :--- | :--- | :--- | :--- |
| **Friday (Oct 6)** | $82.50 | 45.2 | 106.1 | 421M |
| **Monday (Oct 9)** | $86.00 | 52.1 | 106.4 | 421M |
*(Notice there is no Saturday or Sunday because markets are closed).*

**Dataset 2: Raw News Data (Chaotic and messy)**
| Date | Time | Headline |
| :--- | :--- | :--- |
| Friday | 10:00 AM | *"US rig counts fall slightly"* |
| Saturday | 2:00 PM | *"Hamas attacks Israel"* |
| Saturday | 6:00 PM | *"Middle East tensions rise"* |
| Sunday | 9:00 AM | *"OPEC calls emergency meeting"* |
| Monday | 08:00 AM | *"Oil prices expected to surge"* |

### Step 2: Scoring & Aggregating the News
We cannot join the raw text to the prices. First, FinBERT scores every headline from -1.0 (Bearish) to +1.0 (Bullish). Then, we group all the articles by **Calendar Day** and take the average score:

| Calendar Date | Article Count | Average Sentiment Score |
| :--- | :--- | :--- |
| Friday | 1 article | `-0.2` (Slightly Bearish) |
| Saturday | 2 articles | `+0.8` (Highly Bullish for Oil) |
| Sunday | 1 article | `+0.9` (Highly Bullish for Oil) |
| Monday | 1 article | `+0.6` (Bullish for Oil) |

### Step 3: Handling Weekends & Cutoffs (The Crucial Fix)
If we join this directly to the OHLCV data, Saturday and Sunday will disappear because they aren't trading days. To fix this, we apply the **"Next Available Trading Day"** rule.
*   Friday's news happened during Friday's trading session ➔ Belongs to Friday.
*   Saturday's news happened when markets were closed ➔ Pushed to Monday.
*   Sunday's news happened when markets were closed ➔ Pushed to Monday.
*   Monday's news (pre-market) ➔ Belongs to Monday.

We combine all the news for Saturday, Sunday, and Monday morning into a single **Monday Score**.

**Final Cleaned News Table:**
| Trading Date | News_Volume | Sentiment_Score |
| :--- | :--- | :--- |
| **Friday (Oct 6)** | 1 | `-0.20` |
| **Monday (Oct 9)** | 4 | `+0.77` *(Avg of Sat+Sun+Mon)* |

### Step 4: The Final Merge
Now, both datasets speak the exact same language: **One row per Trading Day**.
We perform a simple Python pandas `merge(on='Date')`. The new sentiment numbers become extra columns at the end of the existing dataset!

**The Final Training Dataset:**
| Date | Close | RSI_14 | USD_Index | **News_Volume** | **Sentiment_Score** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Friday | $82.50 | 45.2 | 106.1 | **1** | **-0.20** |
| Monday | $86.00 | 52.1 | 106.4 | **4** | **+0.77** |

When you feed this final row into XGBoost or LSTM, the model doesn't know what "News" is. It just sees that on Monday, the `News_Volume` spiked to 4, and the `Sentiment_Score` spiked to +0.77, and it will mathematically correlate those two numbers to the price jumping to $86.00.
