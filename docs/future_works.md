# Future Works: FundForge Phase 2

With the core algorithmic forecasting engine and sentiment pipeline fully operational, the project will now transition from "predictive modeling" to **"financial simulation and autonomous distribution."**

This document outlines the exact implementation steps for the next two major paths of the FundForge system.

---

## Path 1: The Profit & Loss Simulator (Backtesting Engine)

**Objective:** Prove the financial viability of the 378-model ensemble. We know the models have high mathematical accuracy (MAE / Directional Accuracy), but we need to translate those metrics into Wall Street standard metrics: ROI, Sharpe Ratio, and Maximum Drawdown.

### Implementation Steps

#### Step 1: Historical Signal Generation
*   Create a script (`stage6_signal_generator.py`) that walks through the last 2-3 years of historical test data day-by-day.
*   For each day, it passes the historical features into the Dynamic Ensemble Inference engine to generate a "historical prediction."
*   Save these daily predictions alongside the actual closing prices in a `signals.csv` file.

#### Step 2: Define the Trading Strategy Rules
*   Establish strict entry and exit logic. For example:
    *   **Long Entry:** If the 7-day model predicts a > 2% price increase with > 70% confidence.
    *   **Short Entry:** If the 7-day model predicts a < -2% price decrease with > 70% confidence.
    *   **Position Sizing:** Allocate capital dynamically based on the ensemble's "Confidence Score."

#### Step 3: Backtesting Framework Integration
*   Install a high-performance vector-based backtesting library like `vectorbt` or `backtrader`.
*   Feed the `signals.csv` into the backtester.
*   Apply realistic market constraints: **0.1% transaction fees** per trade and **0.05% slippage** to ensure the simulation isn't overly optimistic.

#### Step 4: Metric Calculation & Dashboard UI
*   Calculate the final performance metrics: Total PnL (Profit & Loss), Win Rate, Sharpe Ratio, and Maximum Drawdown.
*   Update `app.py` to include a new **"Backtest & Strategy"** tab. This tab will visualize the AI's "Equity Curve" (how a $10,000 portfolio would have grown over time using the AI's trades compared to simply "holding" the commodity).

---

## Path 2: Automation & "Robo-Advising" (Email Subscriptions)

**Objective:** Transform FundForge from an interactive dashboard into a proactive, autonomous Robo-Advisor. Users will be able to subscribe via the Streamlit UI to receive comprehensive, AI-written daily financial reports straight to their inbox.

### Implementation Steps

#### Step 1: Streamlit Email Capture & Database
*   Update `app.py` to include an aesthetically pleasing "Subscribe to Daily Insights" box in the sidebar or on the main page.
*   Implement a lightweight database connection (using SQLite locally, or a cloud DB like Supabase/Firebase) to securely store user emails.
*   Script: `db_manager.py` to handle `add_subscriber()` and `get_all_subscribers()`.

#### Step 2: The LLM Report Generator
*   It is not enough to just send an email saying "Gold goes UP." The report must feel premium and institutional.
*   Create `stage7_report_generator.py`. This script will:
    1. Run the ensemble inference to get the target price and direction.
    2. Run the SHAP explainer to get the top 3 market drivers.
    3. Feed these raw JSON metrics into a Large Language Model (like GPT-4o or Claude 3.5 Sonnet) via API.
    4. Prompt the LLM to write a 3-paragraph, human-readable financial newsletter. (e.g., *"Gold is projected to rise 3% this week, largely driven by a massive spike in bearish sentiment surrounding CPI inflation reports..."*)

#### Step 3: Email Dispatch Integration
*   Integrate an email delivery API (such as SendGrid, Resend, or standard Python `smtplib` using an App Password).
*   Create a clean, responsive HTML email template featuring the FundForge logo, the multi-horizon chart image, and the LLM-generated text.

#### Step 4: The Daily Cron Scheduler
*   Set up an automated trigger (Linux `cron` job, or GitHub Actions).
*   Schedule it to execute every weekday at **4:30 PM EST** (shortly after the US markets close).
*   The automated pipeline will run in this exact order:
    `Scrape Live Data` → `Merge Features` → `Run Ensemble` → `Generate LLM Report` → `Fetch Subscriber List` → `Send Emails`.

---

## Path 3: Architectural Fixes & System Hardening

**Objective:** Address the critical vulnerabilities and technical debt currently hiding in the FundForge architecture to upgrade the system from "highly advanced" to "institutional-grade."

### 1. Data Quality: The "NewsAPI" Flaw
*   **The Problem:** We are scraping consumer-level headlines (Yahoo Finance, CNBC) via NewsAPI. Institutional algorithms react to raw Reuters/Bloomberg terminals in milliseconds. Relying on consumer headlines makes us vulnerable to delayed, "clickbait" sentiment.
*   **The Fix:** Upgrade the data pipeline to ingest direct financial data feeds (e.g., AlphaVantage News API, Benzinga) or the X/Twitter API for real-time geopolitical OSINT.

### 2. Static Weights in a Dynamic World (No Online Learning)
*   **The Problem:** The 378 models were trained offline on a static CSV. If the market undergoes a massive regime shift tomorrow (e.g., sudden global recession), our models will keep trading based on the previous year's patterns.
*   **The Fix:** Implement **Online Learning** (Rolling Window Retraining). Set up a pipeline where the models incrementally retrain themselves at the end of every week using the freshest data, dynamically "forgetting" obsolete macro regimes.

### 3. The Danger of the "Feature Router" (Ensemble Disagreement)
*   **The Problem:** The Champion vs. Challenger protocol reverts degraded models back to 27 features (ignoring sentiment), while keeping good models on 36 features (using sentiment). Our final prediction averages models that are observing two different realities.
*   **The Fix:** Replace the simple weighted average with a **Meta-Model** (e.g., Logistic Regression or a gating network) trained specifically to decide *which* underlying model to trust on any given day based on current market volatility, rather than blindly averaging them.

### 4. The Weekend Decay Assumption
*   **The Problem:** We use a simple `0.95` weekend decay factor for Sentiment EMAs, assuming news fades when markets are closed. However, geopolitical crises often happen on weekends, causing massive "Monday Gap-Ups."
*   **The Fix:** Engineer a specific **"Weekend Shock Feature"** that measures the delta between Friday close sentiment and Sunday night sentiment, properly pricing in weekend Black Swan events.

### 5. Predicting a Number vs. Predicting Risk
*   **The Problem:** The system currently outputs a specific target price. In quantitative finance, point predictions carry a terrible risk-reward ratio without a known distribution.
*   **The Fix:** Upgrade the LightGBM and Deep Learning models to utilize **Quantile Regression**. Instead of predicting one price, they should predict the 10th percentile (worst case/stop-loss), 50th percentile (expected), and 90th percentile (best case) to provide true Confidence Intervals.

---

## Priority & Next Actions
Paths 1 (Backtesting) and 2 (Robo-Advising) represent feature expansions, while Path 3 represents core stability. They can be worked on concurrently. When you are ready to begin, we will likely kick off Path 2 by creating the Database and Streamlit Email Input Box!
