# MUFAP Integration Plan & Dataset Overview

## 1. Plan Overview
This document outlines the structure of the MUFAP (Mutual Funds Association of Pakistan) dataset and our plan to integrate it into the FundForge pipeline.
Our goal is to forecast the future Net Asset Value (NAV) of these funds so that investors can make informed decisions. We will follow a similar architectural pipeline to our Crypto and Commodity engines (Data Collection -> Target Engineering -> Macro Injection -> ML Training -> Inference).

## 2. Dataset Columns
The following data points are available for feature engineering:
- **AMC**
- **FundID**
- **Sector**
- **FundType**
- **CategoryID**
- **Fund**
- **Category**
- **Inception_Date**
- **Offer**
- **Repurchase**
- **NAV**
- **Validity_Date**
- **Front_End_Load**
- **Back_End_Load**
- **Contingent_Load**
- **Market_Price**
- **Trustee**
- **Scrape_Time**

## 3. Architectural Grouping Strategy (Macro-Clustering)
With 275 distinct mutual funds, training individual models would lead to severe overfitting and system bloat. Because MUFAP explicitly categorizes these funds into 34 sub-categories (e.g., *Money Market*, *Islamic Equity*), we will employ a **Hybrid Macro-Driven Grouping Strategy**. 

We will compress the 34 sub-categories into **4 Major Super-Clusters**, allowing the AI to learn universal financial laws based on the underlying macro-economic driver of that specific group:

### Super-Cluster 1: The "Equity" Group (High Risk / High Reward)
*   **Includes:** Equity Funds, Shariah Compliant Equity, Dedicated Equity, Index Trackers.
*   **How it behaves:** These funds hold stocks and react aggressively to the Pakistan Stock Exchange.
*   **Macro Feature Injection:** We will feed the **PSX KSE-100 Index** directly into this cluster's training loop.

### Super-Cluster 2: The "Money Market" Group (Zero Risk / Constant Growth)
*   **Includes:** Money Market, Shariah Compliant Money Market, Cash Funds.
*   **How it behaves:** These funds hold cash and short-term bank deposits. Their NAV grows in a nearly perfectly straight line, dictated entirely by national interest rates.
*   **Macro Feature Injection:** We will feed the State Bank of Pakistan (SBP) **KIBOR Interest Rate**.

### Super-Cluster 3: The "Income & Debt" Group (Medium Risk)
*   **Includes:** Income Funds, VPS-Debt, Fixed Rate/Return, Aggressive Fixed Income.
*   **How it behaves:** These hold corporate and government bonds (T-Bills, PIBs). They are mostly stable but their NAV can fluctuate inversely with sudden interest rate spikes.
*   **Macro Feature Injection:** We will feed **Pakistan Bond Yields (PKRV rates)**.

### Super-Cluster 4: The "Commodity" Group 
*   **Includes:** Shariah Compliant Commodities / Gold.
*   **How it behaves:** These mutual funds hold physical Gold and track its global spot price.
*   **Macro Feature Injection:** We will reuse our highly-accurate `GC=F` (Gold) features from the existing Commodities Engine!

---

## 4. Implementation Phases (Progress Log)

### ✅ Phase 1: Data Cleaning & Target Engineering
*   **Status:** Completed
*   **Actions Taken:**
    *   Ingested the raw `MUFAP_Historical_NAV.csv` dataset (449,031 rows spanning August 2016 to August 2026).
    *   Identified and purged **72,129 corrupted rows** (e.g., mathematically impossible negative NAVs, astronomical Offer pricing glitches, and missing `Validity_Date` timestamps).
    *   Engineered the forward-looking target variables (`Target_7d`, `Target_14d`, `Target_28d`, `Target_42d`, `Target_60d`, `Target_90d`, `Target_120d`) for predictive modeling.
    *   Engineered the `Spread_Ratio` feature `(Offer - NAV) / NAV` to track hidden fund fees.
    *   Split the dataset into **272 pristine, individual CSV files** saved in `data/mufap/`.

### 🔄 Phase 2: Feature Engineering & The "Age Filter"
*   **Status:** Next Step
*   **Planned Actions:**
    *   **The 200-Day Age Filter:** Institutional ML models require extensive historical context to learn market cycles. Any mutual fund with less than 200 trading days of history will be mathematically dropped from the pipeline to protect the AI from overfitting on "young" assets.
    *   **Technical Indicators:** Calculate standard alpha-generators for each surviving fund:
        *   Moving Averages: SMA_20, SMA_50, SMA_200.
        *   Momentum: Log Returns over multiple horizons.
        *   Volatility: Historical rolling volatility.

### ✅ Phase 3: Macro-Economic Injection & Clustering
*   **Status:** Completed
*   **Actions Taken:**
    *   **Dynamic Routing:** The 198 mature funds were programmatically cross-referenced with the raw dataset and sorted into 5 targeted Super-Clusters based on their underlying asset classes.
    *   **Proxy Fetching:** Downloaded exactly 10 years of institutional macro data via `yfinance` to act as predictive proxies for the Pakistani economy:
        *   **PAK:** Global X MSCI Pakistan ETF (Proxy for PSX Stock Market).
        *   **PKR=X:** USD to PKR Exchange Rate (Proxy for SBP Interest Rates / Devaluation).
        *   **GC=F:** Global Gold Futures (Proxy for Commodities).
    *   **Feature Integration & Alignment:** Computed stationary macro indicators (`Log_Return`, `SMA_200_Ratio`, `Volatility_30d`) for these proxies and perfectly merged them into the mutual fund matrices using Forward-Fill (`.ffill()`) to bridge the gap between 5-day stock markets and 7-day mutual fund schedules.
    *   **Final Output:** The matrices are now securely saved in `data/mufap_clustered/` categorized precisely by cluster.

### ✅ Phase 4: Master Training (Global ML Models)
*   **Status:** Completed
*   **Actions Taken:**
    *   Trained 35 independent models (XGBoost, LightGBM, and PyTorch LSTM) across 5 Super-Clusters and 7 horizons.
    *   Implemented strict anti-overfitting mechanisms: 50-round early stopping, Huber Loss, L2 Weight Decay, and a global Chronological Purge Gap to prevent data leakage.
    *   Saved exact mathematically frozen `StandardScalers` for live inference.
    *   Saved the master routing map in `ensemble_weights.json` to track winning models and Validation MAE.

### ✅ Phase 5: The Inference Engine (Deployment)
*   **Status:** Completed
*   **Actions Taken:**
    *   Built the highly optimized `MUFAPPredictor` class.
    *   Enabled dynamic routing to load the exact winning model for any queried mutual fund.
    *   Automated live feature engineering, anti-leakage scaling, and mathematical translation of percentage predictions back into real-world Rupee NAV projections.

### ✅ Phase 6: PnL Simulator & Backtesting
*   **Status:** Completed
*   **Actions Taken:**
    *   Built the 365-day backtester using the "Cash Shield" logic (moving to cash during predicted downturns).
    *   Implemented realistic financial tracking by strictly deducting `Spread_Ratio` (Front-End Load) fees on every transaction.
    *   Mathematically verified that the AI models successfully operate, while also concluding that mutual funds require long-term (120d) holding to bypass heavy broker fees.

---

## 5. Available Mutual Funds
There are a total of **275** mutual funds in this dataset. Below is the complete list:

- ABL Cash Fund
- ABL Financial Sector Fund Plan I
- ABL Fixed Rate Plan XXIX
- ABL Fixed Rate Plan XXVII
- ABL Fixed Rate Plan XXVIII
- ABL GOB Islamic Pension Fund
- ABL GOB Pension Fund
- ABL GOKP Islamic Pension Fund
- ABL GOKP Pension Fund
- ABL GOPB Islamic Pension Fund
- ABL GOPB Pension Fund
- ABL Government Securities Fund
- ABL Income Fund
- ABL Islamic Asset Allocation Fund
- ABL Islamic Cash Fund
- ABL Islamic Dedicated Stock Fund
- ABL Islamic Financial Planning Fund (Active Allocation Plan)
- ABL Islamic Financial Planning Fund (Capital Preservation Plan I)
- ABL Islamic Financial Planning Fund (Conservative Allocation Plan)
- ABL Islamic Income Fund
- ABL Islamic Money Market Plan I
- ABL Islamic Pension Fund
- ABL Islamic Sovereign Plan I
- ABL Islamic Stock Fund
- ABL Money Market Plan I
- ABL Optimal Asset Allocation Fund
- ABL Pension Fund
- ABL Special Saving Fund (ABL Special Saving Plan I)
- ABL Special Saving Fund (ABL Special Saving Plan II)
- ABL Special Saving Fund (ABL Special Saving Plan III)
- ABL Special Saving Fund (ABL Special Saving Plan IV)
- ABL Special Saving Fund (ABL Special Saving Plan V)
- ABL Special Saving Fund (ABL Special Saving Plan VI)
- ABL Stock Fund
- AL Habib Asset Allocation Fund
- AL Habib Cash Fund
- AL Habib Fixed Return Fund Plan 19
- AL Habib Fixed Return Fund Plan 23
- AL Habib Fixed Return Fund Plan 28
- AL Habib Fixed Return Fund Plan 31
- AL Habib Fixed Return Fund Plan 32
- AL Habib Fixed Return Fund Plan 33
- AL Habib GOKP Islamic Pension Fund
- AL Habib GOKP Pension Fund
- AL Habib Government Securities Fund
- AL Habib Income Fund
- AL Habib Islamic Cash Fund
- AL Habib Islamic Income Fund
- AL Habib Islamic Money Market Fund
- AL Habib Islamic Munafa Fund Plan 8
- AL Habib Islamic Munafa Fund Plan 9
- AL Habib Islamic Pension Fund
- AL Habib Islamic Savings Fund
- AL Habib Islamic Stock Fund
- AL Habib Money Market Fund
- AL Habib Pension Fund
- AL Habib Sovereign Income Fund Plan 1
- AL Habib Sovereign Income Fund Plan 2
- AL Habib Sovereign Income Fund Plan 3
- AL Habib Stock Fund
- Al Ameen Islamic Aggressive Income Fund
- Al Ameen Islamic Aggressive Income Plan I
- Al Ameen Islamic Asset Allocation Fund
- Al Ameen Islamic Cash Fund
- Al Ameen Islamic Cash Plan I
- Al Ameen Islamic Energy Fund
- Al Ameen Islamic Income Fund
- Al Ameen Islamic Punjab Pension Fund
- Al Ameen Islamic Retirement Savings Fund
- Al Ameen Islamic Sovereign Fund
- Al Ameen Shariah Stock Fund
- Al Ameen Voluntary Pension Fund KPK
- Al Meezan Mutual Fund
- Alfalah Asset Allocation Fund
- Alfalah Balochistan Islamic Pension Fund
- Alfalah Balochistan Pension Fund
- Alfalah Cash Fund - II
- Alfalah Consumer Index Exchange Traded Fund
- Alfalah Financial Sector Income Fund
- Alfalah Financial Sector Opportunity Fund
- Alfalah Financial Value Fund (Alfalah Financial Value Plan I)
- Alfalah Financial Value Fund - II
- Alfalah GHP Alpha Fund
- Alfalah GHP Cash Fund
- Alfalah GHP Dedicated Equity Fund
- Alfalah GHP Income Fund
- Alfalah GHP Income Multiplier Fund
- Alfalah GHP Islamic Dedicated Equity Fund
- Alfalah GHP Islamic Income Fund
- Alfalah GHP Islamic Pension Fund
- Alfalah GHP Islamic Prosperity Planning Fund (Alfalah GHP Islamic Active Allocation Plan II)
- Alfalah GHP Islamic Prosperity Planning Fund (Alfalah GHP Islamic Balance Allocation Plan)
- Alfalah GHP Islamic Prosperity Planning Fund (Alfalah GHP Islamic Moderate Allocation Plan)
- Alfalah GHP Islamic Stock Fund
- Alfalah GHP Islamic Value Fund
- Alfalah GHP Money Market Fund
- Alfalah GHP Pension Fund
- Alfalah GHP Prosperity Planning Fund (Alfalah GHP Active Allocation Plan)
- Alfalah GHP Prosperity Planning Fund (Alfalah GHP Conservative Allocation Plan)
- Alfalah GHP Prosperity Planning Fund (Alfalah GHP Moderate Allocation Plan)
- Alfalah GHP Sovereign Fund
- Alfalah GHP Stock Fund
- Alfalah GHP Value Fund
- Alfalah GOPB Islamic Pension Fund
- Alfalah GOPB Pension Fund
- Alfalah Government Securities Fund - II
- Alfalah Government Securities Fund Plan I
- Alfalah Government Securities Fund Plan II
- Alfalah Income & Growth Fund
- Alfalah Islamic Amdani Fund
- Alfalah Islamic Asset Allocation Fund Plan I
- Alfalah Islamic Income Growth Fund
- Alfalah Islamic KPK Employee Pension Fund
- Alfalah Islamic Money Market Fund
- Alfalah Islamic Sovereign Fund (Alfalah Islamic Sovereign Plan I)
- Alfalah Islamic Sovereign Fund (Alfalah Islamic Sovereign Plan II)
- Alfalah Islamic Sovereign Fund (Alfalah Islamic Sovereign Plan III)
- Alfalah Islamic Stable Return Fund Plan XIX
- Alfalah KPK Employee Pension Fund
- Alfalah KTrade Islamic Plan VII
- Alfalah MTS Fund
- Alfalah Money Market Fund - II
- Alfalah Pension Fund - II
- Alfalah Savings Growth Fund
- Alfalah Special Savings Fund - I
- Alfalah Special Savings Fund - II
- Alfalah Stable Return Fund Plan XX
- Alfalah Stable Return Fund Plan XXII
- Alfalah Stable Return Fund Plan XXVI
- Alfalah Stable Return Fund Plan XXVII
- Alfalah Stable Return Fund Plan XXVIII
- Alfalah Stable Return Fund Plan XXX
- Alfalah Stable Return Fund Plan XXXI
- Alfalah Stock Fund - II
- Alfalah Strategic Allocation Capital Preservation Plan II
- Alfalah Strategic Allocation Fund Plan - I
- Allied Finergy Fund
- EFU Hemayah Pension Fund
- Faysal Halal Amdani Fund
- Faysal Halal Amdani Fund II
- Faysal Halal Amdani Fund III
- Faysal Islamic Asset Allocation Fund
- Faysal Islamic Asset Allocation Fund II
- Faysal Islamic Asset Allocation Fund III (Faysal Shariah Flex Plan I)
- Faysal Islamic Asset Allocation Fund III (Faysal Shariah Flex Plan II)
- Faysal Islamic Asset Allocation Fund III (Faysal Shariah Flex Plan III)
- Faysal Islamic Asset Allocation Fund IV (Faysal Shariah Flex Plan IV)
- Faysal Islamic Asset Allocation Fund IV (Faysal Shariah Flex Plan V)
- Faysal Islamic Asset Allocation Fund IV (Faysal Shariah Flex Plan VI)
- Faysal Islamic Cash Fund
- Faysal Islamic Dedicated Equity Fund
- Faysal Islamic Financial Growth Fund (Faysal Islamic Financial Growth Plan I)
- Faysal Islamic Financial Growth Fund (Faysal Islamic Financial Growth Plan II)
- Faysal Islamic Financial Growth Fund II
- Faysal Islamic Financial Planning Fund II (Faysal Priority Ascend Plan I)
- Faysal Islamic Financial Planning Fund II (Faysal Priority Ascend Plan II)
- Faysal Islamic Financial Planning Fund II (Faysal Priority Ascend Plan III)
- Faysal Islamic KPK Government Pension Fund
- Faysal Islamic Mustakil Munafa Fund (Faysal Islamic Mehdood Muddat Plan XXI)
- Faysal Islamic Mustakil Munafa Fund (Faysal Islamic Mehdood Muddat Plan XXIX)
- Faysal Islamic Mustakil Munafa Fund (Faysal Islamic Mehdood Muddat Plan XXV)
- Faysal Islamic Mustakil Munafa Fund (Faysal Islamic Mehdood Muddat Plan XXVI)
- Faysal Islamic Mustakil Munafa Fund (Faysal Islamic Mehdood Muddat Plan XXVII)
- Faysal Islamic Mustakil Munafa Fund (Faysal Islamic Mehdood Muddat Plan XXVIII)
- Faysal Islamic Pension Fund
- Faysal Islamic Punjab Pension Fund
- Faysal Islamic Savings Growth Fund
- Faysal Islamic Sovereign Fund (Faysal Islamic Sovereign Plan I)
- Faysal Islamic Sovereign Fund (Faysal Islamic Sovereign Plan II)
- Faysal Islamic Special Income Plan I
- Faysal Islamic Stock Fund
- Faysal Islamic Stock Fund II
- Faysal Khushal Mustaqbil Fund (Faysal Barak’ah Women Savers Plan)
- Faysal Khushal Mustaqbil Fund (Faysal Nu’umah Women Savers Plan)
- HBL Cash Fund
- HBL Energy Fund
- HBL Equity Fund
- HBL Financial Sector Income Fund Plan I
- HBL Financial Sector Income Fund Plan II
- HBL Government Securities Fund
- HBL Growth Fund-Class A
- HBL Growth Fund-Class B
- HBL Income Fund
- HBL Investment Fund-Class A
- HBL Investment Fund-Class B
- HBL Islamic Asset Allocation Fund
- HBL Islamic Equity Fund
- HBL Islamic Fixed Term Fund Plan IV
- HBL Islamic Fixed Term Fund Plan VII
- HBL Islamic Fixed Term Fund Plan VIII
- HBL Islamic Fixed Term Fund Plan X
- HBL Islamic Income Fund
- HBL Islamic Money Market Fund
- HBL Islamic Pension Fund
- HBL Islamic Punjab Pension Fund
- HBL Islamic Regualar Income Fund
- HBL Islamic Savings Plan I
- HBL Islamic Stock Fund
- HBL KPK Islamic Pension Fund
- HBL KPK Pension Fund
- HBL Mehfooz Munafa Fund Plan XI
- HBL Mehfooz Munafa Fund Plan XIV
- HBL Mehfooz Munafa Fund Plan XVIII
- HBL Mehfooz Munafa Fund Plan XXI
- HBL Money Market Fund
- HBL Multi Asset Fund
- HBL Pension Fund
- HBL Punjab Pension Fund
- HBL Regular Income Fund
- HBL Stock Fund
- HBL Total Treasury Exchange Traded Fund
- KSE Meezan Index Fund
- Meezan Asset Allocation Fund
- Meezan Balanced Fund
- Meezan Capital Protected Fund III (Meezan Capital Secure Plan I)
- Meezan Cash Fund
- Meezan Daily Income Fund (MDIP I)
- Meezan Daily Income Fund (Meezan Mahana Munafa Plan)
- Meezan Daily Income Fund (Meezan Munafa Plan I)
- Meezan Daily Income Fund (Meezan Sehl Account Plan) (MSHP)
- Meezan Daily Income Fund (Meezan Super Saver Plan) (MSSP)
- Meezan Dedicated Equity Fund
- Meezan Dynamic Asset Allocation Fund (Meezan Dividend Yield Plan)
- Meezan Energy Fund
- Meezan Financial Planning Fund of Funds (Aggressive)
- Meezan Financial Planning Fund of Funds (Conservative)
- Meezan Financial Planning Fund of Funds (MAAP I)
- Meezan Financial Planning Fund of Funds (Moderate)
- Meezan Financial Planning Fund of Funds (Very Conservative Allocation Plan)
- Meezan GOKP Pension Fund
- Meezan Gold Fund
- Meezan Government Securities Fund Plan I
- Meezan Islamic Asaan Cash Fund
- Meezan Islamic Fund
- Meezan Islamic Government of Balochistan Pension Fund
- Meezan Islamic Government of Punjab Pension Fund
- Meezan Islamic Income Fund
- Meezan Paidaar Munafa Plan 34
- Meezan Paidaar Munafa Plan 39
- Meezan Paidaar Munafa Plan 43
- Meezan Paidaar Munafa Plan 45
- Meezan Paidaar Munafa Plan 47
- Meezan Paidaar Munafa Plan 48
- Meezan Paidaar Munafa Plan 49
- Meezan Paidaar Munafa Plan 50
- Meezan Paidaar Munafa Plan 51
- Meezan Paidaar Munafa Plan 52
- Meezan Pakistan ETF
- Meezan Rozana Amdani Fund
- Meezan Sovereign Fund
- Meezan Strategic Allocation Fund (MSAP I)
- Meezan Strategic Allocation Fund (MSAP II)
- Meezan Strategic Allocation Fund (MSAP III)
- Meezan Strategic Allocation Fund (MSAP IV)
- Meezan Strategic Allocation Fund (MSAP V)
- Meezan Tahaffuz Pension Fund
- UBL Asset Allocation Fund
- UBL Cash Fund
- UBL Financial Sector Fund
- UBL Fixed Return Plan II (AB)
- UBL Fixed Return Plan II (M)
- UBL Fixed Return Plan III (Y)
- UBL Fixed Return Plan III (Z)
- UBL Government Securities Fund
- UBL Growth & Income Fund
- UBL Income Opportunity Fund
- UBL Liquidity Fund
- UBL Liquidity Plus Fund
- UBL Money Market Fund
- UBL Pakistan Enterprise Exchange Traded Fund
- UBL Punjab Pension Fund
- UBL Retirement Saving Fund
- UBL Special Savings Plan X
- UBL Stock Advantage Fund
- UBL Voluntary Pension Fund KPK
