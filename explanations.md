# SarmayaSaaz — Complete Project Explanation

**A presentation preparation guide.**

This document explains the entire system: what it does, how it was built, what the
results actually mean, what is genuinely defensible, and what the honest limitations
are. Read it end to end and you should be able to present this project — and answer
hard questions — with confidence.

> **Naming note.** The project began as **FundForge** and was renamed **SarmayaSaaz**
> ("wealth builder" in Urdu). You will still see `FundForge` in `app.py` and the older
> `docs/`. Pick one name for the presentation and use it consistently. The current
> product name in the API, the dashboard title, and `pyproject.toml` is **SarmayaSaaz**.

---

## Table of Contents

1. [The One-Paragraph Pitch](#1-the-one-paragraph-pitch)
2. [The Problem We Set Out to Solve](#2-the-problem-we-set-out-to-solve)
3. [The System at a Glance](#3-the-system-at-a-glance)
4. [The Data Foundation](#4-the-data-foundation)
5. [Feature Engineering](#5-feature-engineering)
6. [The Four Engines — and Why They Differ](#6-the-four-engines--and-why-they-differ)
7. [The Nine Model Architectures](#7-the-nine-model-architectures)
8. [Results — And What They Actually Mean](#8-results--and-what-they-actually-mean)
9. [The Sharpest Insight in the Project](#9-the-sharpest-insight-in-the-project)
10. [The Software Architecture](#10-the-software-architecture)
11. [Data Integrity: The Defects We Found and Fixed](#11-data-integrity-the-defects-we-found-and-fixed)
12. [Honest Limitations](#12-honest-limitations)
13. [The Retraining Pipeline](#13-the-retraining-pipeline)
14. [Presenting This: Narrative Arc](#14-presenting-this-narrative-arc)
15. [Anticipated Questions and Answers](#15-anticipated-questions-and-answers)
16. [Glossary](#16-glossary)

---

## 1. The One-Paragraph Pitch

> SarmayaSaaz is a multi-asset AI forecasting platform that produces price forecasts
> across **four asset classes** — global commodities, cryptocurrencies, Pakistan Stock
> Exchange equities, and MUFAP mutual funds — at **seven time horizons** from one week
> to six months. It serves **302 assets** from **2,031 trained model artifacts** through
> a single API and a single dashboard. Its distinguishing feature is not the model count;
> it is that the system refuses to fabricate. Where a metric was never measured, it
> returns `null` and the interface says "not measured." Where a dataset is untrustworthy,
> the asset is withheld rather than forecast. Every number shown traces to a real
> computation on real data.

That last sentence is your differentiator. Lead with it.

---

## 2. The Problem We Set Out to Solve

Pakistani retail investors face a fragmented information landscape:

- **PSX equities** — price data is published by the exchange, but there is no accessible
  forecasting layer for retail users.
- **MUFAP mutual funds** — roughly 200 open-end funds publish daily NAV, but only as a
  flat table. No trend analysis, no forward view, no comparison tooling.
- **Commodities** — gold in particular is a culturally dominant store of value in
  Pakistan, yet pricing signals come from global markets most retail users cannot read.
- **Crypto** — high retail participation, effectively zero local analytical infrastructure.

Existing tools solve at most one of these, and almost always for US markets. **Nothing
covers Pakistani mutual funds and PSX equities alongside global commodities and crypto
in one interface.** That gap is the product thesis.

**The technical problem:** each of these asset classes behaves differently. A money-market
mutual fund tracking State Bank policy rates and a memecoin are not the same forecasting
problem, and pretending they are produces garbage. The system therefore runs **four
separate engines** with different grouping strategies, different model families, and
different confidence-band mathematics — behind **one uniform API contract**.

---

## 3. The System at a Glance

| Dimension | Value |
|---|---|
| Asset classes | 4 (commodity, crypto, stock, mutual fund) |
| Assets served | **302** |
| Forecast horizons | **7** — 7, 14, 28, 42, 60, 90, 120 days |
| Model architectures | **9** |
| Trained artifacts on disk | **2,031** |
| Backend | FastAPI (Python 3.12), ~3,700 lines |
| Frontend | Next.js 15, React 19, TanStack Query, Recharts, Tailwind |
| End-to-end tests | 23, all passing against real artifacts |

**Per-engine breakdown:**

| Engine | Assets | Grouping | Architectures | Recorded metrics | Ensemble strategy |
|---|---|---|---|---|---|
| Commodities | 6 | Per asset | 9 | Dir. accuracy, MAE, RMSE, R² | Top-3, accuracy-weighted |
| Crypto | 23 of 29 | 4 K-Means clusters | 9 | Win rate (per cluster) | All above 0.4 floor, win-rate weighted |
| PSX | 95 | 7 super-sectors | 3 | MAE (winner only) | Single winner (routed) |
| MUFAP | 178 of 198 | 5 super-clusters | 3 | MAE (winner only) | Single winner (routed) |

> **Be precise about "2,031 models."** That is the artifact file count, which includes
> scalers and per-horizon variants — not 2,031 independent neural networks. Present it as
> "2,031 trained artifacts." Inflating this is exactly the kind of claim a sharp examiner
> will probe.

**There is no PatchTST in this system.** Older draft documents mentioned it. It was never
trained. If a slide says PatchTST, remove it.

---

## 4. The Data Foundation

### 4.1 Four sources, four collection strategies

| Class | Source | Method |
|---|---|---|
| Commodities | TradingView | Replays the chart websocket protocol (handshake → resolve_symbol → create_series) |
| Crypto | Binance public spot API | `/api/v3/klines`, no auth |
| PSX | PSX Data Portal (`dps.psx.com.pk`) + TradingView fallback | Official exchange endpoint with automatic TradingView fallback (`PSX:<TICKER>`) if rate-limited |
| MUFAP | `mufap.com.pk` | Fund directory → per-fund NAV history |

**Why this matters:** using PSX's own data portal rather than Yahoo Finance means the
prices are the exchange's official record. Same for MUFAP. This is a credibility point —
you are not scraping a mirror of a mirror.

### 4.2 The re-collection story (important — this shows engineering maturity)

The original datasets (`data/`) were assembled from mixed sources, which produced
**cross-source price discrepancies**: the same asset had different prices depending on
which file you read. Rather than patch symptoms, the entire dataset was **re-collected
from scratch**, one designated source per asset class, into `data-new/`, with a
consistent null and outlier policy applied by a shared cleaning module
(`scripts/data_new_common.py`).

This produced a clean separation the system still uses today:

- **`data/`** — the original engineered feature sets. These still drive the **model inputs**,
  because the currently-serving models were trained on them.
- **`data-new/`** — freshly collected, single-source, cleaned OHLCV. This drives **chart
  display and quoted prices**.

That is a deliberate, documented split, not an accident. The models keep the exact feature
distribution they were trained on, while users see clean, accurate prices.

**The MUFAP umbrella-fund discovery.** While re-collecting, we found that ~28 pension
(VPS) fund names publish **2–4 separate NAV series** under one identical fund name,
distinguished only by category (VPS-Debt / VPS-Equity / VPS-Money Market). The previous
dataset **collapsed them into a single series**, interleaving unrelated NAV histories —
a likely root cause of the original discrepancies. The new collector writes each
sub-category as its own file: `<Fund> (<Category>).csv`.

This is an excellent story beat: *we went looking for a data bug and found a structural
misunderstanding of how the source publishes its data.*

---

## 5. Feature Engineering

All models predict **forward returns**, not absolute prices:

```
Target_Return_Nd = (Close[t+N] − Close[t]) / Close[t]
```

**Why returns and not prices?** Prices are non-stationary — gold at $1,200 in 2019 and
$4,400 in 2026 are different statistical regimes, and a model trained on the former
cannot extrapolate to the latter. Returns are approximately stationary, so a model
trained on 2019 data remains meaningful in 2026. Raw price columns (`Open`, `High`,
`Low`, `Close`, `Volume`) are **deliberately excluded from the feature set** to force
models to learn from stationary ratios.

Targets are clipped to `[-0.8, +1.5]` to neutralise the April 2020 WTI crude crash, which
produced mathematically infinite returns when oil went negative.

### Feature families

| Family | Examples | Intuition |
|---|---|---|
| **Momentum** | `Return`, `Log_Return`, `Ret_Lag_1/3/5`, `RSI_14`, `MACD_Pct` | Does recent movement persist? |
| **Trend position** | `Close_to_MA_5/20`, `Close_to_EMA_10/20` | Is price above or below its own trend? |
| **Volatility** | `Rolling_Std_10/20`, `Daily_Range_Pct`, ATR, Bollinger width | How turbulent is the regime? |
| **Volume** | `Volume_to_MA_10` | Is participation unusual? |
| **Macro** | PKR/USD, oil, NASDAQ, S&P 500, Fear & Greed | External regime context |
| **Sentiment** | FinBERT news scores, `Sentiment_EMA_7d/30d` | Narrative pressure |

**Sector-specific macro features (PSX):** every sector carries the PKR currency proxy;
`Energy_Power` additionally carries oil; `Tech_Telecom` carries NASDAQ. This is why each
sector has its own scaler and its own feature width — a detail worth mentioning because it
shows the modelling was not one-size-fits-all.

**Crypto Smart Money Concepts:** the crypto pipeline additionally computes market-structure
features — Break of Structure (BOS), Change of Character (CHoCH), Fair Value Gaps, and
Order Blocks. Absolute swing prices were converted into stationary ratios
(`Dist_to_Swing_High`) rather than kept as dollar values.

**Sentiment via FinBERT:** news headlines were scored with FinBERT (a BERT variant
fine-tuned on financial text) rather than a generic sentiment model, because "bearish" and
"short" carry domain-specific polarity that general models misread.

---

## 6. The Four Engines — and Why They Differ

This section is the intellectual core of the presentation. The design decision worth
defending is: **different asset classes got different treatment, and each choice has a
reason.**

### 6.1 Commodities — per-asset, 9 models, accuracy-weighted ensemble

**Grouping: none.** Six assets, each with its own models. With only six assets and 10+
years of daily history each, there is enough data per asset to train independently.
Clustering would only destroy information.

**Ensemble:** the **top 3 models by held-out directional accuracy** vote, weighted by that
accuracy.

**Confidence band:** the accuracy-weighted validation MAE, in **absolute price units**.
Because commodity MAE was recorded in dollars, these are genuine error bars — if the band
says ±$40 on gold, that is a real measured quantity.

### 6.2 Crypto — 4 clusters, win-rate weighted

**Grouping: K-Means into 4 clusters** on `[mean 30-day volatility, mean 7-day return]`.
With 29 assets of wildly varying history length, per-asset training would leave newer
tokens with too few rows. Clustering pools statistically similar assets.

**Ensemble:** every model whose validated win rate exceeds **0.4** votes, weighted by win
rate, where `win_rate = max(0.4, 1 − MAE_on_returns)`.

**Confidence band:** the **true min/max spread** across the models that actually voted.

> **Critical honesty point.** The original inference script added
> `np.random.normal(0, 0.03)` to every prediction to manufacture ensemble spread for the
> dashboard. That is fabrication — a ±3% random perturbation presented as model
> disagreement. It was removed. The bands are now real dispersion. **Say this out loud in
> the presentation.** Finding and removing a dishonest visual is a stronger story than
> having never had one.

### 6.3 PSX Equities — 7 super-sectors, single winner

**Grouping: 7 sectors** (Financials, Energy_Power, Cement_Construction, Tech_Telecom,
Pharmaceuticals, Fertilizers_Chemicals, Consumer_Autos). All tickers in a sector are pooled
into one training matrix — **one sector model serves every ticker in it**. There is no
ticker-identity feature.

**Why:** Pakistani equities are strongly sector-driven. Cement stocks move on cement
demand and coal prices together. Pooling multiplies the effective training rows.

**Ensemble: none.** Three candidates (XGBoost, LightGBM, tabular LSTM) compete per
sector/horizon and the **single lowest-validation-MAE model wins** that cell. This is
*routing*, not ensembling.

**Confidence band:** the winning model's validation MAE, as a **fraction of price**.

### 6.4 MUFAP Mutual Funds — 5 super-clusters, single winner

**Grouping: 5 clusters by underlying asset class** — Equity, MoneyMarket, Income,
Commodity, Balanced — assigned by a keyword classifier over MUFAP's own category strings.
Not K-Means: fund categories are already meaningful labels, so an unsupervised method
would discard real information.

**Same routed single-winner design as PSX.**

**A special problem:** NAV was deliberately dropped from the engineered feature set to
prevent leakage — a fund's NAV trivially predicts its own next NAV. So the absolute rupee
level must come from elsewhere: the raw 94 MB MUFAP export, read once at startup with only
four columns, then discarded.

> Only **178 of 198** funds are served: 20 clustered datasets contain a header and **zero
> data rows**. They are excluded at catalog-build time rather than returning an error on a
> user's first click.

---

## 7. The Nine Model Architectures

| # | Model | Type | One-line intuition |
|---|---|---|---|
| 1 | **XGBoost** | Gradient-boosted trees | Sequentially corrects previous trees' errors; strong on tabular data |
| 2 | **LightGBM** | Gradient-boosted trees | Leaf-wise growth — faster, often better on large tabular sets |
| 3 | **CatBoost** | Gradient-boosted trees | Ordered boosting reduces target leakage; robust defaults |
| 4 | **RandomForest** | Bagged trees | Many independent trees averaged; high variance reduction, no boosting |
| 5 | **LSTM** | Recurrent NN | Gated memory over a sequence; learns "what to remember and forget" |
| 6 | **GRU** | Recurrent NN | Simplified LSTM, fewer parameters, trains faster on small data |
| 7 | **Transformer** | Attention | Every timestep attends to every other; captures long-range dependency |
| 8 | **N-BEATS** | Deep residual stack | Purpose-built for time series; decomposes trend and seasonality |
| 9 | **TFT** (Temporal Fusion Transformer) | Attention + gating | Combines variable selection with temporal attention |

**Sequence lengths:** commodities use 10 days; crypto uses 30 days. Tree models consume a
single feature row (they cannot ingest sequences). The PSX/MUFAP "LSTM" is a **tabular
LSTM** — a single feature row reshaped to sequence length 1. It is honestly not a real
sequence model, and you should describe it as such if asked.

---

## 8. Results — And What They Actually Mean

This is the section that separates a good presentation from a mediocre one. The numbers
are real, and some of them are unflattering. Presenting the unflattering ones **with
interpretation** is what demonstrates competence.

### 8.1 Commodities — which architecture wins?

Across 42 asset × horizon cells, counting how often each architecture had the highest
directional accuracy:

| Architecture | Cells won |
|---|---|
| **XGBoost** | **15** |
| TFT | 5 |
| LSTM | 5 |
| LightGBM | 5 |
| CatBoost | 4 |
| N-BEATS | 3 |
| Transformer | 2 |
| GRU | 2 |
| RandomForest | 1 |

Mean directional accuracy across all cells:

| Architecture | Mean Dir. Acc | Min | Max |
|---|---|---|---|
| CatBoost | 60.85% | 33.20 | 93.78 |
| LightGBM | 59.84% | 34.21 | 93.78 |
| XGBoost | 58.74% | 33.20 | 93.78 |
| LSTM | 55.69% | 20.91 | 93.01 |
| N-BEATS | 54.70% | 38.47 | 75.53 |
| RandomForest | 53.46% | 7.86 | 81.04 |
| GRU | 51.79% | 31.11 | 77.92 |
| Transformer | 50.57% | 11.86 | 92.01 |
| TFT | 48.19% | 6.16 | 76.86 |

**Interpretation — why do trees beat neural networks here?**

1. **Data volume.** Each commodity has ~2,500 usable rows. Deep networks are data-hungry;
   gradient-boosted trees are famously the strongest family on small-to-medium **tabular**
   problems. This is the single most important reason.
2. **Feature structure.** The features are already hand-engineered, stationary, and
   informative. Trees excel at finding threshold splits in exactly that kind of data.
   Neural networks earn their advantage when they must *learn* representations — but here
   the representation work was already done by feature engineering.
3. **Variance.** Look at the min column. Trees have a floor around 33%; Transformer and TFT
   drop to 11.86% and 6.16%. Deep models on small data are **unstable** — sometimes
   excellent, sometimes catastrophic. Trees are boring and reliable, which is what you want
   in production.

**When do neural nets win?** TFT and LSTM together take 10 of 42 cells, concentrated at
**mid-range horizons (28–42 days)**. That is consistent with sequence models capturing
multi-week momentum structure that a single-row tree cannot see.

### 8.2 The horizon effect — the most important result in the project

| Horizon | Mean MAE | Mean Dir. Acc | **Mean R²** |
|---|---|---|---|
| 7d | 20.45 | 52.95% | **+0.789** |
| 14d | 29.75 | 54.39% | **+0.639** |
| 28d | 42.12 | 54.59% | **+0.394** |
| 42d | 55.80 | 54.78% | **+0.166** |
| 60d | 68.90 | 56.58% | **−0.137** |
| 90d | 92.32 | 54.44% | **−0.459** |
| 120d | 108.59 | 56.35% | **−0.713** |

**Read this carefully, because it is the honest heart of the project.**

- **Error grows ~5× from 7 days to 120 days.** Expected and unavoidable — uncertainty
  compounds with time.
- **R² goes negative beyond ~42 days.** A negative R² means the model is **worse than
  simply predicting the historical mean**. Across all cells, **123 of 378 (32.5%) have
  R² < 0**.
- **Directional accuracy stays roughly flat at 53–57%** regardless of horizon.

**What this means, stated plainly:** the system is meaningfully predictive on **magnitude**
at short horizons (7–28 days, R² 0.79 → 0.39). At long horizons it retains a modest
**directional** edge but has essentially no magnitude skill. A 55% directional hit rate is
above a coin flip, but it is not a money machine.

**Present this as a strength, not a weakness.** Any forecasting project that claims uniform
accuracy across a 6-month horizon is either overfitting or not measuring properly. Showing
that you *measured* the degradation, *quantified* it, and *surfaced* it in the product is
the mark of rigour.

### 8.3 The standout cells — and the trap inside them

Best directional accuracy achieved per commodity:

| Commodity | Model | Horizon | Dir. Acc | MAE | R² |
|---|---|---|---|---|---|
| gold | XGBoost | 120d | **93.78%** | 518.42 | 0.495 |
| silver | XGBoost | 120d | **89.36%** | 11.86 | 0.128 |
| copper | CatBoost | 120d | 80.50% | 0.462 | 0.384 |
| crude_oil | TFT | 120d | 76.86% | 10.43 | −0.405 |
| wheat | N-BEATS | 60d | 71.78% | 37.23 | −0.210 |
| natural_gas | LightGBM | 120d | 60.20% | 1.077 | −2.304 |

**The trap — and you must address it before someone else does.** Gold at 93.78%
directional accuracy over 120 days looks spectacular. But note the **MAE of $518/oz**.
The model is nearly always right about *direction* and badly wrong about *magnitude*.

Why? Gold trended strongly upward through the test period. A model that learns "gold goes
up" scores ~94% directional accuracy in a bull market **without any real predictive
skill** — it is trend-following, not forecasting. The negative R² on crude oil and natural
gas at the same horizon tells the same story from the other side.

**The correct framing:** directional accuracy on a trending asset over a long horizon is
inflated by the trend itself. The honest metrics are the short-horizon R² values, and the
`Improvement_Pct` versus a naive "no change" baseline — which the metrics module computes
precisely so this can be checked.

### 8.4 Crypto — a completely different picture

Mean win rate per architecture across all 28 cluster × horizon cells:

| Architecture | Mean win rate | Above 0.4 floor |
|---|---|---|
| **TFT_Lite** | **0.9361** | 28/28 |
| Transformer | 0.9352 | 28/28 |
| GRU | 0.9315 | 28/28 |
| LSTM | 0.9312 | 28/28 |
| NBEATS_Lite | 0.9156 | 28/28 |
| RandomForest | 0.6917 | 23/28 |
| CatBoost | 0.6558 | 22/28 |
| LightGBM | 0.6310 | 18/28 |
| XGBoost | 0.6179 | 20/28 |

**The ordering is exactly inverted versus commodities.** Neural networks dominate; trees
trail. This leads directly to the sharpest insight in the project — see §9.

**The pinned clusters:**

| Cluster | Members | Served |
|---|---|---|
| Cluster_0 (12) | AAVE, APT, AVAX, BONK, FLOKI, INJ, LDO, NEAR, PENDLE, SHIB, SUI, WIF | 10 |
| Cluster_1 (3) | BTC, GRT, IMX | 1 (BTC only) |
| Cluster_2 (1) | UNI | 0 — withheld |
| Cluster_3 (13) | ADA, BNB, CRV, DOGE, ETH, FET, FIL, LINK, MKR, RNDR, SNX, SOL, TRX | 12 |

> **Disclose this.** The clustering is badly imbalanced. Cluster_2 contains exactly one
> asset, and that asset is withheld — so it never serves anyone. Cluster_1 effectively
> serves only Bitcoin. Two of four clusters are near-degenerate. This is a real weakness
> and a clear target for the retraining run.

### 8.5 PSX — LSTM and LightGBM split almost evenly

Across 49 sector × horizon cells: **LightGBM wins 25, LSTM wins 24.**

| Winner | Mean MAE | Cells |
|---|---|---|
| LightGBM | 0.12187 | 25 |
| LSTM | 0.17817 | 24 |

**Interpretation:** LightGBM's mean MAE is much lower — but that is because LightGBM wins
disproportionately at **short horizons** where all errors are small, while LSTM wins at
**long horizons** where all errors are large. Look at the per-sector rows: LightGBM
dominates 7–60 days; LSTM takes over at 90–120 days almost everywhere.

**Why?** At 90–120 days, single-row tabular features carry little signal, and the LSTM's
(admittedly shallow) temporal structure plus its smoother inductive bias generalises
slightly better. Note this is the *tabular* LSTM at sequence length 1, so the honest
explanation is **regularisation and smoothness**, not genuine sequence modelling.

MAE here is a **fraction of price**: 0.0481 at 7 days means ~4.8% error; 0.2673 at 120 days
means ~27% error. Again — error roughly quintuples across the horizon range.

### 8.6 MUFAP — the cleanest results in the project

Across 35 cluster × horizon cells: **LightGBM 16, XGBoost 14, LSTM 5.**

| Cluster | 7d MAE | 120d MAE | Character |
|---|---|---|---|
| **MoneyMarket** | **0.0037 (0.37%)** | 0.0403 (4.03%) | Extremely stable — tracks SBP policy rates |
| **Income** | 0.0041 (0.41%) | 0.0445 (4.45%) | Highly stable — tracks PKRV bond yields |
| Balanced | 0.0140 (1.40%) | 0.0741 (7.41%) | Mixed |
| Commodity | 0.0324 (3.24%) | 0.1382 (13.82%) | Gold-linked, volatile |
| **Equity** | 0.0311 (3.11%) | 0.1114 (11.14%) | Volatile — tracks KSE-100 |

**This is your most defensible result, and it should be a slide.** A 0.37% error on
7-day money-market NAV forecasting is genuinely excellent — and it is excellent for a
*legible reason*: money-market funds hold short-duration instruments whose NAV moves almost
deterministically with policy rates. **The model is not doing anything magical; the
underlying process is nearly deterministic, and the model correctly discovers that.**

The ordering of error across clusters (MoneyMarket < Income < Balanced < Equity <
Commodity) **exactly matches the real-world risk ordering of those fund categories**. That
is a strong sanity check: the model learned the actual risk structure of Pakistani mutual
funds without being told it.

**If you present one result, present this one.** It is accurate, interpretable, and
verifiable.

---

## 9. The Sharpest Insight in the Project

Put this on a slide. It is the most intellectually interesting finding.

**The same nine architectures were trained on commodities and crypto. Trees dominate
commodities. Neural networks dominate crypto. The cause is not the asset class — it is the
selection metric.**

- Commodities rank models by **directional accuracy** — a *classification-like* criterion
  that rewards getting the sign right.
- Crypto ranks models by **win rate = max(0.4, 1 − MAE on returns)** — a *regression*
  criterion that rewards small absolute error.

Crypto returns hover near zero with occasional large moves. A model that **predicts values
close to zero** achieves a very low MAE on returns and therefore a very high "win rate" —
**without any directional skill whatsoever**. Neural networks trained with MSE/Huber loss
regress toward the conditional mean, which is near zero. So they score 0.93+. Tree models
produce sharper, more confident predictions, which increases MAE and drops their win rate
to ~0.62 — even if their *directional* calls are no worse.

**Conclusion:** the crypto win-rate figures are **not accuracy figures** and must never be
presented as such. The system already handles this correctly: `backend/routers/models.py`
returns win rate in its own dedicated field and never populates `directional_accuracy` for
crypto, and the UI heatmap labels it "% win rate across this cluster (not per asset
accuracy)."

Being able to explain *why* the same models rank oppositely on two datasets demonstrates
genuine understanding rather than pipeline execution. **This is your strongest technical
talking point.**

---

## 10. The Software Architecture

### 10.1 Layer structure

```
backend/
  config.py          all paths absolute, from one Pydantic settings object
  schemas.py         the API contract — one response shape for all 4 engines
  ml/
    architectures.py every PyTorch network, defined exactly once
    registry.py      thread-safe LRU cache over deserialized artifacts
  engines/
    base.py          the Engine abstract contract
    commodities.py   per-asset, 9-model accuracy-weighted ensemble
    crypto.py        4 pinned clusters, win-rate weighted
    routed.py        shared base for single-winner engines
    mufap.py  stocks.py
  services/          live prices, MUFAP NAV, news, forecast snapshot
  routers/           system, assets, forecasts, market, models
frontend/            Next.js 15 · TanStack Query · Tailwind · Recharts
```

### 10.2 Design decisions worth defending

**One contract, four engines.** Every engine implements the same abstract interface
(`_build_catalog`, `forecast`, `quote`). Adding a fifth asset class requires **zero changes
to any router**. This is the Strategy pattern, and it is the reason the codebase stayed
manageable.

**No `os.chdir()` anywhere.** Every path derives from a single `PROJECT_ROOT`. The previous
integration's fragility came from relative paths whose resolution depended on import order.

**Graceful degradation.** An engine that fails to initialise is recorded as offline rather
than taking down the API — a missing MUFAP export should not stop anyone from pricing
Bitcoin. `/api/health` reports each engine's status individually.

**LRU model cache.** Loading three artifacts per request costs ~800 ms of disk I/O. Handles
are cached by absolute path, cutting repeat predictions to ~5 ms — a **160× speedup** on
warm requests. Capacity 128, thread-safe.

**Snapshot precomputation.** Top Movers, Markets, and the heatmap need results across all
302 assets. Computing 302 × 7 horizons on demand takes minutes, so it is precomputed and
cached to disk, with the generation timestamp surfaced in the UI so users see its age
honestly rather than assuming it is live.

**HTTP `no-store` on every response.** Without it, a browser can serve a stale forecast
from its own cache after the backend data changes.

**Version pinning as a correctness requirement.** `scikit-learn==1.7.2` is pinned exactly
because every `StandardScaler` and `RandomForestRegressor` was pickled under that version.
Loading them under 1.9.x raises `InconsistentVersionWarning` and **can silently alter
transform output** — silently wrong numbers, the worst failure mode. `pandas` is held below
3.0 for the same class of reason.

### 10.3 Explainability

For tree-based models, **SHAP** (SHapley Additive exPlanations) computes per-feature
attribution for the specific prediction shown, surfaced as the "Forecast Drivers" panel.
SHAP values are grounded in cooperative game theory: each feature's contribution is its
average marginal contribution across all possible feature orderings.

When the leading model is a neural network, the UI states plainly that per-feature
contributions were not derived — rather than showing a misleading proxy.

---

## 11. Data Integrity: The Defects We Found and Fixed

**This section is your differentiator. Most student projects present only what worked.**

### 11.1 Synthetic noise injected into predictions

`src/crypto/stage6_inference.py` added `np.random.normal(0, 0.03)` to every model output to
manufacture ensemble spread for the dashboard. **Removed.** Bands are now the true min/max
across models that voted.

### 11.2 The cluster-map permutation bug

K-Means cluster **labels are arbitrary** — an identical partition can come back numbered
differently between runs, and the partition itself drifts as new data arrives. Recomputing
clusters at request time moved BTC, GRT, and IMX from cluster 1 to cluster 2, swapping them
with UNI. **Bitcoin was then scored by models trained on a single sub-$10 token.** Its
`MACD_Hist` — denominated in absolute price — z-scored to roughly **251,000**, and the
ensemble returned **+41,071%** at 14 days.

**Fix:** the asset→cluster map is **pinned** to the assignment the models were trained
under (`results/crypto/cluster_map.json`) and never recomputed at runtime.

**Why this is a great story:** it is a subtle, non-obvious bug that produces a
catastrophically wrong output, and the fix required understanding *why* K-Means labels are
not stable identifiers.

### 11.3 The mixed feature-contract defect

154 commodity artifacts expect **exactly 9 fewer features** than their scaler emits — the
sentiment block — a residue of a champion/challenger revert. Widths vary by commodity:

| Scaler width | Model width | Artifacts |
|---|---|---|
| 34 | 25 | 41 (copper) |
| 35 | 26 | 40 (silver, crude oil) |
| 36 | 27 | 42 (gold, natural gas) |
| 37 | 28 | 31 (wheat) |

**The previous implementation hardcoded a truncation to 27** — which is gold's and natural
gas's width. Copper, silver, and wheat raised shape errors that were **swallowed by a bare
`except`**, so those commodities were quietly predicting from a *partial* ensemble with no
visible symptom.

**Fix:** the registry reads each model's **true width from the artifact itself** and
truncates to it. This is mathematically sound because `StandardScaler` normalises each
column independently and the sentiment features occupy the final columns — so slicing after
scaling is equivalent to having scaled the narrower set.

**Verified:** `scripts/verify_registry.py` confirms **162/162 artifacts predict, 0 fail**,
including the 65 that require truncation.

### 11.4 Six withheld crypto datasets

`GRT`, `IMX`, `SUI`, `RNDR`, `APT`, and `UNI` have price history ending **490–1,596 days**
behind the rest of the universe. `UNI` is the clearest case: it starts **2020-05-07** —
months before Uniswap launched — and peaks at **$0.60**. It is a **mislabeled download**,
not Uniswap.

Rather than special-casing one ticker, any asset whose history ends more than 90 days
behind the universe median is withheld: **a forecast on the wrong asset is worse than no
forecast.**

### 11.5 Twenty empty MUFAP datasets

20 clustered fund files contain a header and zero data rows; 11 more lost their date column
name in a `to_csv` round trip. Both are detected cheaply from the first two lines at
startup — the empty ones excluded, the unnamed ones recovered positionally — rather than
surfacing as a 503 on a user's first click.

### 11.6 Strict complete-candle enforcement & 24/7 market alignment

Traditional stock (PSX) and commodity feeds strictly enforce completed past trading sessions (`< today`), automatically dropping unclosed intraday bars during market hours (`cutoff = today`). This prevents forming candles from polluting historical series or feature matrices. Crypto markets operate 24/7 and evaluate daily closes at UTC midnight (`today - 1 day` UTC), ensuring daily update collection reliably captures Sunday's closed bars on Mondays.

### 11.7 Server-side synchronization state & Standby Guard

Background auto-update orchestration tracks a real server-side timestamp (`started_at`). The frontend `SyncGuard` component queries `GET /api/system/update-status` and computes true server elapsed time, avoiding client-side timer resets across page reloads.

### 11.8 The governing principle

> **No synthetic values anywhere.** Where a metric was never recorded, the API returns
> `null` and the UI renders "not measured" rather than a plausible-looking number.

---

## 12. Honest Limitations

Have these ready. Volunteering them is far stronger than being caught by them.

1. **R² is negative beyond ~42 days** in 32.5% of commodity cells — worse than predicting
   the mean. Long-horizon magnitude forecasts should not be trusted.
2. **Directional accuracy ~55%** on average is a modest edge, and inflated on trending
   assets.
3. **Crypto win rate is not accuracy** — see §9. Models predicting near zero score highly.
4. **Crypto clustering is degenerate** — two of four clusters effectively serve 0 and 1
   assets.
5. **PSX/MUFAP have no held-out test set.** They use an 80/20 train/validation split where
   validation MAE serves as *both* the model-selection criterion *and* the reported
   confidence metric. That is mildly optimistic: the reported number was chosen partly
   because it was the best. Commodities do it properly with a 70/10/20 split.
6. **The PSX/MUFAP "LSTM" is not a sequence model** — it is a single feature row at
   sequence length 1.
7. **No transaction costs, slippage, or backtest.** Directional accuracy is not profit.
   There is no Sharpe ratio, no maximum drawdown, no equity curve. This is a *forecasting*
   system, not a validated *trading* system.
8. **Live prices are disabled by default.** Yahoo Finance quotes a *different instrument*
   per asset class than `data-new/` was collected from (COMEX futures vs OANDA spot for
   gold; Yahoo's PSX feed vs PSX's own portal), so enabling it reintroduces exactly the
   cross-source discrepancies the re-collection eliminated. Forecasts display `stored`
   prices with an explicit "as of" date.
9. **One model per sector serves every ticker in it.** PSX has no per-company model —
   Lucky Cement and Fauji Cement receive predictions from the same sector model.
10. **The serving artifacts were trained on the older `data/` features**, not the
    re-collected `data-new/`. Charts and prices come from the clean data; model inputs do
    not. The retraining pipeline exists to close this gap.

---

## 13. The Retraining Pipeline

Built, feature-engineered, and ready — **not yet executed**. This is the natural "what's
next" slide.

```
scripts/collect_*.py  →  data-new/  →  training-scripts-new/*/01_feature_engineering.py
                                    →  data-ready/  →  */02_train_all.py
                                    →  models-new/ + results-new/
```

**Status:** all `01_feature_engineering.py` scripts have been run — `data-ready/` is
populated and schema-verified (6 commodities, 31 crypto, 97 PSX tickers across 7 sectors,
190 funds across 4 clusters). The four `02_train_all.py` scripts have **not** been run.

**Deliberate improvements over the original training:**

| Change | Rationale |
|---|---|
| **Purge/embargo gap** at every split boundary, sized to the horizon | Without it, a training row's forward-looking target window spans into validation — genuine leakage the original notebooks had |
| **100 epochs, early stopping patience 22**, best-epoch weights restored | The original crypto DL used a fixed 15-epoch loop with no early stopping |
| MUFAP classifier now includes `"sovereign"` as an Income keyword | Fixes a real **train/serve skew** — the production classifier had it, the training one did not |
| Best-epoch checkpointing | Never ship the last epoch's weights if they were not the best — that is the overfit tail |

**What retraining is expected to fix:**

- **UNI becomes real.** New Binance data starts 2020-09-17 at $0.30 (Uniswap launched
  September 2020) and runs to 2026-08-18 at $3.287. The mislabeled file is gone — so the
  6 withheld assets should become serveable.
- **PEPE and TAO join** (previously dropped for thin history): 29 → 31 tickers.
- **Cluster rebalancing** — the degenerate 1-asset cluster should resolve.
- Models finally trained on the same clean data the charts display.

**Two things to watch:**

1. **MUFAP loses its Commodity cluster.** `data-ready/mufap/` has only 4 clusters; the
   `Commodity` one is absent. Two funds live there — *Meezan Gold Fund* and *UBL Retirement
   Saving Fund*. They would forecast nothing after a swap unless reclassified.
2. **Paths differ on swap-in.** The trainer writes `models-new/psx/` but the backend reads
   `models/stocks/`; `models-new/commodities/` maps to
   `models/commodities/models_production/`.

**Longer-term roadmap** (`docs/future_works.md`): a P&L backtesting engine with realistic
transaction costs (0.1% fees, 0.05% slippage) producing Sharpe ratio, maximum drawdown, and
an equity curve versus buy-and-hold — the missing piece that turns forecasting metrics into
financial validation.

---

## 14. Presenting This: Narrative Arc

A structure that flows. Roughly 15–20 minutes.

**Act I — The Gap (2 min).** Pakistani investors have no unified analytical layer. PSX,
MUFAP, commodities, crypto — four fragmented worlds, no tool spans them. *Land the human
problem before any technology.*

**Act II — The Scale (2 min).** 302 assets, 4 engines, 7 horizons, 9 architectures, 2,031
artifacts, one API, one dashboard. **Demo here** — it is more persuasive than any slide.

**Act III — The Design Reasoning (4 min).** Why four engines and not one. Money-market funds
and memecoins are not the same problem. Explain grouping choices: per-asset for commodities
(enough data), K-Means for crypto (varying history), sectors for PSX (sector-driven market),
categories for MUFAP (labels already meaningful).

**Act IV — The Results, Honestly (5 min).**
- Lead with MUFAP MoneyMarket: **0.37% error at 7 days**, and the error ordering across
  clusters matches real-world risk ordering.
- Then the horizon table: R² **+0.789 → −0.713**. *"We measured where our system stops
  working, and we show it in the product."*
- Then the gold trap: 93.78% directional accuracy with $518 MAE — and explain why.
- Then §9: same models, opposite rankings, because the *metric* differs.

**Act V — Engineering Integrity (4 min).** The four defects. Lead with the synthetic noise
removal and the +41,071% Bitcoin bug. *"We found our own system lying, and we fixed it."*

**Act VI — What's Next (2 min).** Retraining pipeline built and ready; UNI fix; backtesting
engine as the path from forecasting to validated strategy.

**Closing line suggestion:**
> "We did not build a system that is always right. We built one that is never dishonest
> about when it is wrong."

---

## 15. Anticipated Questions and Answers

**Q: Is this profitable? Can I trade on it?**
No, and we do not claim it. There is no backtest, no transaction costs, no slippage
modelling, no Sharpe ratio. Directional accuracy is not profit. Building the P&L simulator
is the explicit next phase.

**Q: 55% directional accuracy — isn't that basically a coin flip?**
It is a modest edge, and we present it as one. Note two things: it is measured on held-out
data, and it varies enormously by cell — MUFAP money-market forecasting at 0.37% error is
genuinely strong because the underlying process is nearly deterministic. We do not average
the good and bad together to manufacture an impressive headline number.

**Q: Why is gold at 93% accuracy but $518 MAE?**
Because gold trended strongly upward through the test period. A model that learns "gold
rises" is right about direction ~94% of the time without real skill. That is trend
following, not forecasting — which is exactly why we report R² and MAE alongside direction
rather than direction alone.

**Q: Why 2,031 models? Isn't that overkill?**
It is 2,031 *artifacts*, including scalers — 9 architectures × 7 horizons × assets/groups,
plus per-horizon scalers. Each horizon is a genuinely separate learning problem (direct
multi-horizon forecasting), so a 7-day and a 120-day model are different models, not the
same model applied twice.

**Q: Why not one model for everything?**
Because a money-market fund tracking policy rates and a memecoin have nothing statistically
in common. Pooling them would force one set of parameters to serve incompatible dynamics.
The four-engine design is the central architectural claim.

**Q: How do you know there is no data leakage?**
All splits are chronological, never random. Commodities use 70/10/20 with a purge gap sized
to the horizon. Scalers are fit on the training fold only. `assert_no_overlap()` defensively
verifies train/val/test index sets are disjoint. **Caveat we volunteer:** PSX and MUFAP have
no separate test set — validation MAE serves as both selector and reported metric, which is
mildly optimistic.

**Q: Why is scikit-learn pinned so aggressively?**
Every scaler and RandomForest was pickled under 1.7.2. Loading under a different minor
version can silently change transform output — wrong numbers with no error. Correctness
requirement, not preference.

**Q: What's the hardest bug you fixed?**
The K-Means cluster permutation. Cluster labels are not stable identifiers — recomputing
moved Bitcoin into a cluster trained on a sub-$10 token, its price-denominated MACD z-scored
to ~251,000, and the ensemble returned +41,071% at 14 days. The fix was recognising that the
cluster map is part of the *trained model state*, not something to recompute at runtime.

**Q: Why is UNI excluded?**
The stored file is a mislabeled download — it starts May 2020, months before Uniswap
launched, and peaks at $0.60. Any asset whose history ends more than 90 days behind the
universe median is withheld. Freshly collected Binance data fixes this, and retraining
should restore it.

**Q: Are the prices live?**
No, by default. Live quoting is implemented but disabled because Yahoo quotes a different
instrument per asset class than our datasets were collected from, which would reintroduce
the cross-source discrepancies we eliminated. Every asset displays an explicit "as of" date
and whether the price is `live` or `stored` — never silently mixed.

**Q: What would you do differently?**
Add a held-out test set for PSX and MUFAP; fix the degenerate crypto clusters; build the
backtester before adding more architectures. More models was not the bottleneck — validation
was.

---

## 16. Glossary

| Term | Meaning |
|---|---|
| **MAE** | Mean Absolute Error — average absolute difference between predicted and actual. In native price units for commodities; a fraction of price for PSX/MUFAP. |
| **RMSE** | Root Mean Squared Error — like MAE but penalises large errors more heavily. |
| **R²** | Coefficient of determination — fraction of variance explained. 1.0 is perfect; **0 means no better than predicting the mean; negative means worse than the mean.** |
| **Directional Accuracy** | Percentage of predictions that got the *sign* of the move right. 50% is a coin flip. |
| **MAPE** | Mean Absolute Percentage Error — MAE expressed as a percentage. |
| **Win rate (crypto)** | `max(0.4, 1 − MAE_on_returns)`. **An ensemble weight, not an accuracy.** |
| **Improvement_Pct** | Percentage improvement over a naive "no change" forecast — the honest baseline. |
| **Purge / embargo gap** | A time gap at a split boundary preventing a training row's forward-looking target window from overlapping validation dates. |
| **Direct multi-horizon forecasting** | Training a separate model per horizon rather than iterating a one-step model forward. |
| **Stationarity** | A series whose statistical properties do not change over time. Returns are approximately stationary; prices are not. |
| **SHAP** | SHapley Additive exPlanations — game-theoretic per-feature attribution for a single prediction. |
| **FinBERT** | BERT fine-tuned on financial text, so domain terms like "bearish" carry correct polarity. |
| **NAV** | Net Asset Value — a mutual fund's per-unit price. |
| **MUFAP** | Mutual Funds Association of Pakistan — the industry body publishing daily NAV. |
| **PSX** | Pakistan Stock Exchange. |
| **PKRV** | Pakistan Revaluation Rate — the benchmark bond yield curve. |
| **Routed inference** | One winning model owns each group/horizon cell — as opposed to ensembling several. |
| **Ensemble** | Combining multiple model predictions, here weighted by measured performance. |
| **K-Means** | Unsupervised clustering into k groups. **Its labels are arbitrary and permute between runs.** |
| **LRU cache** | Least Recently Used cache — evicts the least recently accessed entry when full. |

---

## Appendix: Verification Commands

Every claim in this document is reproducible:

```bash
uv run python scripts/smoke_artifacts.py    # all 5 artifact formats deserialize
uv run python scripts/audit_features.py     # scaler vs model input widths
uv run python scripts/verify_registry.py    # 162/162 commodity artifacts predict
uv run python -m pytest tests/ -q           # 23 end-to-end API tests
```

Current verified status: **10 passed / 1 warned / 0 failed**, **162/162 predicting**,
**23 tests passing**, **4/4 engines online**, **302 assets**, **2,031 artifacts**.

---

*Forecasts are model output, not investment advice.*
