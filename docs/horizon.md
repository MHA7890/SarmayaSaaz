# Multi-Horizon Forecasting Plan

This document covers how the system predicts multiple time horizons (1, 7, 14, 28, 42, 60, 90, 120 days), the full implementation plan for this using the chosen approach, and how news/sentiment data will later be integrated into the same structure. Use this alongside the earlier project brief and system explanation files as AI coding context.

## Horizons in scope

The system forecasts eight horizons per commodity: **1-day, 7-day, 14-day, 28-day, 42-day, 60-day, 90-day, 120-day.**

## Chosen approach: Direct multi-horizon forecasting (Approach 1)

A separate model is trained per horizon, per model type, per commodity. The target column changes per horizon; the feature table and pipeline structure stay the same.

### Why this approach (not recursive, not multi-output)

- **Recursive forecasting** (predict day+1, feed it back in to predict day+2, repeat) was rejected — errors compound over iterations and become unreliable by the time you reach 60 days out.
- **Multi-output regression** (one model predicting all horizons at once) was considered but rejected for the initial build — harder to debug, and a noisy signal at one horizon can drag down accuracy at others since they'd share model weights.
- **Direct/separate models per horizon** was chosen because: no compounding error, each horizon can independently learn what matters at that timescale (daily technicals may dominate short horizons, macro/fundamentals may matter more at longer horizons), and it fits the existing pipeline without restructuring — just a new target column and a new training run per horizon.

### Scale of this approach

Per commodity: 8 horizons × 3 model types (XGBoost, LightGBM, LSTM) = 24 models.
Across all 6 commodities: 24 × 6 = 144 models total.

This is mechanically simple even though the count is large — every model uses the identical feature table for that commodity, just a different target column and a separate training run.

## Full implementation steps

### Step 1 — Create horizon-specific target columns

For each commodity's feature table, add one target column per horizon:

```
Target_1d  = Close price 1 trading day ahead
Target_7d  = Close price 7 trading days ahead
Target_14d = Close price 14 trading days ahead
Target_28d = Close price 28 trading days ahead
Target_42d = Close price 42 trading days ahead
Target_60d = Close price 60 trading days ahead
Target_90d = Close price 90 trading days ahead
Target_120d = Close price 120 trading days ahead
```

Note: use trading days, not calendar days, since your data only has rows for trading days. Also consider adding `Target_Return_Nd` (percentage return instead of raw price) alongside each — return targets are often more stationary and easier for models to learn, especially at longer horizons.

### Step 2 — Handle the "trailing rows" problem

The last N rows of your dataset won't have a valid target for horizon N (e.g. the last 60 rows can't have a `Target_60d` value, since there's no future data 60 days beyond your most recent row). Drop or explicitly exclude these trailing rows from training for each horizon's dataset — don't leave NaNs in the target column.

### Step 3 — Build the full model roster per commodity

For each commodity, train:

| Horizon | XGBoost | LightGBM | LSTM |
|---|---|---|---|
| 1-day | Model_XGB_1d | Model_LGB_1d | Model_LSTM_1d |
| 7-day | Model_XGB_7d | Model_LGB_7d | Model_LSTM_7d |
| 14-day | Model_XGB_14d | Model_LGB_14d | Model_LSTM_14d |
| 28-day | Model_XGB_28d | Model_LGB_28d | Model_LSTM_28d |
| 42-day | Model_XGB_42d | Model_LGB_42d | Model_LSTM_42d |
| 60-day | Model_XGB_60d | Model_LGB_60d | Model_LSTM_60d |
| 90-day | Model_XGB_90d | Model_LGB_90d | Model_LSTM_90d |
| 120-day | Model_XGB_120d | Model_LGB_120d | Model_LSTM_120d |

Build order: get the 1-day horizon fully working end-to-end first (data → features → 3 models → ensemble → output). Once that's validated, replicate the exact same process for 7-day by swapping only the target column. Repeat for the remaining horizons — by this point it is largely copy-paste with a changed target, not new design work.

### Step 4 — Evaluate each horizon independently

Each horizon-specific model is evaluated against its own naive baseline (naive forecast for horizon N = today's price, unchanged). Track MAE, RMSE, and directional accuracy per model, per horizon, per commodity — extend the existing comparison table with a `Horizon` column.

It's expected that accuracy degrades as horizon increases — this is normal and should be reflected honestly in the output, not treated as a failure to fix.

### Step 5 — Ensemble per horizon

The ensemble step (weighted combination of XGBoost/LightGBM/LSTM predictions) is run separately per horizon. The weighting for the 1-day ensemble may look completely different from the weighting for the 60-day ensemble — each is calculated from that horizon's own backtested model accuracy, not a fixed global weight reused across horizons.

### Step 6 — Uncertainty must widen with horizon

The predicted range and confidence score must be calculated per-horizon-model from that model's own backtested error distribution. A 1-day forecast should show a tighter range and higher confidence than a 60-day forecast for the same commodity — this should emerge naturally from each horizon's actual backtested error, not be manually forced, but it must be checked and should never show a suspiciously similar range width across horizons.

### Step 7 — Output structure per commodity

Instead of one single prediction, each commodity's output now contains one entry per horizon:

```
Gold:
  1d:  range [$2,410–$2,425], confidence 68%
  7d:  range [$2,395–$2,460], confidence 61%
  14d: range [$2,370–$2,490], confidence 55%
  28d: range [$2,340–$2,520], confidence 49%
  42d: range [$2,310–$2,550], confidence 46%
  60d: range [$2,300–$2,580], confidence 44%
  90d: range [$2,280–$2,600], confidence 41%
  120d: range [$2,250–$2,630], confidence 38%
```

Range width and confidence should visibly widen/decrease as horizon increases. This feeds the dashboard's per-commodity card as a horizon selector or a stacked horizon list, not a redesign of the card itself.

## Future plan: adding news/sentiment data (Stage 5)

This stays a later stage, unchanged in sequence — added only after the macro-data stage (Stage 4) is validated. Two things change once multi-horizon forecasting is in place:

### Sentiment must be horizon-aware

- A single day's news spike (e.g. a sudden supply disruption headline) is highly relevant to 1-day and 7-day forecasts but should be decayed/averaged out for 28-60 day forecasts, since old news naturally loses relevance over longer timeframes.
- Practical implementation: compute sentiment features at multiple rolling windows — e.g. `Sentiment_1d` (today's news sentiment), `Sentiment_7d_avg` (rolling 7-day average sentiment), `Sentiment_30d_avg` (rolling 30-day average sentiment) — and feed the appropriate window's sentiment feature(s) into each horizon's model. Short-horizon models get more weight on same-day/recent sentiment; long-horizon models get more weight on the smoothed/averaged sentiment.

### Complete plan for implementing Approach 1 after sentiment data is added

1. **Collect and align sentiment data first**, following the same release-date/alignment discipline used for macro data (Stage 4) — sentiment scores must be timestamped to when the news was actually published, not backfilled, to avoid look-ahead bias.
2. **Generate the rolling sentiment features** described above (1d, 7d avg, 30d avg, and optionally 60d avg) per commodity, since news relevance differs by commodity (e.g. wheat cares about weather/export-ban news, oil cares about OPEC/geopolitical news).
3. **Merge these sentiment features into each commodity's existing feature table** (the same table already holding OHLCV, technical indicators, and macro data) — sentiment becomes additional columns, not a separate pipeline.
4. **Retrain all 18 models per commodity (6 horizons × 3 model types) with sentiment features added**, keeping the prior macro-only version of each model saved separately so before/after comparison is possible.
5. **Re-run the full evaluation table** (MAE, RMSE, directional accuracy, per model, per horizon, per commodity) comparing sentiment-included models against the macro-only baseline from Stage 4. Confirm sentiment actually improves results before keeping it — per the project's core principle, no data source is kept just because it was added.
6. **Expect uneven impact across horizons** — sentiment is likely to show a bigger improvement for 1-day and 7-day models than for 42-day and 60-day models, given news relevance naturally decays over time. This unevenness is a legitimate, expected finding, not a bug.
7. **Re-run the ensemble step per horizon** with the updated (sentiment-included) models now in the roster, re-checking whether ensemble weighting shifts once sentiment-aware models are added — a model that was previously the weakest of the three for a given horizon may become the strongest if sentiment features especially benefited it.
8. **Update the output structure** to optionally surface sentiment-derived drivers in the "top drivers" section of the forecast output (e.g. "Negative sentiment spike: Ukraine export restrictions" as a driver for wheat's 7-day forecast), sourced from the SHAP importance of the sentiment features on the winning model.
9. **Re-validate the full backtesting process** (rolling-window backtest, held-out most recent 1-2 years) with sentiment included, to confirm the improvement holds across different market regimes and isn't an artifact of one favorable period.

## Key principles carried over from the rest of the project

- No look-ahead bias at any stage, horizon, or data source — always align by actual public availability date
- Every addition (new horizon, new data source) is benchmarked against the prior best result before being accepted
- Separate models per commodity remains unchanged; multi-horizon adds a second dimension (per commodity, per horizon) rather than replacing the per-commodity separation
- Uncertainty is always reported honestly — range and confidence must reflect each horizon's own backtested error, never a single fixed formula applied everywhere