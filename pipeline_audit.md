# End-to-End Pipeline Audit — Collection → Features → Training

**Question asked:** *Did we do everything correctly, from data collection through final
retraining? Are there errors that could undermine the project's credibility?*

**Scope:** `scripts/collect_*.py` → `data-new/` → `training-scripts-new/*/01_feature_engineering.py`
→ `data-ready/` → `training-scripts-new/*/02_train_all.py` → `models-new/` + `results-new/`

**Method:** source review of every stage, plus **empirical measurement** — splits were
reproduced and the actual calendar gaps and contamination rates computed from the real
data rather than inferred from the code's comments.

---

## Verdict

**The pipeline is fundamentally sound.** Three of four engines are leakage-free by
measurement, feature construction is causal throughout, and no lookahead exists in the
technical indicators.

**Two genuine defects were found.** Neither invalidates the project, both are fixable, and
one directly contradicts a claim written in the code's own docstring. You need to know
about them before you present, because a competent examiner reading the source could find
the first one.

| # | Finding | Severity | Affects credibility? |
|---|---|---|---|
| 1 | PSX purge gap measured in **rows**, not days | **High** | **Yes** — contradicts a stated "no data leakage" claim |
| 2 | Crypto retains `Close` (absolute price) as a model feature | **Medium-High** | Yes — reverses a deliberate original design decision |
| 3 | Macro features fetched live at feature-engineering time | Medium | No — reproducibility, not correctness |
| 4 | PSX/MUFAP report validation MAE as both selector and metric | Medium | Partly — pre-existing, disclose it |
| 5 | MUFAP Commodity cluster absent; crypto Cluster_3 lacks 90d/120d | Low | No — handled gracefully |

---

## Finding 1 — PSX purge gap is measured in rows, not days ⚠️ **HIGH**

### The claim in the code

`training-scripts-new/psx/02_train_all.py` docstring states:

> *"No data leakage: chronological 80/20 split with a 120-row purge gap at the boundary
> (same fixed gap the original used, sized to the longest 120d horizon)"*

### What actually happens

```python
PURGE_GAP_ROWS = 120
train_df = df.iloc[:split_idx]
val_df   = df.iloc[split_idx + PURGE_GAP_ROWS:]
```

`df` here is a **pooled sector matrix** — every ticker in the sector concatenated and
sorted by date. So each calendar date contributes **one row per ticker**. Skipping 120
*rows* does not skip 120 *days*; it skips `120 / tickers_per_day` days.

### Measured reality

| Sector | Tickers | Rows/day | Actual gap |
|---|---|---|---|
| Consumer_Autos | 33 | 27.2 | **6 days** |
| Financials | 16 | 15.6 | **9 days** |
| Energy_Power | 13 | 13.0 | **11 days** |
| Cement_Construction | 12 | 11.9 | 14 days |
| Fertilizers_Chemicals | 9 | 8.2 | 18 days |
| Pharmaceuticals | 9 | 6.9 | 20 days |
| Tech_Telecom | 5 | 4.4 | 34 days |

**The gap needs to be `h` days — up to 120. It is 6–34 days, and it does not scale with
horizon at all.** Perversely, the *larger* the sector, the *smaller* the gap.

### Severity — quantified, not assumed

Percentage of training rows whose target window reaches into the validation period:

| Sector | 7d | 14d | 28d | 42d | 60d | 90d | 120d |
|---|---|---|---|---|---|---|---|
| Consumer_Autos | 0.1% | 0.4% | 1.0% | 1.7% | 2.5% | 3.9% | **5.3%** |
| Pharmaceuticals | 0.0% | 0.0% | 0.4% | 1.1% | 1.8% | 3.3% | **4.8%** |
| Financials | 0.0% | 0.2% | 0.8% | 1.3% | 1.9% | 3.2% | 4.2% |
| **Mean** | **0.0%** | **0.1%** | **0.5%** | **1.1%** | **1.8%** | **3.1%** | **4.3%** |

By comparison, commodities measured **0.0% at every horizon**.

### Honest assessment

**This is a real leakage defect, but a modest one.** At 7d and 14d it is effectively zero.
At 120d, ~4.3% of training rows have targets overlapping validation. That is enough to
bias long-horizon PSX validation MAE **optimistically**, but not enough to make the results
meaningless.

**What it affects:**
- Reported PSX validation MAE — mildly optimistic at 60d/90d/120d
- The **−10.8% PSX improvement** claim — the comparison is old-vs-new, and *both* runs had
  this same defect, so the relative improvement stands; the absolute error is understated
- Winning-model selection per cell — marginally
- Production confidence bands, which derive from that MAE

**What it does not affect:** commodities, crypto, and MUFAP are all clean (verified below).

### The fix

Replace the row-based gap with a date-based one, matching what `common/splits.py` already
does correctly for the other engines:

```python
def split_with_purge(df, horizon_days):
    dates = pd.DatetimeIndex(df.index)
    split_date = dates[int(len(df) * 0.8)]
    train = df[dates <= split_date]
    val   = df[dates >  split_date + pd.Timedelta(days=horizon_days)]
    if len(val) < 50:
        val = df[dates > split_date]
    return train, val
```

This requires passing `h` into the function and retraining PSX (~7 minutes).

---

## Finding 2 — Crypto keeps absolute `Close` as a model feature ⚠️ **MEDIUM-HIGH**

### The original design intent

`docs/crypto_project_tracker.md`, Phase 2:

> *"Aggressively drop non-stationary raw prices (open, high, low, close) to force the
> models to learn from stationary ratios and momentum... purged all absolute dollar values
> (Close, Volume)."*

### What the new pipeline does

`training-scripts-new/crypto/01_feature_engineering.py:118`:

```python
df = df.drop(columns=["Open", "High", "Low", "Volume",
                      "Last_Swing_High_Price", "Last_Swing_Low_Price"], ...)
```

**`Close` is not in that list.** And `02_train_all.py` selects features as
`[c for c in df.columns if not c.startswith("Target_")]` — so `Close` is fed to every
crypto model as a feature.

Verified: crypto has 35 features including `Close`. Commodities, PSX, and MUFAP have
**no raw price columns** in their feature sets.

### Why this matters

Crypto models are trained **per cluster, pooled across assets**. Price ranges within a
cluster are extreme:

| Cluster | Assets | Close range | Ratio |
|---|---|---|---|
| Cluster_0 | 7 | $0.000002 – $192.20 | 82,844,828× |
| Cluster_1 | 9 | $0.069840 – $64,237.71 | 919,784× |
| Cluster_2 | 11 | $0.000004 – $1,813.70 | **408,490,991×** |
| Cluster_3 | 4 | $0.000003 – $1.34 | 522,957× |

Under a shared `StandardScaler`, the z-scores become:

```
Cluster_1:  BTC  $64,237.71  ->  z = +3.33
            ETH   $1,899.61  ->  z = -0.20
            BNB     $601.23  ->  z = -0.27
Cluster_2:  MKR   $1,813.70  ->  z = +3.06
            AAVE     $89.41  ->  z = -0.14
```

**Two problems:**

1. **It is a non-stationary feature.** BTC at $30,000 in training and $64,000 at inference
   is out-of-distribution — exactly the failure mode returns-based targets exist to avoid.
2. **It acts as an implicit asset-identity label.** In a pooled cluster model, `Close`
   alone identifies which asset a row belongs to. The model can memorise per-asset
   behaviour rather than learning transferable structure — which defeats the entire
   purpose of clustering.

### Honest assessment

**This is a methodological regression, not a catastrophe.** The z-scores (~3.3 max) are
nowhere near the old `MACD_Hist` disaster that hit ~251,000 and produced +41,071%
forecasts. The models still produce sane output — the live BTC forecast is −0.30% at 7d.

But it **reverses a documented, deliberate design decision** made in the original project
for a good reason. If someone asks *"why did you re-introduce absolute prices after
explicitly removing them?"*, there is no good answer other than "it was an oversight."

### The fix

Add `"Close"` to the drop list in `crypto/01_feature_engineering.py:118`, re-run crypto
feature engineering and training (~15 minutes total). Note the commodity trainer keeps
`Close` in the *file* but excludes it from features via `non_feature_columns()` — crypto
should do the same, since nothing downstream needs it.

---

## Finding 3 — Macro features are fetched live, not pinned 🟡 **MEDIUM**

`psx/01_feature_engineering.py` and `mufap/01_feature_engineering.py` call:

```python
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
pkr    = macro_proxy("PKR=X", "Currency", start_date, end_date)
oil    = macro_proxy("CL=F",  "Oil",      start_date, end_date)
nasdaq = macro_proxy("^IXIC", "Tech",     start_date, end_date)
```

Macro proxies come from **live yfinance calls at run time**, with `today()` as the end
date. Re-running `01_feature_engineering.py` on a different day produces **different
`data-ready/` contents**, and yfinance can silently revise history.

**This is not leakage** — all macro features are backward-looking (`shift(1)`, rolling
windows). It is a **reproducibility** gap: `data-ready/` cannot be regenerated
bit-identically, so the training run is not exactly reproducible.

**Fix (low effort, high credibility value):** cache the three macro series to
`data-ready/_macro/*.csv` on first fetch and read from cache thereafter — the same pattern
`crypto/01` already uses for its FinBERT sentiment cache.

---

## Finding 4 — PSX and MUFAP have no held-out test set 🟡 **MEDIUM**

Both use an 80/20 train/validation split where validation MAE is **both**:
- the criterion used to pick the winning model per cell, and
- the number reported as that model's accuracy and served as its confidence band.

Selecting the minimum of three numbers and then reporting that minimum as an unbiased
estimate is **optimistically biased** — you reported the best of three draws.

Commodities does this correctly: 70/10/20, with validation used only for early stopping
and all reported metrics computed on the untouched test split.

**This is pre-existing and unchanged by retraining** — the original had it too. It is not
a new error, but it is a real limitation you should disclose rather than let someone
find. Expected bias is small (best-of-3), but it is not zero.

---

## Finding 5 — Coverage gaps 🟢 **LOW**

Both already documented in `retraining_analysis.md`:

- **MUFAP `Commodity` cluster absent** — *Meezan Gold Fund* and *UBL Retirement Saving
  Fund* would forecast nothing after a swap.
- **Crypto Cluster_3 (FLOKI, PENDLE, PEPE, SUI) has no 90d/120d models** — all four
  launched late 2023; the data-sufficiency guard correctly refused to train on too little
  history.

Both degrade gracefully (horizon omitted, explained in `warnings`). Neither is a
correctness error — the second is arguably the pipeline behaving *well*.

---

## What Was Verified Clean ✅

These were actively checked, not assumed:

| Check | Result |
|---|---|
| **Commodities purge gap** | 8d @ 7d horizon, **123d @ 120d** — correct, scales with horizon |
| **MUFAP purge gap** | 8d @ 7d, **121d @ 120d** — correct |
| **Crypto purge gap** | 8d @ 7d, **121d @ 120d** — correct |
| **Training-row contamination** (commodities) | **0.0% at every horizon** |
| **Centered rolling windows** | None anywhere in any of the 4 FE scripts |
| **Negative shifts** | Only in `Target_*` construction — 4 occurrences, all correct |
| **Swing high/low detection** | Fully causal — pivot at `shift(3)` compared against `shift(1..6)`, no future bars |
| **Fair Value Gaps / BOS** | Use `shift(1)`, `shift(2)` — backward-looking |
| **Raw price columns in features** | Absent in commodities, PSX, MUFAP (present in crypto — Finding 2) |
| **Scaler fit on training fold only** | Verified all 4 engines; PSX confirmed empirically via `n_samples_seen_` |
| **PSX scaler train/serve skew** | Fixed and verified — 7/7 scalers match their 120d fold |
| **Outlier cleaning** | `clean_ohlcv()` uses row-local rules (OHLC consistency, per-row return threshold) — **no full-series statistics**, so no lookahead is introduced by cleaning |
| **Data sources** | Single authoritative source per class — PSX's own DPS portal, MUFAP's own site, Binance, TradingView. No third-party aggregators. |
| **Target construction** | `(Close.shift(-h) - Close) / Close` — correct forward return, consistent across all 4 |
| **Sequence construction** | Never crosses a ticker boundary in pooled crypto training |

---

## What To Say If Challenged

**"Is there data leakage?"**
> "Three of four engines are verified leakage-free by direct measurement — we reproduced
> the splits and computed the actual calendar gaps. Commodities, MUFAP, and crypto all
> apply a purge gap that scales with the horizon; measured contamination is 0%. We found
> one defect in PSX, where the gap is counted in rows rather than days on a pooled matrix,
> giving 6–34 days instead of up to 120. It affects 0% of training rows at 7 days and
> about 4% at 120 days. We know the exact magnitude and the fix is a few lines."

**"Why does crypto include Close when your docs say you removed prices?"**
> "That is an oversight in the new feature-engineering script — the drop list omits Close.
> It is a regression against the original design. We've quantified the impact: it acts as
> an implicit asset-identity feature in pooled clusters. It's a one-line fix plus a
> 15-minute retrain."

**"Are your results reproducible?"**
> "The training step is deterministic — fixed seeds throughout. The feature-engineering
> step is not bit-reproducible because three macro proxies are fetched live from yfinance
> at run time. We'd cache those to close it."

Volunteering these is far stronger than being caught. Every one of them was found by
auditing our own pipeline.

---

## Recommended Actions Before Presenting

**Must do (credibility-affecting):**

1. **Fix the PSX purge gap** (date-based, horizon-scaled) and retrain PSX — ~7 minutes.
   Expect long-horizon MAE to get slightly *worse*, which is the correct direction.
2. **Drop `Close` from the crypto feature set**, re-run crypto FE + training — ~15 minutes.
3. **Correct the PSX docstring**, which currently asserts a leakage guarantee it does not
   deliver.

**Should do:**

4. Cache the macro proxies for reproducibility.
5. Add an explicit limitations slide covering the PSX/MUFAP validation-as-test issue.

**Optional:**

6. Give PSX/MUFAP a true 70/10/20 split with a held-out test set. Larger change; would
   make those metrics directly comparable to commodities.

Total time for items 1–3: **under 30 minutes of compute.**

---

## Bottom Line

The project's core methodology is sound and, in several respects, more rigorous than the
original: purge gaps where there were none, proper early stopping with best-epoch
restoration, single-source data collection, scalers fit only on training folds, and a
train/serve skew closed.

Two defects were found. Both are narrow, both are quantified, and both are fixable in
under half an hour. Neither changes the project's conclusions — the PSX improvement holds
because both runs shared the same flaw, and the crypto models produce sane output despite
the extra feature.

**The honest framing:** this is a well-built pipeline with two identified bugs, one of
which contradicts a claim in its own documentation. Fix them, and the "no data leakage"
claim becomes true rather than aspirational.

---

*Audit performed 2026-08-19 against the committed source and the actual contents of
`data-ready/`, `models-new/`, and `results-new/`. Every measurement in this document is
reproducible from those files.*

---

# ADDENDUM — Both Defects Fixed and Re-Verified (2026-08-19)

Findings 1 and 2 have been fixed, the affected engines retrained, and the results
measured against pre-fix snapshots.

## Finding 1 — PSX purge gap → **RESOLVED**

`split_with_purge()` now takes `horizon_days` and gaps by `pd.Timedelta(days=horizon_days)`.
`PURGE_GAP_ROWS` is deleted, and the docstring no longer asserts a guarantee it did not
deliver.

Measured gaps after the fix (calendar days):

| Sector | 7d | 14d | 28d | 42d | 60d | 90d | 120d |
|---|---|---|---|---|---|---|---|
| Financials | 8 | 15 | 29 | 43 | 63 | 91 | **121** |
| Consumer_Autos | 10 | 17 | 31 | 45 | 61 | 91 | **123** |
| Tech_Telecom | 8 | 15 | 29 | 43 | 62 | 93 | **122** |

Every gap now ≥ its horizon. Contamination is **0%** at every horizon (was up to 4.3%).

### What the leakage was actually worth

PSX retrained (49/49 cells, 154 artifacts, 6m43s). Change in reported validation MAE,
leaky vs fixed:

| | 7d | 14d | 28d | 42d | 60d | 90d | 120d |
|---|---|---|---|---|---|---|---|
| Mean Δ | −0.3% | −0.7% | −1.9% | −0.9% | −0.4% | −3.7% | −4.5% |

Overall mean MAE **0.13334 → 0.13027 (−2.30%)**. 5 of 49 cells changed winning model.

**Important caveat on interpretation.** This is *not* a clean measurement of the leakage.
Widening the gap also **moves the validation window** — at 120d, validation now begins
~110 days later and covers a shorter, different period. Two variables changed, so the
−2.30% cannot be attributed solely to removing contamination.

**What can be claimed:** removing the contamination did not degrade the metrics, so the
leakage — capped at 4.3% of training rows — was **real but immaterial in magnitude**. The
fix was warranted for correctness and for the truthfulness of the documentation, not
because the numbers were materially inflated.

**Versus the original production baseline:** PSX now stands at **0.14945 → 0.13027, a
−12.8% error reduction** (previously reported as −10.8%).

## Finding 2 — Crypto absolute `Close` → **RESOLVED**

`Close` was added to the drop list in `crypto/01_feature_engineering.py`. Feature
engineering re-ran in 3 seconds (FinBERT sentiment served from cache — no re-scoring), and
crypto retrained (28/28 cells, 260 artifacts, 9m39s).

Verified: **0 raw price columns** remain in the crypto feature set (43 → 42 columns; the
scaler contract moved from 35 → 34 features). All 7 targets intact, all 31 tickers present.

**The cluster map is byte-identical before and after**, confirming that clustering never
depended on `Close` (it uses `BTC_Volatility_30d` and `Target_7d`). No re-pinning risk.

### Effect on model quality

| Model | with `Close` | without | Δ |
|---|---|---|---|
| **LightGBM** | 0.6852 | 0.7449 | **+0.0597** |
| **XGBoost** | 0.6427 | 0.6963 | **+0.0536** |
| **RandomForest** | 0.6498 | 0.6765 | **+0.0268** |
| CatBoost | 0.7223 | 0.7328 | +0.0105 |
| GRU | 0.8076 | 0.8172 | +0.0096 |
| NBEATS_Lite | 0.7827 | 0.7915 | +0.0088 |
| Transformer | 0.8353 | 0.8352 | −0.0000 |
| LSTM | 0.8175 | 0.8148 | −0.0027 |
| TFT_Lite | 0.8309 | 0.8214 | −0.0095 |

**Overall mean win rate 0.7527 → 0.7701 (+0.0174).**

**The pattern is exactly what theory predicts, which is the strongest evidence the
diagnosis was right.** The four **tree** models — which split on raw feature values and
were therefore using absolute `Close` as an asset-identity proxy that does not generalise
across a pooled cluster — improved substantially (LightGBM +0.060, XGBoost +0.054). The
five **neural** models, which z-score their inputs and for which `Close` was one of 35
features, are essentially unchanged (±0.01).

Ensemble health also improved. Cluster_0 previously dropped to 7–8 of 9 models above the
0.4 floor at 60–120d; it now holds **9/9 at every horizon**:

```
with Close:  Cluster_0  60d:8/9  90d:8/9  120d:7/9
without:     Cluster_0  60d:9/9  90d:9/9  120d:9/9
```

**Caveat, stated for honesty:** removing a feature changes the model, so this is a
before/after on two different feature sets rather than a controlled ablation of leakage.
But the direction, the magnitude, and — critically — the *selective* effect on trees
versus networks all match the mechanism described in Finding 2.

## Remaining open items (unchanged)

- **Finding 3** — macro proxies still fetched live at feature-engineering time
  (reproducibility, not correctness).
- **Finding 4** — PSX/MUFAP still report validation MAE as both selector and metric.
- **Finding 5** — MUFAP `Commodity` cluster absent; crypto Cluster_3 still has no 90d/120d
  models (four tokens launched late 2023; the data-sufficiency guard is behaving
  correctly).

## Final state

| Engine | Artifacts | Retrained for these fixes |
|---|---|---|
| Commodities | 420 | No — unaffected |
| Crypto | 260 | **Yes** — `01` + `02` |
| MUFAP | 88 | No — unaffected |
| PSX | 154 | **Yes** — `02` only |
| **Total** | **922** | |

Both leakage/feature defects are closed. The "no data leakage" claim in the PSX docstring
is now true rather than aspirational, and the crypto feature set once again honours the
original project's explicit rule against absolute prices.
