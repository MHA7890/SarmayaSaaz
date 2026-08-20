# Retraining Analysis — `models-new/` vs `models/`

**Date of analysis:** 2026-08-19
**Scope:** all four engines retrained from `data-ready/` via `training-scripts-new/*/02_train_all.py`

---

## Verdict Up Front

**Yes — the retraining was performed correctly.** All four pipelines completed, artifact
counts match expectation exactly, and the methodology is measurably more rigorous than the
original run.

**But the headline numbers must not be read naively.** Directional accuracy went *up* while
R² went *down*, and that combination is the most important thing in this report. It is
mostly a sign of **correctness** (leakage removed), partly a sign of **trend artifact**
(the metals rallied), and it exposes a pre-existing weakness that was always there and is
now measurable.

Three defects need decisions before any swap into production. None invalidate the run.

---

## 1. Completeness Audit — Did It Actually Finish?

| Engine | Expected | Actual | Status |
|---|---|---|---|
| Commodities | 6 assets × 7h × (9 models + 1 scaler) = **420** | **420** | ✅ complete |
| PSX | 7 sectors × 7h × 3 + 7 scalers = **154** | **154** | ✅ complete |
| MUFAP | 4 clusters × 7h × 3 + 4 scalers = **88** | **88** | ✅ complete *(4 clusters, not 5 — see §5.1)* |
| Crypto | 4 clusters × 7h × (9 + 1) = **280** | **260** | ⚠️ 2 cells missing — see §5.2 |

**Total: 922 new artifacts.**

### "But the old tree has 2,031 — did we lose 1,100 models?"

**No.** The counts are not comparable, and this is worth understanding before anyone
panics in a presentation:

| Old `models/` contents | Files |
|---|---|
| `commodities/models_production/` — 7 production horizons | **420** |
| `commodities/models_production/` — legacy **1d** horizon | 54 |
| `commodities/models_production/` — scalers | 36 |
| `commodities/` — loose top-level legacy duplicates | ~1,020 |
| `psx/` — stray directory | 1 |

The new run produced **exactly 420** commodity artifacts for the 7 production horizons —
a **perfect match** with the old production set. The difference is entirely legacy
duplicates, the abandoned 1-day horizon, and dead directories. **`models-new/` is cleaner
and complete, not smaller.**

### Run timings (all four completed same day)

| Engine | Duration |
|---|---|
| Commodities | 14 min (16:33 → 16:47) |
| PSX | 7 min (17:10 → 17:17) |
| MUFAP | 10 min (17:20 → 17:30) |
| Crypto | 10 min (17:35 → 17:45) |

Roughly 41 minutes total on the A6000. Fast because the datasets are small and early
stopping fired often.

---

## 2. Commodities — The Most Complex Result

### Aggregate by horizon

| Horizon | Dir. Acc old | new | Δ | R² old | new | Δ |
|---|---|---|---|---|---|---|
| 7d | 52.95% | 54.97% | **+2.02** | +0.789 | +0.759 | −0.030 |
| 14d | 54.39% | 54.47% | +0.08 | +0.639 | +0.574 | −0.065 |
| 28d | 54.59% | 55.06% | +0.46 | +0.394 | +0.272 | −0.123 |
| 42d | 54.78% | 56.64% | **+1.87** | +0.166 | **−0.134** | −0.299 |
| 60d | 56.58% | 57.60% | +1.02 | −0.137 | −0.534 | −0.397 |
| 90d | 54.44% | 62.08% | **+7.64** | −0.459 | −0.843 | −0.384 |
| 120d | 56.35% | 64.83% | **+8.48** | −0.713 | **−1.268** | −0.555 |

**Overall: directional accuracy 54.87% → 57.95% (+3.08). R² +0.097 → −0.168 (−0.265).
Cells with R² < 0 rose from 123/378 (32.5%) to 149/378 (39.4%).**

### Why R² fell — this is expected and *good*

The new pipeline adds a **purge/embargo gap** at every split boundary, sized to the
horizon. The original notebooks had none. Without a gap, a training row's forward-looking
target window overlaps dates appearing in validation/test — the model indirectly sees what
it is being graded on.

**The old R² was inflated by leakage. The new R² is the honest number.** A drop here is
evidence the fix worked, not evidence of regression. The degradation scales with horizon
(−0.03 at 7d, −0.56 at 120d) exactly as leakage theory predicts: longer horizons had wider
target windows and therefore more overlap to remove.

### Why directional accuracy rose — this is *mostly artifact*

Per-asset breakdown reveals the gain is not uniform:

| Asset | Dir. Acc (all horizons) old → new | Δ | at 120d |
|---|---|---|---|
| **gold** | 57.29% → 67.91% | **+10.63** | **+16.61** |
| **copper** | 52.13% → 59.33% | **+7.21** | **+16.28** |
| **silver** | 60.26% → 63.66% | +3.40 | **+13.36** |
| crude_oil | 50.14% → 52.24% | +2.10 | +5.90 |
| natural_gas | 51.45% → 49.71% | **−1.75** | −0.10 |
| wheat | 57.95% → 54.85% | **−3.10** | −1.17 |

**The entire gain comes from the three metals, concentrated at long horizons.** Gold,
copper, and silver rallied strongly through the extended `data-new/` window (which runs to
2026-08-18). A model that learns "metals go up" scores high directional accuracy in a bull
market **without predictive skill**. Natural gas and wheat — which did not trend — got
*worse*.

**Do not present "+3.08% directional accuracy" as a modelling improvement.** It is
predominantly a change in the test-period market regime.

### The finding that matters most

`Improvement_Pct` measures performance against a naive "assume the price does not change"
baseline. **Negative means worse than doing nothing.**

| Horizon | old | new |
|---|---|---|
| 7d | −3.40% | −3.02% |
| 14d | −5.73% | −6.95% |
| 28d | −4.96% | −8.28% |
| 42d | −5.41% | −11.17% |
| 60d | −6.78% | −15.25% |
| 90d | −17.27% | −10.25% |
| 120d | −12.22% | −5.99% |

**Every horizon, both runs, is negative.** On *magnitude* (MAE), the commodity models do
not beat a naive no-change forecast. They have a genuine **directional** edge (55–65%,
above coin flip) but they are not better than "assume flat" at predicting *how much*.

This was true of the old models too — retraining did not cause it, it made it visible.
This is the honest ceiling of the current commodity approach and should be stated plainly
rather than discovered by an examiner.

### Architecture rankings shifted toward neural networks

| Model | Mean Dir. Acc old → new | Δ |
|---|---|---|
| TFT | 48.19% → 56.29% | **+8.10** |
| Transformer | 50.57% → 58.27% | **+7.69** |
| GRU | 51.79% → 56.34% | +4.55 |
| RandomForest | 53.46% → 57.65% | +4.19 |
| N-BEATS | 54.70% → 57.12% | +2.41 |
| XGBoost | 58.74% → 60.23% | +1.49 |
| LightGBM | 59.84% → 60.25% | +0.41 |
| LSTM | 55.69% → 55.16% | −0.53 |
| CatBoost | 60.85% → 60.26% | −0.59 |

Cell wins: **OLD** XGBoost 15, TFT 5, LSTM 5 → **NEW** XGBoost 11, **GRU 9**, Transformer 5.

**Interpretation:** the deep models gained most, and the reason is in
`common/dl_train.py` — 100 epochs with patience 22 and **best-epoch checkpointing**,
versus the original's shorter schedules. The old TFT/Transformer numbers (48–50%, below
coin flip) indicate **undertrained** models, not unsuitable ones. Given a proper training
budget they became competitive. Trees, already near their ceiling, barely moved.

This is a legitimate methodological win and safe to claim.

---

## 3. PSX — Clear, Legitimate Improvement

**Mean validation MAE: 0.14945 → 0.13334 — a 10.8% error reduction.**

Change in MAE by sector (negative = better):

| Sector | 7d | 14d | 28d | 42d | 60d | 90d | 120d |
|---|---|---|---|---|---|---|---|
| Cement_Construction | +0.9% | −0.7% | −1.9% | −1.2% | +0.9% | +8.2% | **−12.1%** |
| Consumer_Autos | +1.4% | +0.9% | +1.2% | +2.7% | −2.1% | −4.9% | **−22.2%** |
| Energy_Power | +3.3% | −0.4% | −5.1% | −18.0% | **−29.4%** | **−32.2%** | **−35.8%** |
| Fertilizers_Chemicals | +2.1% | +0.7% | −0.2% | +0.1% | −2.8% | −7.6% | −16.7% |
| Financials | +4.9% | +2.9% | +3.3% | −1.9% | −1.3% | +2.1% | −6.0% |
| Pharmaceuticals | −5.2% | −11.2% | −20.4% | −29.3% | **−41.6%** | **−44.8%** | −19.2% |
| Tech_Telecom | −7.9% | −10.9% | −6.6% | −6.1% | −8.6% | −6.4% | −11.0% |

**The pattern is consistent: long horizons improved dramatically, short horizons are flat
or marginally worse.** Pharmaceuticals at 90d improved 44.8%; Energy_Power at 120d
improved 35.8%. Short-horizon regressions are small (+1% to +5%) and confined to sectors
that were already accurate.

**Winner shift: LightGBM 25 / LSTM 24 → LightGBM 9 / LSTM 40.** The LSTM now wins 82% of
cells. Same cause as commodities: the original PSX LSTM trained 200 epochs with patience
15; the new one trains 100 with patience 22 and restores best-epoch weights, which
suppresses the overfit tail.

### On the scaler fix — an honest correction

I fixed a train/serve skew in `training-scripts-new/psx/02_train_all.py` where the scaler
was refit per horizon but only the last (120d) survived on disk. **The fix is verified
applied** — all 7 scalers have `n_samples_seen_` exactly matching their 120d training fold.

**However, on this dataset the fix changed nothing numerically.** PSX feature engineering
already drops every row with a NaN target, so *all seven* `Target_{h}d` columns have an
identical NaN mask (verified: 0 NaN in all of them). Every horizon's fold is therefore the
same rows, and the old code produced an identical scaler by coincidence.

**The fix is still correct and worth keeping** — it guarantees the invariant instead of
relying on an accident of the upstream feature-engineering step, which a future change
could silently break. But **the 10.8% PSX improvement is not attributable to it.** That
improvement comes from the retraining itself. I want that stated clearly so no one
credits the wrong cause in a presentation.

---

## 4. MUFAP — Best Result in the Run

**Mean validation MAE: 0.03929 → 0.03513 — a 10.6% error reduction.**

| Cluster | 7d | 14d | 28d | 42d | 60d | 90d | 120d |
|---|---|---|---|---|---|---|---|
| Balanced | −15.2% | −16.3% | −12.6% | −13.9% | −10.3% | −9.8% | −11.4% |
| Equity | −14.1% | −13.6% | −10.5% | −9.4% | −7.6% | −11.4% | −6.2% |
| Income | −8.7% | −14.1% | −10.0% | −11.7% | −5.6% | −2.5% | **+5.1%** |
| MoneyMarket | −5.1% | **−24.6%** | −18.6% | **−30.5%** | −25.4% | −18.8% | −11.3% |
| Commodity | **MISSING** | | | | | | |

**Nearly every cell improved** — 27 of 28, with only Income@120d marginally worse. This is
the cleanest, most trustworthy improvement in the entire run.

New absolute error, and the risk ordering still holds perfectly:

| Cluster | 7d MAE | 120d MAE |
|---|---|---|
| MoneyMarket | **0.0035 (0.35%)** | 0.0357 (3.57%) |
| Income | 0.0037 (0.37%) | 0.0467 (4.67%) |
| Balanced | 0.0119 (1.19%) | 0.0656 (6.56%) |
| Equity | 0.0267 (2.67%) | 0.1045 (10.45%) |

The ordering **MoneyMarket < Income < Balanced < Equity** still exactly matches real-world
fund risk ordering — the sanity check survives retraining, which is strong evidence the
models are learning genuine structure.

**Winner shift: LightGBM 16 / XGBoost 14 / LSTM 5 → LightGBM 9 / LSTM 14 / XGBoost 5.**
Same LSTM-favouring pattern.

---

## 5. The Three Defects Requiring Decisions

### 5.1 MUFAP lost its Commodity cluster ⚠️ *(predicted before the run)*

`data-ready/mufap/` contains only 4 clusters. The old routing table has 5. Two funds are
affected:

- **Meezan Gold Fund**
- **UBL Retirement Saving Fund**

`backend/engines/mufap.py::classify()` still routes any category containing `"commodit"`
or `"gold"` to `"Commodity"`. After a swap, `RoutedEngine._route()` finds no entry and
these funds emit `no model registered for 'Commodity'` for all 7 horizons — they would
appear in the catalog but forecast nothing.

**Options:** (a) re-run MUFAP feature engineering to emit a Commodity cluster;
(b) reclassify those two funds into `Balanced`; (c) accept and document a 2-fund coverage
gap. **Recommended: (b)** — 2 funds do not justify a 5th cluster with so few rows, and
`Balanced` is the closest behavioural match.

### 5.2 Crypto Cluster_3 has no 90d or 120d models ⚠️

`Cluster_3` = **FLOKI, PENDLE, PEPE, SUI** — all launched late 2023:

| Ticker | Rows | History starts |
|---|---|---|
| FLOKI | 1,001 | 2023-11-20 |
| PEPE | 1,003 | 2023-11-20 |
| SUI | 1,005 | 2023-11-18 |
| PENDLE | 944 | 2024-01-18 |

After dropping NaN targets at 90d/120d, applying the 80/20 split, the horizon-sized purge
gap, and a 30-day sequence window, `prepare_cluster_horizon()` returned `None` — the
guard fired correctly and skipped the cell rather than training on garbage.

**This is the code behaving properly, not a bug.** But it means those four assets will
have no 90d or 120d forecast. The engine handles this gracefully (the horizon is omitted
and explained in `warnings`), so it degrades safely.

**Options:** (a) accept — these are young, highly speculative memecoins where a 6-month
forecast is dubious anyway; (b) reduce `SEQ_LEN` for this cluster; (c) merge Cluster_3
into a neighbour. **Recommended: (a)**, documented.

### 5.3 The R² regression is expected — but must be communicated

Covered in §2. Not a defect in execution; a defect in how it will be *read* if presented
without explanation.

---

## 6. What Genuinely Improved — Crypto Clustering

The single clearest structural win. Old clustering was degenerate:

| | Old | New |
|---|---|---|
| Cluster_0 | 12 assets | 7 |
| Cluster_1 | **3** (BTC, GRT, IMX) | 9 |
| Cluster_2 | **1** (UNI — withheld, served nobody) | 11 |
| Cluster_3 | 13 | 4 |
| **Universe** | 29 tickers, 23 served | **31 tickers** |

New membership:

- **Cluster_0 (7):** APT, BONK, IMX, LDO, RNDR, TAO, WIF
- **Cluster_1 (9):** ADA, BNB, BTC, DOGE, ETH, FET, LINK, SOL, TRX
- **Cluster_2 (11):** AAVE, AVAX, CRV, FIL, GRT, INJ, MKR, NEAR, SHIB, SNX, UNI
- **Cluster_3 (4):** FLOKI, PENDLE, PEPE, SUI

The old 1-asset cluster is gone. **PEPE and TAO joined** (previously dropped for thin
history), and **UNI is now genuine Uniswap** — the new Binance series starts 2020-09-17 at
$0.30 and runs to 2026-08-18 at $3.287, versus the old mislabeled file starting 2020-05-07
at $0.00075. Cluster_3 is now a coherent "recent memecoin/new-launch" group rather than
an arbitrary partition.

### But crypto win rates fell

| Model | old | new | Δ |
|---|---|---|---|
| Transformer | 0.9352 | 0.8353 | −0.0999 |
| TFT_Lite | 0.9361 | 0.8309 | −0.1052 |
| LSTM | 0.9312 | 0.8175 | −0.1137 |
| GRU | 0.9315 | 0.8076 | −0.1239 |
| NBEATS_Lite | 0.9156 | 0.7827 | −0.1329 |
| CatBoost | 0.6558 | 0.7223 | **+0.0665** |
| LightGBM | 0.6310 | 0.6852 | **+0.0542** |
| XGBoost | 0.6179 | 0.6427 | **+0.0248** |
| RandomForest | 0.6917 | 0.6498 | −0.0419 |

**This is not a regression — it is the clustering fix showing up in the metric.** Old
Cluster_1 held 3 assets and Cluster_2 held 1; fitting a near-homogeneous group is easy, so
return-MAE was tiny and win rate near 0.94. The new clusters pool 7–11 genuinely diverse
assets, which is harder and produces honestly higher MAE.

Recall from the win-rate analysis that `win_rate = max(0.4, 1 − MAE_on_returns)` rewards
predicting near zero. The **narrowing gap between DL (0.78–0.84) and trees (0.64–0.72)**
suggests the new models are producing less degenerate near-zero output. That is a
qualitative improvement the metric penalises.

Ensemble health also improved: old Cluster_0 dropped to 5/9 models alive at 120d; new
holds 7/9. Old Cluster_2 fell to 5/9 at 42–90d; new holds 7–9/9.

---

## 7. Was the Retraining Methodologically Correct?

**Yes.** Assessment against each deliberate change:

| Change | Verified | Assessment |
|---|---|---|
| Purge/embargo gap at split boundaries | ✅ `common/splits.py`, applied in all four | **Correct.** The R² drop is the expected signature of removing leakage. |
| 100 epochs, patience 22, best-epoch restore | ✅ `common/dl_train.py` | **Correct.** Explains DL gains across all engines. |
| Scaler fit on training fold only | ✅ verified per engine | **Correct.** No scaler sees validation data. |
| MUFAP `"sovereign"` keyword added | ✅ matches production `classify()` | **Correct.** Closes a real train/serve skew. |
| PSX single scaler per sector | ✅ `n_samples_seen_` matches 120d fold, 7/7 | **Correct** — though numerically a no-op here (§3). |
| Chronological splits, never random | ✅ | **Correct.** |
| Metrics computed on held-out data | ⚠️ commodities use a true test split; **PSX/MUFAP still report validation MAE**, which is also the selection criterion | **Pre-existing limitation, unchanged.** Mildly optimistic. |

### The one confound you must acknowledge

**This is not a controlled experiment.** Two things changed simultaneously:

1. **The methodology** (purge gaps, longer DL training, best-epoch checkpointing)
2. **The data** (`data/` → `data-ready/`, rebuilt from re-collected single-source
   `data-new/`, with a longer window extending to 2026-08-18)

**You cannot cleanly attribute any individual metric change to either cause.** The gold
directional gain is almost certainly data (metals rallied). The DL gains are almost
certainly methodology (undertrained models given a proper budget). But the split cannot be
proven without an ablation.

If asked "did your changes improve the model?", the honest answer is: *"The methodology is
strictly more rigorous and the data is strictly cleaner. PSX and MUFAP improved ~10.7% on
identical metrics, which is a real gain. The commodity directional gain is confounded with
a market regime change and I would not claim it."*

---

## 8. Recommendation

**Do not swap `models-new/` into production yet.** Sequence:

1. **Resolve MUFAP Commodity** (§5.1) — reclassify the 2 funds or accept the gap.
2. **Decide on crypto Cluster_3 long horizons** (§5.2) — accepting is defensible.
3. **Map the paths correctly** — the trainer writes `models-new/psx/` but the backend
   reads `models/stocks/`; `models-new/commodities/` maps to
   `models/commodities/models_production/`. `crypto/` and `mufap/` map directly.
4. **Copy `results-new/crypto/cluster_map.json` alongside the crypto models.** Non-negotiable
   — it is the pinned map these models were trained under. Serving new models against the
   old map reproduces the +41,071% Bitcoin bug exactly.
5. **Update `backend/engines/crypto.py`'s quarantine expectations** — `_quarantine()` reads
   `data/crypto/*.csv` (old, stale). With `data-new/` the 6 withheld assets should now
   pass. This needs checking or the fix will not surface to users.
6. **Rebuild the snapshot** — `uv run python scripts/build_snapshot.py`.
7. **Re-run the test suite** — several tests assert current behaviour (`UNI` must 404,
   crypto count ≥ 23, mutual_fund ≥ 178). Those assertions will need updating, and that is
   expected.

**Keep `models/` intact until the new set passes the full suite.**

---

## 9. Summary Table

| Engine | Metric | Old | New | Change | Verdict |
|---|---|---|---|---|---|
| Commodities | Dir. Accuracy | 54.87% | 57.95% | +3.08 | ⚠️ Confounded by metal rally |
| Commodities | R² | +0.097 | −0.168 | −0.265 | ✅ Honest number; leakage removed |
| Commodities | vs naive baseline | −8.0% | −8.7% | — | ❌ Both worse than "assume flat" |
| PSX | Validation MAE | 0.14945 | 0.13334 | **−10.8%** | ✅ Real improvement |
| MUFAP | Validation MAE | 0.03929 | 0.03513 | **−10.6%** | ✅ Real improvement, 27/28 cells |
| Crypto | Cluster balance | 12/3/1/13 | 7/9/11/4 | — | ✅ Major structural fix |
| Crypto | Universe | 29 (23 served) | 31 | +2 | ✅ UNI now genuine |
| Crypto | Mean DL win rate | 0.93 | 0.82 | −0.11 | ✅ Expected; harder clusters |
| All | Artifacts | 420 prod. commodities | 420 | — | ✅ Complete |

---

*Analysis performed against `results/` and `results-new/` on 2026-08-19. Every figure is
reproducible from the JSON metric files in those directories.*
