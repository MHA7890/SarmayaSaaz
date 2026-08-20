# Retraining on data-new/

Reproduces the four original training pipelines (crypto, PSX stocks, MUFAP,
commodities) against the freshly collected, single-source, null/outlier-
cleaned data in `data-new/`, instead of the old `data/` datasets that had
cross-source discrepancies.

## Layout

```
training-scripts-new/
  common/                  shared code every asset class imports
    progress.py            StageProgress ticker + epoch_bar() for DL loops
    metrics.py              price-unit MAE/RMSE/MAPE/Dir_Acc/R2 + win_rate
    splits.py               leak-safe chronological splitting w/ purge gaps
    architectures.py        every PyTorch model class, one definition each
    dl_train.py              shared training loop: 100 epochs, patience 22
  crypto/
    01_feature_engineering.py   data-new/crypto-data       -> data-ready/crypto
    02_train_all.py              9 models x 4 clusters x 7 horizons
  psx/
    01_feature_engineering.py   data-new/psx-data          -> data-ready/psx/<sector>
    02_train_all.py              3 models x 7 sectors x 7 horizons
  mufap/
    01_feature_engineering.py   data-new/mufap-data        -> data-ready/mufap/<cluster>
    02_train_all.py              3 models x 5 clusters x 7 horizons
  commodities/
    01_feature_engineering.py   data-new/commodities-data  -> data-ready/commodities
    02_train_all.py              9 models x 6 assets x 7 horizons
```

Each `01_feature_engineering.py` has already been run once (see below) - its
output already sits in `data-ready/`. Each `02_train_all.py` has **not**
been run; that step is left for you.

## Running

From the repo root, with the project's venv active:

```
uv run python training-scripts-new/crypto/01_feature_engineering.py
uv run python training-scripts-new/crypto/02_train_all.py

uv run python training-scripts-new/psx/01_feature_engineering.py
uv run python training-scripts-new/psx/02_train_all.py

uv run python training-scripts-new/mufap/01_feature_engineering.py
uv run python training-scripts-new/mufap/02_train_all.py

uv run python training-scripts-new/commodities/01_feature_engineering.py
uv run python training-scripts-new/commodities/02_train_all.py
```

Each is independent - run whichever asset class you want, in either order,
any time. Re-running a `01_` script is safe (idempotent, overwrites its
output); re-running a `02_` script re-trains and overwrites that asset
class's artifacts.

Every script prints a live progress ticker for its outer loop (item, %,
elapsed, ETA) and, for every neural net, a `tqdm` epoch bar with running
train/val loss - nothing runs silently.

## Output locations

- `data-ready/<class>/...` - engineered features, ready to train on (this is
  the "data files ready for training" you asked for).
- `models-new/<class>/...` - trained model artifacts (mirrors the existing
  `models/<class>/` layout and naming convention exactly, so it can later
  replace `models/` if you're happy with the new run - it's kept separate
  for now so nothing overwrites the currently-serving artifacts).
- `results-new/<class>/...` - metrics/ensemble-weights/cluster-map JSON
  (mirrors `results/<class>/`, same reasoning).

## What's faithfully reproduced vs. deliberately changed

**Same as before:** every model family (XGBoost/LightGBM/CatBoost/
RandomForest + LSTM/GRU/Transformer/N-BEATS/TFT for crypto and commodities;
XGBoost/LightGBM/tabular-LSTM for PSX/MUFAP), the same clustering/grouping
logic (K-Means on [volatility, 7d return] for crypto, static sector map for
PSX, category classifier for MUFAP, independent per-asset for commodities),
the same technical-indicator formulas, the same metric definitions (MAE in
native price units, directional accuracy vs. prior close, R2/RMSE/MAPE for
commodities, win_rate = max(0.4, 1-MAE) for crypto, lowest-val-MAE-wins
routing for PSX/MUFAP).

**Deliberately changed, per explicit instruction:**
- Every neural net now trains up to **100 epochs with early stopping,
  patience 22** (the middle of the requested 20-25 range), regardless of
  what each original pipeline used individually (some had no early stopping
  at all, some patience 15). See `common/dl_train.py`.
- Commodities training now applies a **purge/embargo gap** at the
  train/val/test split boundaries (sized to the horizon), which the
  original notebooks did not. Without it, a training row's forward-looking
  target window can span into the validation period.
- MUFAP's cluster classifier includes `"sovereign"` as an Income keyword,
  matching the *production* `backend/engines/mufap.py::classify()` - the
  original training-time classifier omitted it, a real train/serve skew.

**Data sourcing:** every technical/price-derived feature comes straight from
`data-new/`. Macro data that's a cheap, no-key-needed live pull (PKR/oil/
NASDAQ/S&P500 via yfinance, Fear&Greed via the free alternative.me history
already cached in `data/crypto_raw/`) is reproduced live, same as the
originals. Macro/sentiment data that needs a paid API key or heavy NLP
inference over data that isn't price data (commodity macro spreadsheets,
FinBERT-scored news) is reused as-is from `data/` - it was never part of
what `data-new/` was collected to fix, and re-scoring already-scored news
would just reproduce the same numbers at real compute cost. Crypto's raw
news headlines had never been scored, so `crypto/01_feature_engineering.py`
does run FinBERT once over them (cached to
`data-ready/crypto/_cache/sentiment_daily.csv` afterwards).

## A structural note on MUFAP/PSX coverage

Some very recently-launched funds/plans in `data-new/mufap-data/` have too
little history (under ~200 rows once rolling-window warmup is accounted for)
to pass the same minimum-history filter the original pipeline used - these
are skipped by `01_feature_engineering.py` and logged, not silently dropped.
