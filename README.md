# SarmayaSaaz

Multi-asset AI forecasting across Pakistani and global markets — commodities,
cryptocurrencies, PSX equities and MUFAP mutual funds. Seven horizons (7/14/28/42/60/90/120
days), ~2,000 trained model artifacts, one API, one dashboard.

---

## Running it

Two processes. Both from the repo root.

```bash
# 1. Backend — http://127.0.0.1:8000  (docs at /docs)
uv sync --extra dev
uv run uvicorn backend.main:app --reload --port 8000

# 2. Frontend — http://localhost:3000
cd frontend
npm install
npm run dev
```

The first backend start takes ~30s: it deserializes four engine registries and reads a
94MB MUFAP export to build the NAV lookup.

### Build the forecast snapshot

The **Markets** and **Top Movers** pages read a precomputed snapshot rather than forecasting
329 assets per request. Without it they return 503 with an explanatory message.

```bash
uv run python scripts/build_snapshot.py
```

Rebuild after retraining or when new market data lands. There is also
`POST /api/snapshot/rebuild`, which runs it in the background.

### Keep the data current

```bash
uv run python scripts/daily_update.py
```

Collects fresh closes/NAVs for every asset class, rebuilds the feature frames
the forecasts read, and refreshes the snapshot.

Registered via `scripts/register_daily_task.ps1` as one Windows task per market,
each firing after that market closes — PSX 16:15 PKT, MUFAP 19:45 PKT,
commodities and crypto 06:00 PKT (their trading day ends after midnight local).

A session is only written once it has genuinely ended, so an in-progress price
can never be recorded as a close. Full details, including the one known gap
(MUFAP forecasts do not advance yet), are in `docs/automation.md`.

---

## Layout

```
backend/
  config.py          all paths, from one settings object — nothing calls os.chdir()
  schemas.py         the API contract
  ml/
    architectures.py every PyTorch net, defined once
    registry.py      LRU cache over deserialized artifacts + input-width resolution
  engines/
    base.py          the Engine contract
    commodities.py   per-asset, 9-model accuracy-weighted ensemble
    crypto.py        4 pinned clusters, win-rate weighted
    routed.py        shared base for single-winner engines
    mufap.py         5 super-clusters, PKR NAV
    stocks.py        7 super-sectors, PKR
  services/snapshot.py
  routers/           system, assets, forecasts, market, models
frontend/            Next.js 15 · TanStack Query · Tailwind · Recharts
scripts/             diagnostics + snapshot builder
tests/               end-to-end API tests against real artifacts
```

---

## Why the versions are pinned

`pyproject.toml` pins `scikit-learn==1.7.2` deliberately. Every `StandardScaler` and
`RandomForestRegressor` in `models/` was pickled under that exact version; loading them
under 1.9.x raises `InconsistentVersionWarning` and can silently change transform output.
`pandas` is held below 3.0 for the same reason — the `src/` pipelines rely on APIs it
removed. **Do not bump either without retraining and re-serializing.**

Verify artifacts still load after any dependency change:

```bash
uv run python scripts/smoke_artifacts.py   # all 5 formats deserialize
uv run python scripts/audit_features.py    # scaler vs model input widths
uv run python scripts/verify_registry.py   # 162 commodity artifacts predict
uv run python -m pytest tests/ -q          # 23 end-to-end API tests
```

> `pytest.exe` is blocked by Windows Application Control on this machine — invoke it as
> `python -m pytest`.

---

## Data integrity notes

These are real defects found in the artifacts and datasets. The code works around them
explicitly rather than silently.

**Mixed feature contracts (commodities).** 154 artifacts expect exactly 9 fewer features
than their scaler emits — the sentiment block — a residue of the champion/challenger
revert. Width varies by commodity (gold/gas 27, silver/crude 26, copper 25, wheat 28).
The registry reads each model's true width from the artifact and truncates to it. This is
sound because `StandardScaler` normalises columns independently and the sentiment features
occupy the final columns. The previous implementation hardcoded `27`, so reverted copper,
silver and wheat models raised on shape and were swallowed by a bare `except` — those
commodities were quietly predicting from a partial ensemble.

**Crypto cluster map is pinned, never recomputed.** `results/crypto/cluster_map.json` holds
the asset→cluster assignment the models were trained under. K-Means labels are arbitrary and
permute between runs; recomputing moved BTC/GRT/IMX from cluster 1 to 2 and swapped them with
UNI, so Bitcoin was scored by models trained on a single sub-$10 token. Its `MACD_Hist`
z-scored to ~251,000 and the ensemble returned +41,071% at 14 days. Regenerate the map only
from a training run, never from live data.

**Six crypto datasets are withheld.** `GRT`, `IMX`, `SUI`, `RNDR`, `APT` and `UNI` have price
history ending 490–1,596 days behind the rest of the universe. `UNI` in particular is not
Uniswap — it starts 2020-05-07, months before Uniswap launched, and peaks at $0.60. They are
excluded rather than forecast from bad history.

**No synthetic values anywhere.** `src/crypto/stage6_inference.py` added
`np.random.normal(0, 0.03)` to every prediction to manufacture ensemble spread for the
dashboard. That is removed; crypto bands are the true min/max across the models that voted.
Where a metric was never recorded, the API returns `null` and the UI renders "not measured"
rather than a plausible-looking number.

---

## Model coverage

Metric granularity is uneven and the UI reflects that honestly — see `/api/models/available`.

| Engine | Assets | Grouping | Architectures | Recorded metrics |
|---|---|---|---|---|
| Commodities | 6 | per asset | 9 | Dir. accuracy, MAE, RMSE, R² — per asset |
| Crypto | 23 of 29 | 4 clusters | 9 | win rate — per cluster |
| PSX | 95 | 7 sectors | 3 | MAE — winner only |
| MUFAP | 198 | 5 clusters | 3 | MAE — winner only |

There is no PatchTST model in this system. The architectures are XGBoost, LightGBM,
CatBoost, RandomForest, LSTM, GRU, Transformer, N-BEATS and TFT.

---

## Disclaimer

Forecasts are model output, not investment advice. Figures derive from historical data and
may be stale — check the "as of" date shown on every asset.
