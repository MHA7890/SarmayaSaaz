# SarmayaSaaz

A multi-asset AI forecasting platform for Pakistani investors — commodities,
cryptocurrencies, PSX equities and MUFAP mutual funds. Seven forecast horizons
(7 / 14 / 28 / 42 / 60 / 90 / 120 days), 2,031 trained model artifacts, one
unified API, one dashboard.

---

## Features

- **Four asset classes** served through a single API contract — a commodity future
  and a Karachi mutual fund return the same response shape
- **9-model ensemble** for commodities (XGBoost, LightGBM, CatBoost, RandomForest,
  LSTM, GRU, Transformer, N-BEATS, TFT), accuracy-weighted with SHAP explainability
- **Cluster-routed crypto** — 4 K-Means clusters, 9 models per cluster, win-rate
  weighted ensembling across 20 liquid assets (6 withheld for bad history)
- **PSX equities** — 97 tickers across 7 super-sectors, single-winner routing
  (XGBoost / LightGBM / LSTM)
- **MUFAP mutual funds** — ~198 funds across 5 super-clusters (Money Market,
  Income, Balanced, Equity, Commodity), same routing architecture
- **FinBERT sentiment** — crypto forecasts incorporate multi-year NLP sentiment
  scores and Fear & Greed macro-regime context
- **News catalyst markers** on price charts via TradingView headlines + Google News RSS
- **Live price / NAV overlay** — optional yfinance quotes for commodities, crypto,
  and stocks; live MUFAP NAV scraping
- **Automated daily refresh** — collectors, feature engineering, and snapshot rebuild
  orchestrated by a single script with per-asset freshness verification

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.12.x | `pyproject.toml` requires `>=3.12, <3.13` |
| [uv](https://docs.astral.sh/uv/) | latest | Python package manager — handles the venv and lockfile |
| Node.js | 18+ | For the Next.js frontend |
| npm | 9+ | Comes with Node.js |

---

## Running it

### Option 1: 1-Click Control Center (Recommended)

Run the standalone launcher directly from the repository root:
- **Windows Executable**: Double-click **`SarmayaSaaz_Launcher.exe`** or **`SarmayaSaaz.bat`**.
- **Universal Python**: Run `python SarmayaSaaz_Launcher.pyw` (works on Windows, macOS, and Linux).

The Control Center GUI provides:
- **▶ Start Platform**: Launches Backend API (`:8000`) and Next.js Frontend (`:3000`) in background worker threads.
- **■ Stop Platform**: Safely terminates all platform services cleanly.
- **🌐 Launch Website**: Opens `http://localhost:3000` directly in your default web browser.
- **⚙ Install Dependencies**: Performs a smart pre-check and automatically installs Python (`uv sync`) and Node.js (`npm install`) dependencies if missing.

### Option 2: Manual Terminal Startup

Two processes, both started from the repo root:

```bash
# 1. Backend — http://127.0.0.1:8000  (interactive docs at /docs)
uv sync --extra dev
uv run uvicorn backend.main:app --reload --port 8000

# 2. Frontend — http://localhost:3000
cd frontend
npm install
npm run dev
```

The first backend start takes ~30 s: it deserializes four engine registries and
reads the MUFAP export to build the NAV lookup. The Next.js dev server
proxies `/api/*` to the backend automatically via `next.config.mjs`.

### Build the forecast snapshot

The **Dashboard** (Top Movers) and **Markets** pages read a precomputed snapshot
rather than forecasting all ~329 assets per request. Without it they return 503
with an explanatory message.

```bash
uv run python scripts/build_snapshot.py
```

Rebuild after retraining or when new market data lands. There is also
`POST /api/snapshot/rebuild`, which runs it in the background.

### Keep the data current

```bash
uv run python scripts/daily_update.py
```

Collects fresh closes/NAVs for every asset class, rebuilds the engineered
feature frames the forecasts read from, refreshes the snapshot, and notifies
a running API to drop its caches.

Registered via `scripts/register_daily_task.ps1` as one Windows Task Scheduler
job per market, each firing after that market closes — PSX 16:15 PKT, MUFAP
19:45 PKT, commodities and crypto 06:00 PKT.

Every step is idempotent: collectors re-fetch historical data and merge incrementally.
Strict candle enforcement ensures that traditional equity/commodity markets only store
and forecast from fully closed past trading sessions (`< today`), dropping forming candles
during active market hours. Crypto markets run 24/7 and evaluate daily closes at UTC midnight.
Full details in `docs/automation.md`.

---

## Data sources

| Asset class | Source | Authentication | History |
|---|---|---|---|
| Commodities | TradingView chart websocket feed | None (anonymous) | ~10 years daily OHLCV |
| Crypto | Binance spot klines REST API | None (public) | ~10 years daily OHLCV |
| PSX Stocks | PSX Data Portal Services (`dps.psx.com.pk`) + TradingView fallback | None (public) | ~10 years daily OHLCV |
| MUFAP Funds | mufap.com.pk FundDirectory + GetFundDetailbyAMCByDate | None (public) | ~10 years daily NAV |
| News | TradingView headlines API + Google News RSS | None | Rolling window + archive |

---

## Project layout

```
backend/
  config.py              all paths from one Settings object — nothing calls os.chdir()
  schemas.py             the API contract (Pydantic v2)
  main.py                FastAPI app, lifespan, CORS, error handlers
  ml/
    architectures.py     every PyTorch net, defined once (commodity, crypto, tabular)
    registry.py          thread-safe LRU cache over ~2031 deserialized artifacts
  engines/
    base.py              abstract Engine contract
    commodities.py       per-asset, 9-model accuracy-weighted ensemble + SHAP
    crypto.py            4 pinned K-Means clusters, win-rate weighted
    routed.py            shared base for single-winner engines (MUFAP, PSX)
    mufap.py             ~198 funds, 5 super-clusters, PKR NAV
    stocks.py            97 tickers, 7 super-sectors, PKR
    __init__.py          EngineRegistry — routes tickers, aggregates catalogs
  routers/
    assets.py            GET /api/assets — catalog + search
    forecasts.py         GET /api/forecast/{ticker} — full multi-horizon prediction
    market.py            GET /api/market, /api/movers — snapshot-backed
    models.py            GET /api/models/leaderboard, /available — performance metrics
    system.py            GET /api/health, /stats; POST /snapshot/rebuild, /data/reload
  services/
    snapshot.py          whole-universe forecast cache (results/snapshot.json)
    live_prices.py       yfinance live quotes with TTL cache
    mufap_live.py        MUFAP daily NAV table scraper
    news_live.py         TradingView + Google News RSS catalyst lookups

frontend/                Next.js 15 · React 19 · TypeScript · TanStack Query · Tailwind · Recharts
  app/
    page.tsx             Dashboard — Top Movers with horizon and class filters
    forecasts/           Single-asset forecast page with chart + panels
    market/              Market table across all asset classes
    heatmap/             Model performance heatmap / leaderboard
    methodology/         Static methodology explainer
  components/
    AssetPicker.tsx      Fuzzy search across the full asset universe
    charts/              Recharts-based price chart with catalyst markers
    forecast/            Forecast detail panels (horizons, drivers, model votes)
    nav/                 Navbar + live ticker tape
    ui/                  Loading / error / empty states

scripts/
  daily_update.py        orchestrates collect → features → snapshot → reload
  collect_commodities_tv.py   TradingView websocket collector (6 commodities)
  collect_crypto_binance.py   Binance klines collector (26 tickers)
  collect_psx_stocks.py       PSX DPS collector (97 tickers)
  collect_mufap_funds.py      MUFAP NAV history collector (~200 funds)
  collect_news.py             TradingView + Google News headline collector
  build_snapshot.py           forecasts the whole universe to results/snapshot.json
  register_daily_task.ps1     Windows Task Scheduler registration
  smoke_artifacts.py          deserialize one artifact of each format
  audit_features.py           scaler-vs-model input width audit
  verify_registry.py          predict through 162 commodity artifacts

src/                     original training / inference pipelines (stage-based)
  commodities/           stages 0-6: baseline → ML/DL → macro → sentiment → backtest
  crypto/                stages 1-7: collection → targets → macro → sentiment → training → inference → PnL
  stocks/                stages 1-5: download → features → clustering → training → inference
  mufap/                 stages 1-6: cleaning → features → clustering → training → inference → PnL
  analysis/              metric comparison, feature importance generation
  generators/            notebook generators for ML/DL across all horizons
  utils/                 data patching, live testing, China data parsing

training-scripts-new/    retrained pipelines (produce artifacts in models-new/)
  common/                shared architectures, training loops, progress utilities
  commodities/           01_feature_engineering.py, 02_train_all.py
  crypto/                01_feature_engineering.py, 02_train_all.py
  psx/                   01_feature_engineering.py, 02_train_all.py
  mufap/                 01_feature_engineering.py, 02_train_all.py

data-new/                freshly re-collected OHLCV / NAV (display + chart source)
data-ready/              engineered feature frames built from data-new/ (model inputs)
models/                  production model artifacts (~2031 files)
results/                 metrics, ensemble weights, cluster maps, snapshot
tests/                   end-to-end API tests against real artifacts (23 tests)
docs/                    automation guide, metrics, horizons, sentiment plans, future works
```

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Engine status, model cache stats |
| `GET` | `/api/stats` | Total assets, engines online, artifact count |
| `GET` | `/api/assets` | Full asset catalog with search and class filter |
| `GET` | `/api/assets/{ticker}` | Single asset metadata |
| `GET` | `/api/forecast/{ticker}` | Multi-horizon forecast with chart history, drivers, catalysts |
| `GET` | `/api/market` | Current quotes across the universe (snapshot-backed) |
| `GET` | `/api/movers` | Top gainers and losers at a given horizon |
| `GET` | `/api/models/available` | Architectures and recorded metrics per class |
| `GET` | `/api/models/leaderboard` | Per-asset/cluster model scores at one horizon |
| `GET` | `/api/snapshot` | Snapshot age and coverage |
| `POST` | `/api/snapshot/rebuild` | Trigger background snapshot rebuild |
| `POST` | `/api/data/reload` | Clear cached dataframes (after daily refresh) |

Interactive OpenAPI docs are available at `/docs` when the backend is running.

---

## Why the versions are pinned

`pyproject.toml` pins `scikit-learn==1.7.2` and `xgboost==3.4.0` deliberately.
Every `StandardScaler` and `RandomForestRegressor` in `models/` was pickled under
that exact scikit-learn version; loading them under 1.9.x raises
`InconsistentVersionWarning` and can silently change transform output. The 226
XGBoost artifacts use `SerializeToBuffer`, which is not stable even across a
patch release — loading 3.4.0 artifacts under 3.4.1 raises "input stream
corrupted". `pandas` is held below 3.0 because the `src/` pipelines rely on
APIs it removed. **Do not bump any of these without retraining and re-serializing.**

Verify artifacts still load after any dependency change:

```bash
uv run python scripts/smoke_artifacts.py   # all 5 formats deserialize
uv run python scripts/audit_features.py    # scaler vs model input widths
uv run python scripts/verify_registry.py   # 162 commodity artifacts predict
uv run python -m pytest tests/ -q          # 23 end-to-end API tests
```

> `pytest.exe` is blocked by Windows Application Control on this machine — invoke
> it as `python -m pytest`.

---

## Forecast accuracy

The PPTX presentation documents the most important empirical finding across all
engines: **forecast skill degrades with horizon length**.

### Commodities — the horizon effect

The system is meaningfully predictive on magnitude at 7–28 days (R² 0.79 → 0.39).
At long horizons it keeps a modest directional edge but essentially no magnitude
skill. 32.5% of all 378 commodity cells have R² < 0 (worse than predicting the
historical mean). Directional accuracy stays flat at 53–57% regardless of horizon.

### Architecture dominance

Same nine architectures, dramatically different winners:
- **Commodities** — tree models (XGBoost / CatBoost / LightGBM) dominate
- **Crypto** — neural nets (LSTM / GRU / Transformer / TFT) dominate with 0.93+ win rates

### MUFAP — the cleanest result

Error rises exactly along the real-world risk ordering of Pakistani mutual funds:

| Cluster | 7d MAE | 120d MAE | Character |
|---|---|---|---|
| Money Market | 0.37% | 4.03% | Tracks SBP policy rates (nearly deterministic) |
| Income | 0.41% | 4.45% | Tracks PKRV bond yields |
| Balanced | 1.40% | 7.41% | Mixed exposure |
| Equity | 3.11% | 11.14% | Volatile — tracks KSE-100 |
| Commodity | 3.24% | 13.82% | Gold-linked, volatile |

The model learned the actual risk structure of Pakistani mutual funds without being
told it.

---

## Data integrity notes

These are real defects found in the artifacts and datasets. The code works around
them explicitly rather than silently.

**Automated Stock Split & Corporate Action Adjustment.** Raw exchange feeds (e.g. PSX DPS) publish unadjusted traded prices, creating artificial single-day price drops during stock splits or bonus share distributions (such as Systems Limited's 5-for-1 bonus split in June 2025). The data pipeline automatically detects corporate action price steps (`adjust_splits` in `scripts/data_new_common.py`) and retroactively adjusts past prices and volumes so historical charts and features remain continuous.

**Mixed feature contracts (commodities).** 154 artifacts expect exactly 9 fewer
features than their scaler emits — the sentiment block — a residue of the
champion/challenger revert. Width varies by commodity (gold/gas 27, silver/crude
26, copper 25, wheat 28). The registry reads each model's true width from the
artifact and truncates to it. This is sound because `StandardScaler` normalises
columns independently and the sentiment features occupy the final columns.

**Crypto cluster map is pinned, never recomputed.** `results/crypto/cluster_map.json`
holds the asset→cluster assignment the models were trained under. K-Means labels
are arbitrary and permute between runs; recomputing moved BTC/GRT/IMX from
cluster 1 to 2 and swapped them with UNI, so Bitcoin was scored by models trained
on a single sub-$10 token. Its `MACD_Hist` z-scored to ~251,000 and the ensemble
returned +41,071% at 14 days. Regenerate the map only from a training run, never
from live data.

**Six crypto datasets are withheld.** `GRT`, `IMX`, `SUI`, `RNDR`, `APT` and `UNI`
have price history ending 490–1,596 days behind the rest of the universe. `UNI` in
particular is not Uniswap — it starts 2020-05-07, months before Uniswap launched,
and peaks at $0.60. They are excluded rather than forecast from bad history.

**No synthetic values anywhere.** The original `src/crypto/stage6_inference.py` added
`np.random.normal(0, 0.03)` to every prediction to manufacture ensemble spread.
That is removed; crypto bands are the true min/max across the models that voted.
Where a metric was never recorded, the API returns `null` and the UI renders "not
measured" rather than a plausible-looking number.

---

## Known limitations

- **R² is negative beyond ~42 days** in 32.5% of commodity cells — long-horizon
  magnitude forecasts should not be trusted
- **Mutual fund NAV growth bias** — consistent NAV growth makes it harder for the
  model to detect underlying patterns
- **Crypto training instability** — highly volatile markets make the training
  process unstable; needs guided training with proper hyperparameter tuning
- **Conflicting signals** — macro-economic features and news/sentiment data can
  conflict, confusing neural network architectures

---

## Model coverage

Metric granularity is uneven and the UI reflects that honestly — see
`GET /api/models/available`.

| Engine | Assets | Grouping | Architectures | Ensembling | Recorded metrics |
|---|---|---|---|---|---|
| Commodities | 6 | per asset | XGBoost, LightGBM, CatBoost, RandomForest, LSTM, GRU, Transformer, N-BEATS, TFT | Top-3 accuracy-weighted | Dir. accuracy, MAE, RMSE, R² — per asset |
| Crypto | 20 of 26 | 4 K-Means clusters | XGBoost, LightGBM, CatBoost, RandomForest, LSTM, GRU, Transformer, NBEATS_Lite, TFT_Lite | Win-rate weighted (floor 0.4) | Win rate — per cluster |
| PSX | 97 | 7 super-sectors | XGBoost, LightGBM, LSTM | Single winner | MAE — winner only |
| MUFAP | ~198 (77 trained clusters) | 5 super-clusters | XGBoost, LightGBM, LSTM | Single winner | MAE — winner only |

There is no PatchTST model in this system.

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12 · FastAPI · Pydantic v2 · uvicorn |
| ML runtime | PyTorch · scikit-learn · XGBoost · LightGBM · CatBoost |
| NLP | HuggingFace Transformers (FinBERT) |
| Explainability | SHAP (TreeExplainer) |
| Feature engineering | pandas · ta (technical analysis) · yfinance |
| Frontend | Next.js 15 · React 19 · TypeScript · TanStack Query · Recharts · Tailwind CSS |
| Package management | uv (Python) · npm (Node.js) |
| Task scheduling | Windows Task Scheduler via PowerShell |

---

## Retraining pipeline

The retraining pipeline under `training-scripts-new/` is built and
feature-engineered, but **training scripts have not yet been executed**.

```
Collect (data-new/) → Feature Engineer (data-ready/) → Train (models-new/) → Serve (swap in)
```

Status: feature engineering complete and schema-verified (6 commodities, 20 crypto,
97 PSX tickers, 77 funds). Improvements over the original `src/` pipelines:

- **Purge / embargo gaps** — prevents a training row's forward-looking window
  from leaking into validation
- **Early stopping + best-epoch weights** — replaces the original crypto DL's
  fixed 15-epoch loop with no early stopping
- **Fixed classifier train/serve skew** — MUFAP classifier now includes
  "sovereign" as an Income keyword, matching production

Expected outcomes: UNI becomes a real, correctly-dated asset; PEPE and TAO join;
the degenerate crypto cluster rebalances.

---

## License

Not specified. Contact the repository owner for licensing terms.

---

## Disclaimer

Forecasts are model output, not investment advice. Figures derive from historical
data and may be stale — check the "as of" date shown on every asset.
