# Automated daily data refresh

One command keeps every asset current:

```bash
uv run python scripts/daily_update.py
```

It runs, per asset class, `collect` → `features`, then rebuilds the forecast
snapshot once at the end. Everything is idempotent — the collectors re-fetch the
full 10-year window and rewrite each CSV, so a missed day self-heals on the next
run and there is no incremental state to corrupt.

---

## Why both layers have to move

There are two data layers, and refreshing only the obvious one leaves the app
half-updated:

| Layer | Path | Drives |
|---|---|---|
| Display | `data-new/` | the close/NAV and chart you see |
| Model input | `data-ready/` | what the forecast is actually computed from |

Refresh `data-new` alone and the quoted price advances daily while every
forecast keeps running off a frozen feature frame. It looks correct and isn't —
which is exactly what MUFAP did until its swap, described below.
So each class runs the collector *and* its `01_feature_engineering.py`.

---

### Rows that are real but self-contradictory

PSX publishes tick-rounding artifacts on thin days: IBFL closed 2026-08-19 at
265.03 with a `Low` of 265.05 — over by 0.02, on 43 shares traded. The OHLC
consistency rule dropped the whole session, leaving that one ticker a day
behind the market. But PSX *also* publishes genuinely broken prints: across
IBFL's history those violations reach 104% of the close.

`clean_ohlcv` therefore clamps violations within `OHLC_CLAMP_TOLERANCE`
(0.05% of close) — widening High/Low to contain Open and Close, preserving the
close — and still drops anything larger. For IBFL that admits 12 rounding rows
and rejects the other 214.

---

## Never recording an unfinished session

A source will happily hand back a row for a session that hasn't closed.
Verified live: PSX's DPS endpoint queried at 11:25 PKT with the market open
returned HBL dated that same day with `Close = 322.25` on 112k volume against a
typical 600k–1.8M. Six minutes later the same "close" read 322.01 — it was the
live price. MUFAP is milder but ragged: some funds publish today's NAV by
mid-morning while others still end yesterday.

`drop_unclosed_sessions()` in `scripts/data_new_common.py` decides what counts
as finished. It is **close-aware**: given the session's end time, today's row is
kept once that time has passed plus a 30-minute buffer, because the session is
genuinely over and the close is final. That is what lets a collector run right
after the closing bell publish the same day instead of waiting for tomorrow.

| Source | Timezone | Session close | Today's row kept from |
|---|---|---|---|
| PSX | `Asia/Karachi` | 15:30 | 16:00 PKT |
| MUFAP | `Asia/Karachi` | 19:00 (NAV publication) | 19:30 PKT |
| commodities, crypto | UTC | — | never; see below |

Without a session close the rule is the conservative one — only dates strictly
before today are final. That is correct for the 24h feeds, whose trading day
ends *after* local midnight: a crypto UTC day closes at 00:00 UTC the following
day, so "today" is never final in UTC terms.

Verified at simulated clock times: a PSX run at 11:25 or 15:45 drops today's
row, one at 16:05 keeps it.

---

## Scheduling

One task per market, each firing shortly after that market's session ends, so
every class publishes its close the same day rather than waiting for a single
overnight batch:

| Task | Time (PKT) | Classes | Why then |
|---|---|---|---|
| `SarmayaSaaz Refresh (PSX)` | 16:15 | psx | board closes 15:30 |
| `SarmayaSaaz Refresh (MUFAP)` | 19:45 | mufap | NAV publication complete by 19:00 |
| `SarmayaSaaz Refresh (Global)` | 06:00 | commodities, crypto | previous UTC day final by 05:00 PKT |

Commodities and crypto cannot publish "same day" in any meaningful sense — US
commodities settle ~02:00 PKT and the crypto UTC day rolls at 05:00 PKT, both
*after* midnight locally. The 06:00 run is the first moment their previous
session is final.

Every run ends with a snapshot rebuild, so the dashboard reflects whichever
class just refreshed.

Tasks are registered with `-StartWhenAvailable`, so a run missed while the
machine was off fires at the next opportunity rather than being skipped.

```powershell
# inspect all three
Get-ScheduledTask | Where-Object { $_.TaskName -like 'SarmayaSaaz*' }

# run one now / check last result
Start-ScheduledTask   -TaskName "SarmayaSaaz Refresh (PSX)"
Get-ScheduledTaskInfo -TaskName "SarmayaSaaz Refresh (PSX)"
```

Register or re-register with `scripts/register_daily_task.ps1`; remove them all
with the same script and `-Remove`.

---

## Failure handling

Three independent guards, because a step exiting 0 is not proof the data moved:

1. **Exit codes.** Collectors *and* feature scripts used to exit 0 regardless.
   A real collect run reported *"4 ok, 2 failed"* for gold and copper and still
   looked successful; feature engineering skipped copper and crude_oil for
   months of wall-clock because `openpyxl` was missing for their `.xlsx` macro
   inputs, and reported success. Both now exit non-zero on real failure.

   Feature scripts distinguish **skipped** (thin history — a fund with 38 rows
   is expected and benign) from **FAILED** (an exception). Only the latter
   fails the run. MUFAP legitimately skips 29 short-history funds every night;
   failing on that would train everyone to ignore failures.

2. **Per-asset freshness.** `daily_update.py` reads what landed on disk and
   reports `26/26 @ 2026-08-19` per layer, naming any asset behind its class's
   newest date. Checking only the class *maximum* is what let copper and
   crude_oil sit a day behind unnoticed — four current commodities made the
   class look fresh. Tune with `--allow-lagging` if some assets genuinely do
   not trade every session.

3. **Display vs model-input comparison.** `data-ready/` is built *from*
   `data-new/`, so it can never legitimately be older. When it is, feature
   engineering did not complete. This is the only check that catches a
   *uniform* lag — when all 97 PSX inputs sat at Aug 18 while display was at
   Aug 19, the per-asset check saw a healthy uniform class.

Absolute staleness still fails the run too, via `--max-age-days` (default 4,
tolerating a weekend plus a public holiday).

Steps are isolated: one class failing does not stop the others, and a failed
`collect` skips that class's `features` rather than re-emitting the previous
day. The process exit code is non-zero if anything failed, so Task Scheduler
shows a failed run instead of a silent no-op.

### Retries

Every upstream here drops connections under sustained sequential load, and the
first full run proved it. All four collectors now retry with backoff:

| Source | Observed without retry | Retry |
|---|---|---|
| TradingView (commodities) | gold + copper timed out in a 6-symbol pass | 4x on the whole websocket handshake |
| Binance (crypto) | — | 4x, already present |
| PSX DPS | **53 ok / 44 failed** across 97 tickers, nearly all ConnectTimeout | 4x per symbol |
| MUFAP | one ConnectTimeout on the fund-directory call aborted the entire run before a single fund was fetched | 4x on the directory; per-fund retry already present |

The MUFAP case is the one to note: `fetch_fund_history()` was already retrying,
but `discover_fund_ids()` was not — and that single call gates everything
downstream, so it was a one-request single point of failure for the whole job.

Logs: `logs/daily_update_YYYY-MM-DD.log`. Add `-v` to capture each step's stdout.

---

## Useful invocations

```bash
uv run python scripts/daily_update.py --only crypto,psx    # subset
uv run python scripts/daily_update.py --skip-collect       # rebuild features from existing data-new
uv run python scripts/daily_update.py --skip-features --skip-snapshot   # collect only
uv run python scripts/daily_update.py -v                   # full step output
```

---

## Chart news catalysts

Each price chart marks dated news as a vertical line with a hover box; clicking
the marker opens the article. Backend and frontend for this already existed —
`NewsCatalyst`, all three engines populating `catalysts`, and `ForecastChart`
drawing the `ReferenceLine` plus hover card. The feature was dark for one
reason: `fetch_latest()` began with `if not settings.enable_live_prices`.

That gate conflated two unrelated things. `enable_live_prices` is off because
yfinance quotes a *different instrument* per asset class than `data-new/` is
collected from, which would put a price on screen contradicting its own chart.
A headline is a headline whichever feed the candles came from. News now has its
own switch, `enable_news_catalysts` (default on).

### Sources, and why it takes two

| | TradingView | Google News RSS |
|---|---|---|
| Crypto, commodities | good | good |
| PSX | patchy — OGDC 55 headlines, LUCK 3, **HBL and ENGRO none** | good, keyed on company name |
| History | rolling recent window only — BTC spanned 4 days | reaches years back |

TradingView alone leaves most PSX tickers with no markers at all and bunches
crypto markers against the right edge of a 30D chart. Google News RSS is
keyless, dated, linked, and supports `after:`/`before:`, so `collect_news.py`
issues one plain query plus four quarterly windows per asset — which is what
produces markers spread across the chart instead of a cluster at today.

Every marker is clickable. TradingView sets `link` only for syndicated
partners — its own wire copy (Dow Jones, Reuters) carries none, which left
roughly half the markers dead — but `storyPath` is always present and resolves
to the full story on tradingview.com.

### Collected to disk, not fetched per request

```bash
uv run python scripts/collect_news.py            # all assets
uv run python scripts/collect_news.py BTC HBL    # a subset
```

Output: `data-new/news/<asset_class>/<ticker>.csv`. The windowed queries take
seconds per asset — fine on a schedule, far too slow inside a page request — so
the API reads the archive (cached on file mtime) and only tops it up with
TradingView's live call for today's headlines. Both sources are flaky enough to
need it: `"Systems Limited" PSX` returned nothing on one attempt and 73 items
on the next, so every request retries.

The daily job runs this as its own step (`--skip-news` to opt out). A failure
there costs chart markers, not prices or forecasts.

### Mutual funds: shared feeds, not per-fund ones

No source carries fund-level headlines. Verified, not assumed:

```
"ABL Income Fund"   0    "NBP Stock Fund"              0
"Al Meezan Mutual Fund"  0    "AL Habib Money Market Fund"  0
```

News about what *moves* a fund's NAV is abundant, so funds are collected as
shared feeds keyed to the MUFAP cluster instead of one query per fund:

| Cluster | Funds | Source |
|---|---|---|
| Income, MoneyMarket | 58 | rate pool — SBP policy, T-bill/PIB auctions, CPI, industry AUM |
| Equity | 11 | the PSX index feed already collected for the stock charts, reused |
| Balanced | 8 | both |

Plus one file per asset management company (`_amc_<slug>.csv`), resolved from
the fund-name prefix by `news_live.amc_for` — the single source of truth, which
`collect_news.py` imports so the table cannot drift. Al Ameen resolves to UBL
and KSE Meezan to Al Meezan, since those are brands rather than houses.

Nine requests cover all 77 funds; one per fund would be 462 and return nothing.

```bash
uv run python scripts/collect_news.py mufap
```

Every fund catalyst is flagged `market_wide` — none of it is about the
individual fund, and every fund in a cluster shows the same markers. Two
consequences worth knowing: AMC news runs about one item per year, so it gets a
reserved slice of the marker budget (`_MAX_AMC_DATES`) or it loses every even
sample against a ~500-item rate pool; and for the 58 income and money-market
funds the NAV is a near-straight accrual line (~0.10%/day against Equity's
0.74%), so a rate-decision marker explains a change in *slope*, not a spike.

### Choosing which markers to draw

A 30D BTC window has news on 22 of 30 days, and the chart caps markers at 10.
Taking the most recent would defeat the purpose, so the chart samples evenly
across the visible window. Within a day it leads with an asset-specific
headline over a market-wide one (`PSX dips 1,109 points`), since the hover box
shows the first item and that decides what the marker appears to be about.

The API trims before sending: 60 dates x 2 headlines, down from whole archives
(crude oil was shipping 851 rows). Two rules keep the trim honest — a headline
with a link outranks one without, since the legacy commodity archive carries no
URLs; and recent dates get the budget first, because the widest range button is
1Y and sampling evenly across a decade spent 35 of crude oil's 60 slots on
2014-2019 dates no range setting can reach.

---

## Keeping a running backend in sync

Engines memoise their CSV loads with `@lru_cache`, which is right for serving
but means a long-running process keeps returning the file it read first. The
collectors rewrite those files underneath it, so without intervention the API
goes on quoting a stale close while the data on disk is current — the files
update and the dashboard does not.

`daily_update.py` therefore POSTs `/api/data/reload` when it finishes. That
clears every cached frame (10 across the four engines), rebuilds each catalog,
and re-reads the snapshot. It is best-effort: if no backend is running the call
is logged and skipped, since the next start reads fresh files anyway. Disable
with `--skip-reload`, or point elsewhere with `--api-url`.

Verified end-to-end on a live fund: with a new NAV written to disk the API kept
serving the cached 10.4089, and after the reload returned the new value — no
restart involved.

Model artifacts are deliberately left cached; they change on a retrain, not a
data refresh.

---

## MUFAP: NAVs are automated, forecasts are not

**MUFAP is fully automated, forecasts included.** The 19:45 task collects into
`data-new/mufap-data/` for charts and quotes, and rebuilds `data-ready/mufap/`,
which is what the models now read.

That second half is new. Until the swap the engine served
`data/mufap_clustered/`, an export that stopped at **2026-08-07** and never
advanced — so every fund's forecast aged a day, every day, while the collectors
ran perfectly against a directory nothing read. `as_of` now tracks the other
three classes.

One freshness check stays MUFAP-specific:

- **`per_asset_max_age_days=10`** instead of the relative "behind the freshest
  asset" test. AMCs publish on their own schedules — on 2026-08-20, 70 of 82
  funds were at the latest date, 8 were two days back and three Alfalah VPS
  funds a week back, all still publishing normally. Their latest NAV *is* the
  most recent one available, so a relative test would flag a dozen healthy
  funds nightly while an absolute age still catches one that has gone silent.

### What the swap changed

| | Before | After |
|---|---|---|
| Frames | `data/mufap_clustered/` — 5 clusters, 95 files, frozen 2026-08-07 | `data-ready/mufap/` — 4 clusters, 82 files, current |
| Models | 64 artifacts, ragged coverage | 88 artifacts, complete 4x7x3 grid |
| Scaler width | 22 inputs | 21 inputs |
| Served funds | 77 | 81 |
| Held-out MAE | — | better in 27 of 28 cells, mean −12.3% |

The two halves each had to move together: new frames against models trained on
legacy frames is exactly the train/serve skew `backend/config.py` warns about.
Following the convention set by the earlier commodity, crypto and PSX swaps,
`models-new/mufap` and `results-new/mufap/ensemble_weights.json` were copied
into `models/mufap` and `results/mufap/`, with the originals preserved under
`_backup_v1/mufap/`.

Two consequences worth knowing:

- **The universe moved 77 → 81 funds.** 64 carried over, 13 dropped out, 18
  arrived — mostly VPS pension sub-funds gained, fixed-return plans lost. The
  old set also carried a `Commodity` cluster that no fund ever mapped to.
- **One fund is screened out.** *Meezan Pakistan ETF* has a feature frame but
  no NAV in either `data-new/mufap-data` or the raw export. A forecast is a
  return applied to a NAV, so it is excluded at catalog build rather than
  503-ing on first click — see `MUFAPEngine._drop_navless`.

---

## What this does *not* do

- **No retraining.** Models are frozen; only the data they read moves. Retrain
  deliberately via `training-scripts-new/*/02_train_all.py`.
- **No live/intraday prices.** Values move once a day, after each market
  closes. `enable_live_prices` stays off: with a daily cadence the finalized
  close *is* the value, and the collectors are already the same source as the
  chart.
