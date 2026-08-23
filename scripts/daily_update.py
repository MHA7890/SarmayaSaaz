"""
Daily data refresh: collect -> engineer features -> rebuild the snapshot.

Run this once a day (see docs/automation.md for the Windows Task Scheduler
registration). Every step is idempotent: collectors perform incremental updates
(trimming recent buffer rows and merging newly fetched bars into existing CSVs),
ensuring fast daily updates and robust self-healing.


Two layers have to move together for the dashboard to actually advance:

    data-new/     the displayed close/NAV and the chart
    data-ready/   the frames the forecasts are computed from

Refreshing only the first updates the quoted price while every forecast keeps
running off a frozen feature frame, which looks fine and is wrong. So each
asset class runs `collect` then `features`, and the snapshot rebuild goes last.

Steps are isolated: one class failing (a source down, a network blip) does not
prevent the others from updating. The exit code is non-zero if any step failed,
so Task Scheduler surfaces it as a failed run rather than a silent no-op.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"

# Per-step wall-clock ceiling. The collectors are network-bound and paced with
# deliberate sleeps between symbols (PSX has ~97 tickers, MUFAP ~200 funds), so
# these are generous on purpose - a step that overruns is a hung socket, not a
# slow day.
DEFAULT_TIMEOUT_S = 3600
# News collection is far slower than any collector - see the news:collect step.
NEWS_TIMEOUT_S = 4 * 3600


STATUS_FILE = ROOT / "data-ready" / ".sync_status.json"


def notify_sync_status(api_url: str | None, is_syncing: bool, current_step: str, progress: int) -> None:
    """Update disk status file directly AND notify running API endpoint."""
    status_data = {
        "is_syncing": is_syncing,
        "current_step": current_step,
        "step": current_step,
        "progress": progress,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATUS_FILE.write_text(json.dumps(status_data, indent=2), encoding="utf-8")
    except Exception:
        pass

    if not api_url:
        api_url = "http://127.0.0.1:8000"
    try:
        import requests
        endpoint = f"{api_url.rstrip('/')}/api/system/sync-status"
        requests.post(
            endpoint,
            json=status_data,
            timeout=3,
        )
    except Exception:
        pass




@dataclass
class Step:
    name: str
    script: Path
    # A step whose failure should not by itself fail the run. Nothing is
    # optional today; kept so a flaky non-essential source can be demoted
    # without restructuring the runner.
    optional: bool = False


@dataclass
class AssetClass:
    name: str
    collect: Path
    features: Path
    # (label, glob) used only for the freshness report at the end.
    display_glob: str = ""
    model_glob: str = ""
    note: str = ""
    # Whether the model-input layer is expected to track the display layer.
    # False for MUFAP, whose engine still serves frozen legacy frames on
    # purpose - checking it would raise the same alarm every single night.
    model_checks: bool = True
    # When set, each asset is judged by its own absolute age rather than by
    # how far it sits behind the freshest asset in the class. Exchange-traded
    # classes all trade the same sessions, so "behind the leader" is a real
    # defect there. MUFAP is different: AMCs publish on their own schedules
    # and the VPS/pension funds run several days behind the money-market ones
    # while still publishing daily. Their latest NAV *is* the most recent one
    # available, so relative lag would flag a dozen healthy funds nightly.
    per_asset_max_age_days: int | None = None


CLASSES = [
    AssetClass(
        name="commodities",
        collect=ROOT / "scripts" / "collect_commodities_tv.py",
        features=ROOT / "training-scripts-new" / "commodities" / "01_feature_engineering.py",
        display_glob="data-new/commodities-data/*.csv",
        model_glob="data-ready/commodities/*.csv",
    ),
    AssetClass(
        name="crypto",
        collect=ROOT / "scripts" / "collect_crypto_binance.py",
        features=ROOT / "training-scripts-new" / "crypto" / "01_feature_engineering.py",
        display_glob="data-new/crypto-data/*.csv",
        model_glob="data-ready/crypto/*.csv",
    ),
    AssetClass(
        name="psx",
        collect=ROOT / "scripts" / "collect_psx_stocks.py",
        features=ROOT / "training-scripts-new" / "psx" / "01_feature_engineering.py",
        display_glob="data-new/psx-data/*.csv",
        model_glob="data-ready/psx/**/*.csv",
    ),
    AssetClass(
        name="mufap",
        collect=ROOT / "scripts" / "collect_mufap_funds.py",
        features=ROOT / "training-scripts-new" / "mufap" / "01_feature_engineering.py",
        display_glob="data-new/mufap-data/*.csv",
        model_glob="data-ready/mufap/**/*.csv",
        # Observed spread on 2026-08-20: 97 funds 1d behind, 8 at 3d, and three
        # Alfalah VPS funds at 7d that stopped publishing on 08-13. 10d clears
        # normal AMC lag while still catching a fund that has genuinely gone
        # silent or a collector that has stopped writing.
        per_asset_max_age_days=10,
    ),
]

logger = logging.getLogger("daily_update")


def setup_logging(verbose: bool) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"daily_update_{datetime.now():%Y-%m-%d}.log"
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return log_path


def run_step(step: Step, timeout: int) -> tuple[bool, float, str]:
    """Run one script in its own interpreter. Returns (ok, seconds, detail)."""
    if not step.script.exists():
        return False, 0.0, f"script not found: {step.script}"

    started = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, str(step.script)],
            capture_output=True,
            text=True,
            timeout=timeout,
            # cwd matters only for the scripts/ collectors, which import
            # data_new_common as a top-level module. The training scripts
            # resolve everything from their own __file__ and manage sys.path
            # themselves, so this is safe for both.
            cwd=str(step.script.parent),
        )
    except subprocess.TimeoutExpired:
        return False, time.monotonic() - started, f"timed out after {timeout}s"
    except Exception as e:  # noqa: BLE001 - runner must survive any launch failure
        return False, time.monotonic() - started, f"{type(e).__name__}: {e}"

    elapsed = time.monotonic() - started
    for line in (proc.stdout or "").splitlines():
        logger.debug("    | %s", line.rstrip())

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = tail[-1] if tail else f"exit code {proc.returncode}"
        for line in tail[-15:]:
            logger.error("    ! %s", line.rstrip())
        return False, elapsed, detail

    return True, elapsed, ""


@dataclass
class Freshness:
    """Per-asset freshness for one directory of CSVs."""
    total: int = 0
    newest: pd.Timestamp | None = None
    # asset name -> its own newest date, for everything behind `newest`
    lagging: dict[str, pd.Timestamp] = field(default_factory=dict)
    # asset name -> its own newest date, for every asset
    per_asset: dict[str, pd.Timestamp] = field(default_factory=dict)

    @property
    def current(self) -> int:
        return self.total - len(self.lagging)

    @property
    def newest_str(self) -> str:
        return str(self.newest.date()) if self.newest is not None else "-"


def scan_freshness(pattern: str) -> Freshness:
    """
    Newest date **per asset**, not just the maximum across the class.

    Checking only the class maximum hides partial failures, which is exactly
    how copper and crude_oil sat a day behind for hours: the other four
    commodities were current, so the class looked fresh. Their features had
    been skipped because openpyxl was missing for the two .xlsx macro inputs.
    """
    per_asset: dict[str, pd.Timestamp] = {}
    for path in ROOT.glob(pattern):
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True, usecols=[0])
        except Exception:
            continue
        # NB: reading usecols=[0] as the index yields a frame with zero
        # columns, and pandas calls that .empty even when the index is full -
        # so test the index length, not .empty.
        if len(df.index) == 0:
            continue
        try:
            top = df.index.max()
        except Exception:
            continue
        if isinstance(top, pd.Timestamp):
            per_asset[path.stem] = top

    out = Freshness(total=len(per_asset), per_asset=per_asset)
    if not per_asset:
        return out
    out.newest = max(per_asset.values())
    out.lagging = {name: d for name, d in per_asset.items() if d < out.newest}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated asset classes (default: all)")
    ap.add_argument("--skip", help="comma-separated asset classes to skip")
    ap.add_argument("--skip-collect", action="store_true", help="reuse data-new as-is, only rebuild features")
    ap.add_argument("--skip-features", action="store_true", help="refresh data-new only, leave data-ready alone")
    ap.add_argument("--skip-snapshot", action="store_true", help="do not rebuild results/snapshot.json")
    ap.add_argument("--skip-news", action="store_true", help="do not refresh chart news catalysts")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S, help=f"per-step timeout (default {DEFAULT_TIMEOUT_S}s)")
    ap.add_argument(
        "--max-age-days", type=int, default=4,
        help="fail the run if a class's newest date is older than this (default 4, "
             "which tolerates a weekend plus a public holiday)",
    )
    ap.add_argument(
        "--allow-lagging", type=int, default=0,
        help="how many assets may sit behind their class's newest date before the "
             "run fails (default 0). Raise it only if some assets legitimately do "
             "not trade every session - an illiquid PSX ticker, say - after "
             "confirming from the listed names that it is not a partial failure",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="log each step's stdout")
    ap.add_argument("--api-url", default="http://127.0.0.1:8000",
                    help="running backend to notify after the refresh (default %(default)s)")
    ap.add_argument("--skip-reload", action="store_true",
                    help="do not tell a running backend to re-read the new data")
    args = ap.parse_args()

    selected = [c.name for c in CLASSES]
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        unknown = want - set(selected)
        if unknown:
            print(f"unknown asset class(es): {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        selected = [n for n in selected if n in want]
    if args.skip:
        drop = {s.strip() for s in args.skip.split(",") if s.strip()}
        selected = [n for n in selected if n not in drop]

    log_path = setup_logging(args.verbose)
    started = datetime.now()
    logger.info("=" * 78)
    logger.info("Daily update starting - classes: %s", ", ".join(selected) or "(none)")
    logger.info("Log file: %s", log_path)
    logger.info("=" * 78)

    notify_sync_status(args.api_url, True, "Initializing data procurement...", 10)

    results: list[tuple[str, bool, float, str]] = []
    total_classes = len(selected)

    for idx, cls in enumerate([c for c in CLASSES if c.name in selected]):
        step_progress = 10 + int((idx / max(1, total_classes)) * 70)
        notify_sync_status(args.api_url, True, f"Updating {cls.name} daily bars & features...", step_progress)

        steps: list[Step] = []
        if not args.skip_collect:
            steps.append(Step(f"{cls.name}:collect", cls.collect))
        if not args.skip_features:
            steps.append(Step(f"{cls.name}:features", cls.features))

        for step in steps:
            logger.info("-> %s", step.name)
            ok, secs, detail = run_step(step, args.timeout)
            results.append((step.name, ok, secs, detail))
            if ok:
                logger.info("   done in %.1fs", secs)
            else:
                logger.error("   FAILED after %.1fs: %s", secs, detail)
                # Deliberately NOT skipping features here. A collect that loses
                # a few symbols out of many is the common case - PSX came back
                # 94/97 - and refusing to build features then leaves *every*
                # asset in the class frozen to punish three. Feature
                # engineering is idempotent and reads whatever is on disk, so
                # running it publishes the 94 that did update. The collect
                # failure still stands in `results`, so the run exits non-zero,
                # and the per-asset freshness check below names exactly which
                # assets are behind.

    if not args.skip_news:
        # Chart catalyst markers. Independent of the price pipeline: a failure
        # here costs markers on the chart, not prices or forecasts.
        logger.info("-> news:collect")
        # Its own ceiling: this walks 123 assets x ~6 windowed queries with
        # retries, which ran ~75s per asset against a flaky Google - well past
        # the per-step default. Timing it out mid-way would leave a partial
        # archive and no markers for the assets it never reached.
        ok, secs, detail = run_step(
            Step("news:collect", ROOT / "scripts" / "collect_news.py"),
            max(args.timeout, NEWS_TIMEOUT_S),
        )
        results.append(("news:collect", ok, secs, detail))
        logger.info("   done in %.1fs", secs) if ok else logger.error("   FAILED: %s", detail)

    if not args.skip_snapshot:
        notify_sync_status(args.api_url, True, "Rebuilding universe snapshot & predictions...", 90)
        logger.info("-> snapshot:rebuild")
        ok, secs, detail = run_step(Step("snapshot:rebuild", ROOT / "scripts" / "build_snapshot.py"), args.timeout)
        results.append(("snapshot:rebuild", ok, secs, detail))
        logger.info("   done in %.1fs", secs) if ok else logger.error("   FAILED: %s", detail)

    # -- tell a running API to re-read what we just wrote ----------------
    if not args.skip_reload:
        try:
            import urllib.request

            req = urllib.request.Request(f"{args.api_url}/api/data/reload", method="POST")
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.load(resp)
            logger.info(
                "-> notified API at %s (caches cleared: %s)",
                args.api_url, payload.get("caches_cleared"),
            )
        except Exception as e:  # noqa: BLE001 - a down backend is not a failure
            logger.info("-> no running API to notify at %s (%s)", args.api_url, type(e).__name__)

    # Mark sync as completed & unlock site
    notify_sync_status(args.api_url, False, "Completed", 100)

    # -- report ---------------------------------------------------------
    failed = [r for r in results if not r[1]]
    logger.info("=" * 78)
    logger.info("Freshness after this run (assets current / total @ newest date):")
    logger.info("  %-13s %-26s %-26s", "class", "display", "model input")

    # A step can exit 0 and still leave data stale - a source serves a cached
    # page, a symbol fails inside a loop that swallows it, an optional import
    # is missing. So what actually landed on disk is checked independently,
    # per asset, and any shortfall fails the run on its own.
    today = pd.Timestamp.now().normalize()
    for cls in [c for c in CLASSES if c.name in selected]:
        disp = scan_freshness(cls.display_glob) if cls.display_glob else Freshness()
        model = scan_freshness(cls.model_glob) if cls.model_glob else Freshness()

        logger.info(
            "  %-13s %-26s %-26s%s",
            cls.name,
            f"{disp.current}/{disp.total} @ {disp.newest_str}",
            f"{model.current}/{model.total} @ {model.newest_str}",
            f"   ({cls.note})" if cls.note else "",
        )

    try:
        notify_sync_status(args.api_url, True, "Initializing data procurement...", 10)

        for idx, cls in enumerate([c for c in CLASSES if c.name in selected]):
            step_progress = 10 + int((idx / max(1, total_classes)) * 70)
            notify_sync_status(args.api_url, True, f"Updating {cls.name} daily bars & features...", step_progress)

            steps: list[Step] = []
            if not args.skip_collect:
                steps.append(Step(f"{cls.name}:collect", cls.collect))
            if not args.skip_features:
                steps.append(Step(f"{cls.name}:features", cls.features))

            for step in steps:
                logger.info("-> %s", step.name)
                ok, secs, detail = run_step(step, args.timeout)
                results.append((step.name, ok, secs, detail))
                if ok:
                    logger.info("   done in %.1fs", secs)
                else:
                    logger.error("   FAILED after %.1fs: %s", secs, detail)

        if not args.skip_news:
            notify_sync_status(args.api_url, True, "Refreshing market news & sentiment...", 85)
            logger.info("-> news:collect")
            ok, secs, detail = run_step(
                Step("news:collect", ROOT / "scripts" / "collect_news.py", optional=True),
                NEWS_TIMEOUT_S,
            )
            results.append(("news:collect", ok, secs, detail))

        if not args.skip_snapshot:
            notify_sync_status(args.api_url, True, "Rebuilding whole-universe forecast snapshot...", 90)
            logger.info("-> snapshot:rebuild")
            started = time.monotonic()
            try:
                from backend.engines import engines
                from backend.services.snapshot import snapshot
                engines.load_all()
                snapshot.build(progress=False)
                secs = time.monotonic() - started
                logger.info("   done in %.1fs", secs)
                results.append(("snapshot:rebuild", True, secs, ""))
            except Exception as e:
                secs = time.monotonic() - started
                logger.error("   FAILED after %.1fs: %s", secs, e)
                results.append(("snapshot:rebuild", False, secs, str(e)))


        # Freshness verification pass
        logger.info("=" * 78)
        logger.info("Verifying post-update freshness against %s", today.date())
        for cls in [c for c in CLASSES if c.name in selected]:
            disp = audit_freshness(cls.name, "display", cls.display_glob)
            model = audit_freshness(cls.name, "model inputs", cls.model_glob)

            for layer, f in (("display", disp), ("model inputs", model)):
                if f.newest is None:
                    logger.error("     %s: no CSV files found under %s", layer, f.glob)
                    results.append((f"{cls.name}:{layer}", False, 0.0, "no data files"))
                    continue

                age = (today - f.newest).days
                if age > args.max_age_days:
                    results.append((f"{cls.name}:{layer}", False, 0.0, f"newest is {age}d old ({f.newest_str})"))

                if cls.per_asset_max_age_days is not None:
                    stale = {n: d for n, d in f.per_asset.items()
                             if (today - d).days > cls.per_asset_max_age_days}
                    if stale:
                        shown = sorted(stale.items(), key=lambda kv: kv[1])[:6]
                        detail = ", ".join(f"{n} @ {d.date()}" for n, d in shown)
                        if len(stale) > len(shown):
                            detail += f", +{len(stale) - len(shown)} more"
                        logger.error("     %s: %d asset(s) older than %dd -> %s",
                                     layer, len(stale), cls.per_asset_max_age_days, detail)
                        results.append((f"{cls.name}:{layer}", False, 0.0,
                                        f"{len(stale)} asset(s) older than {cls.per_asset_max_age_days}d"))
                    continue

                if len(f.lagging) > args.allow_lagging:
                    shown = sorted(f.lagging.items(), key=lambda kv: kv[1])[:6]
                    detail = ", ".join(f"{n} @ {d.date()}" for n, d in shown)
                    if len(f.lagging) > len(shown):
                        detail += f", +{len(f.lagging) - len(shown)} more"
                    logger.error("     %s: %d asset(s) behind %s -> %s", layer, len(f.lagging), f.newest_str, detail)
                    results.append((f"{cls.name}:{layer}", False, 0.0, f"{len(f.lagging)} asset(s) behind {f.newest_str}"))

            if (cls.model_checks and disp.newest is not None and model.newest is not None
                    and model.newest < disp.newest):
                behind = (disp.newest - model.newest).days
                logger.error(
                    "     model inputs are %dd behind display (%s vs %s) - features did not complete",
                    behind, model.newest_str, disp.newest_str,
                )
                results.append((
                    f"{cls.name}:features-lag", False, 0.0,
                    f"model inputs at {model.newest_str} vs display {disp.newest_str}",
                ))
    finally:
        notify_sync_status(args.api_url, False, "Complete", 100)
        if not args.skip_reload:
            try:
                import requests
                requests.post(f"{args.api_url.rstrip('/')}/api/data/reload", timeout=5)
            except Exception:
                pass

    failed = [r for r in results if not r[1]]

    logger.info("-" * 78)
    took = (datetime.now() - started).total_seconds()
    if failed:
        logger.error("Daily update finished in %.0fs with %d FAILED step(s):", took, len(failed))
        for name, _, _, detail in failed:
            logger.error("   %-24s %s", name, detail)
    else:
        logger.info("Daily update finished in %.0fs - all %d step(s) OK", took, len(results))
    logger.info("=" * 78)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
