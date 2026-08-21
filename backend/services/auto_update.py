"""
Automated Data Freshness Check & Procurement Service.

Checks dataset freshness on startup or on-demand. If data sitting on disk
(under data-new/ and data-ready/) is behind the expected latest closed
market day, it automatically invokes procurement (collectors, feature
engineering, and snapshot rebuild) before backend initialization completes.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backend.config import settings

logger = logging.getLogger("sarmayasaaz.auto_update")


def get_expected_latest_date() -> pd.Timestamp:
    """
    Determine the expected latest completed trading day.
    - Sat/Sun -> Previous Friday
    - Monday (before 20:00 UTC) -> Previous Friday
    - Tue-Fri (before 20:00 UTC) -> Yesterday
    - Weekday (after 20:00 UTC) -> Today
    """
    now_utc = datetime.now(timezone.utc)
    today = pd.Timestamp(now_utc.date())
    weekday = today.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun

    if weekday == 5:  # Saturday
        return today - pd.Timedelta(days=1)
    if weekday == 6:  # Sunday
        return today - pd.Timedelta(days=2)
    if weekday == 0 and now_utc.hour < 20:  # Monday before evening
        return today - pd.Timedelta(days=3)
    if now_utc.hour < 20:
        return today - pd.Timedelta(days=1)
    return today


def get_asset_class_newest_date(glob_pattern: str) -> pd.Timestamp | None:
    """Find the newest date across CSV files matching glob_pattern."""
    newest: pd.Timestamp | None = None
    for path in settings.project_root.glob(glob_pattern):
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True, usecols=[0])
            if len(df.index) > 0:
                top = df.index.max()
                if isinstance(top, pd.Timestamp):
                    if newest is None or top > newest:
                        newest = top
        except Exception:
            continue
    return newest


def check_data_freshness() -> dict[str, dict[str, str | bool | int]]:
    """
    Inspect freshness of all asset classes on disk.
    Returns dict mapping asset_class -> info dict with keys:
      'newest_date', 'expected_date', 'stale_days', 'is_stale'
    """
    expected = get_expected_latest_date()
    patterns = {
        "commodities": "data-new/commodities-data/*.csv",
        "crypto": "data-new/crypto-data/*.csv",
        "psx": "data-new/psx-data/*.csv",
        "mufap": "data-new/mufap-data/*.csv",
    }

    report: dict[str, dict[str, str | bool | int]] = {}
    for name, pattern in patterns.items():
        newest = get_asset_class_newest_date(pattern)
        if newest is None:
            stale_days = 999
            is_stale = True
            newest_str = "missing"
        else:
            stale_days = (expected - newest).days
            # MUFAP publishes NAVs with AMC lag; allow up to 3 days for MUFAP before calling stale
            max_allowed_lag = 3 if name == "mufap" else 0
            is_stale = stale_days > max_allowed_lag
            newest_str = str(newest.date())

        report[name] = {
            "newest_date": newest_str,
            "expected_date": str(expected.date()),
            "stale_days": stale_days,
            "is_stale": is_stale,
        }

    return report


def procure_fresh_data(stale_classes: list[str] | None = None) -> bool:
    """
    Run daily_update.py to fetch fresh data from sources and rebuild features.
    If stale_classes is specified, runs collect & features for those classes only.
    """
    script = settings.project_root / "scripts" / "daily_update.py"
    if not script.exists():
        logger.error("[Auto-Procure] Daily update script not found at %s", script)
        return False

    cmd = [sys.executable, str(script)]
    if stale_classes:
        cmd.extend(["--only", ",".join(stale_classes)])

    logger.info("[Auto-Procure] Launching data procurement command: %s", " ".join(cmd))
    started = time.monotonic()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(settings.project_root),
            timeout=1800,  # 30 min ceiling
        )
        elapsed = time.monotonic() - started
        if proc.returncode == 0:
            logger.info("[Auto-Procure] Successfully completed data procurement in %.1fs", elapsed)
            return True
        else:
            logger.warning("[Auto-Procure] Data procurement finished with non-zero exit code (%d) in %.1fs",
                           proc.returncode, elapsed)
            # Log output excerpt for debugging
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-10:]
            for line in tail:
                logger.warning("   | %s", line)
            return False
    except Exception as e:
        logger.error("[Auto-Procure] Failed to launch procurement: %s", e)
        return False


def ensure_fresh_data_on_startup() -> bool:
    """
    Startup hook for FastAPI main.py.
    Verifies freshness; if any asset class is stale and auto_procure_on_startup is enabled,
    triggers procurement before the backend loads engines into memory.
    """
    if not getattr(settings, "auto_procure_on_startup", True):
        logger.info("[Data Auto-Sync] Auto-procurement on startup is disabled in config.")
        return True

    logger.info("[Data Auto-Sync] Verifying dataset freshness against expected market closed date...")
    freshness = check_data_freshness()
    stale_classes = [ac for ac, info in freshness.items() if info["is_stale"]]

    for ac, info in freshness.items():
        status = "STALE" if info["is_stale"] else "OK"
        logger.info("[Data Auto-Sync] %-12s: disk=%s vs expected=%s -> %s",
                    ac, info["newest_date"], info["expected_date"], status)

    if not stale_classes:
        logger.info("[Data Auto-Sync] All market datasets are fresh. Proceeding with system start.")
        return True

    logger.warning("[Data Auto-Sync] Stale market data detected for: %s. Procuring fresh data from sources...",
                   ", ".join(stale_classes))
    success = procure_fresh_data(stale_classes)
    if success:
        logger.info("[Data Auto-Sync] Data procurement succeeded. Reloading fresh datasets into system...")
    else:
        logger.warning("[Data Auto-Sync] Data procurement encountered errors; continuing with available data.")

    return success
