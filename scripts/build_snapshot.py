"""
Precompute forecasts for the entire universe.

Top Movers, the market table and the heatmap all read this snapshot rather than
running ~329 assets x 7 horizons per request. Run after retraining, after new
market data lands, or on a schedule.

    uv run python scripts/build_snapshot.py
"""
import logging
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("snapshot")

from backend.engines import engines  # noqa: E402
from backend.ml.registry import registry  # noqa: E402
from backend.services.snapshot import SNAPSHOT_PATH, snapshot  # noqa: E402


def main() -> int:
    log.info("Loading engines...")
    started = time.perf_counter()
    engines.load_all()

    online = len(engines.online)
    if online == 0:
        log.error("No engines loaded - nothing to snapshot")
        return 1
    log.info("%d/4 engines online, %d assets", online, engines.asset_count)

    log.info("Building snapshot (this takes a few minutes)...")
    meta = snapshot.build(persist=True, progress=True)
    elapsed = time.perf_counter() - started

    log.info("=" * 66)
    log.info("  Snapshot written to %s", SNAPSHOT_PATH)
    log.info("  Assets forecast : %d", meta["asset_count"])
    log.info("  Failed          : %d", meta["failed"])
    log.info("  Elapsed         : %.1fs", elapsed)
    log.info("  Model cache     : %s", registry.stats())
    log.info("=" * 66)

    if meta["failed"]:
        log.warning("Some assets could not be forecast; see snapshot 'failures'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
