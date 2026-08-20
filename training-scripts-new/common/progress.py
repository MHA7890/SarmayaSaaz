"""
Shared progress reporting for the training-scripts-new/ pipelines.

Every training run here is long (hundreds of model fits across assets x
horizons x model families), so every script reports where it is, not just
whether it finished. Two levels:

  - `StageProgress` - coarse "N/total, elapsed, ETA" ticker for outer loops
    (asset x horizon x model cells).
  - `epoch_bar()` - a tqdm progress bar for one model's training epochs,
    annotated with train/val loss so a stuck or overfitting run is visible
    live, not just in a final log line.
"""
from __future__ import annotations

import time

from tqdm import tqdm


class StageProgress:
    """Console ticker for an outer loop with a known total item count."""

    def __init__(self, label: str, total: int):
        self.label = label
        self.total = total
        self.done = 0
        self.start = time.perf_counter()
        print(f"\n{'=' * 70}\n{label}: {total} items\n{'=' * 70}")

    def step(self, note: str = "") -> None:
        self.done += 1
        elapsed = time.perf_counter() - self.start
        rate = self.done / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - self.done) / rate if rate > 0 else 0.0
        pct = 100.0 * self.done / self.total
        bar_width = 30
        filled = int(bar_width * self.done / self.total)
        bar = "#" * filled + "-" * (bar_width - filled)
        msg = (
            f"[{bar}] {self.done}/{self.total} ({pct:5.1f}%) "
            f"elapsed={_fmt(elapsed)} eta={_fmt(remaining)}"
        )
        if note:
            msg += f"  | {note}"
        print(msg, flush=True)

    def close(self) -> None:
        elapsed = time.perf_counter() - self.start
        print(f"{self.label} complete: {self.done}/{self.total} in {_fmt(elapsed)}\n")


def _fmt(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def epoch_bar(total_epochs: int, desc: str):
    """A tqdm bar for one model's DL training loop. Call .set_postfix(...)
    each epoch and .update(1); the caller owns closing it."""
    return tqdm(
        total=total_epochs,
        desc=desc,
        unit="epoch",
        leave=False,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]",
    )
