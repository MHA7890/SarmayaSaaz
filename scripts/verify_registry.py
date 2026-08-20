"""
Verifies the registry loads every architecture and correctly handles reverted
champions whose input width is narrower than their scaler.

The old inference path hardcoded a truncation to 27 features, which is gold's
and natural_gas's width. Copper (25), silver (26) and wheat (28) fell through
and raised on shape. This proves all four now predict.

Run: uv run python scripts/verify_registry.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from backend.config import settings  # noqa: E402
from backend.ml.registry import ArtifactError, registry  # noqa: E402

MD = settings.commodities_models
SEQ = settings.commodity_seq_length

SUFFIX = {
    "XGBoost": ("xgboost", "json"), "LightGBM": ("lightgbm", "txt"),
    "CatBoost": ("catboost", "cbm"), "RandomForest": ("randomforest", "pkl"),
    "LSTM": ("lstm", "pt"), "GRU": ("gru", "pt"), "Transformer": ("transformer", "pt"),
    "N-BEATS": ("nbeats", "pt"), "TFT": ("tft", "pt"),
}

rows = []
for commodity, scaler_width in [
    ("gold", 36), ("silver", 35), ("copper", 34), ("crude_oil", 35),
    ("natural_gas", 36), ("wheat", 37),
]:
    # A scaled feature matrix of the full width the scaler emits.
    rng = np.random.default_rng(0)
    X_flat = rng.normal(size=(1, scaler_width))
    X_seq = rng.normal(size=(1, SEQ, scaler_width))

    for horizon in (7, 60, 120):
        for model_name, (stem, ext) in SUFFIX.items():
            path = MD / f"{commodity}_{stem}_{horizon}d.{ext}"
            if not path.exists():
                continue
            try:
                h = registry.get(path, model_name=model_name, family="commodity")
                X = X_flat if h.kind == "tree" else X_seq
                value = h.predict(X)
                truncated = h.input_width < scaler_width
                ok = np.isfinite(value)
                rows.append((commodity, horizon, model_name, h.input_width,
                             truncated, "ok" if ok else "non-finite", value))
            except ArtifactError as e:
                rows.append((commodity, horizon, model_name, -1, False,
                             f"FAIL {e}"[:48], float("nan")))

print("\n" + "=" * 92)
print("  REGISTRY VERIFICATION - reverted champions must predict, not vanish")
print("=" * 92)

n_ok = sum(1 for r in rows if r[5] == "ok")
n_trunc = sum(1 for r in rows if r[4] and r[5] == "ok")
n_fail = sum(1 for r in rows if r[5].startswith("FAIL"))

for commodity in ["gold", "silver", "copper", "crude_oil", "natural_gas", "wheat"]:
    sub = [r for r in rows if r[0] == commodity]
    if not sub:
        continue
    trunc = [r for r in sub if r[4]]
    fails = [r for r in sub if r[5].startswith("FAIL")]
    widths = sorted({r[3] for r in sub if r[3] > 0})
    print(f"  {commodity:<12} {len(sub):>3} artifacts | widths {widths} | "
          f"{len(trunc):>2} truncated | {len(fails)} failed")

print("-" * 92)
print(f"  {n_ok}/{len(rows)} predicted successfully")
print(f"  {n_trunc} of those required truncation (reverted pre-sentiment champions)")
print(f"  {n_fail} failed")

if n_fail:
    print("\n  Failures:")
    for r in rows:
        if r[5].startswith("FAIL"):
            print(f"    {r[0]}/{r[1]}d/{r[2]}: {r[5]}")

print("=" * 92)
print(f"  cache: {registry.stats()}")
print("=" * 92)
raise SystemExit(1 if n_fail else 0)
