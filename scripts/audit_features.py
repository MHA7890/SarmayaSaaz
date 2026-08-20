"""
Feature-contract audit.

For every commodity x horizon, report the input width each production artifact
expects, plus the scaler width. Any row where the scaler width disagrees with a
model width is an unservable combination: the scaler cannot produce a vector
that model can consume.

Run: uv run python scripts/audit_features.py
"""
import os
import warnings

import joblib

warnings.filterwarnings("ignore")
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

MD = "models/commodities/models_production"
COMMODITIES = ["gold", "silver", "copper", "crude_oil", "natural_gas", "wheat"]
HORIZONS = [1, 7, 14, 28, 42, 60, 90, 120]


def width(commodity, horizon, kind):
    try:
        if kind == "scaler":
            for n in (f"{commodity}_scaler_{horizon}d.pkl", f"{commodity}_scaler.pkl"):
                p = os.path.join(MD, n)
                if os.path.exists(p):
                    return joblib.load(p).n_features_in_
            return None
        if kind == "xgboost":
            from xgboost import XGBRegressor
            m = XGBRegressor()
            m.load_model(os.path.join(MD, f"{commodity}_xgboost_{horizon}d.json"))
            return m.n_features_in_
        if kind == "lightgbm":
            from lightgbm import Booster
            return Booster(model_file=os.path.join(
                MD, f"{commodity}_lightgbm_{horizon}d.txt")).num_feature()
        if kind == "catboost":
            from catboost import CatBoostRegressor
            m = CatBoostRegressor()
            m.load_model(os.path.join(MD, f"{commodity}_catboost_{horizon}d.cbm"))
            return len(m.feature_names_)
        if kind == "randomforest":
            return joblib.load(
                os.path.join(MD, f"{commodity}_randomforest_{horizon}d.pkl")).n_features_in_
        # torch
        import torch
        sd = torch.load(os.path.join(MD, f"{commodity}_{kind}_{horizon}d.pt"),
                        map_location="cpu", weights_only=True)
        for k in sd:
            if "weight_ih_l0" in k or k == "input_proj.weight":
                return sd[k].shape[1]
        for k in sd:
            if "fc1.weight" in k or "block1.fc1.weight" in k:
                return sd[k].shape[1] // 10
        return None
    except Exception:
        return None


KINDS = ["xgboost", "lightgbm", "catboost", "randomforest",
         "lstm", "gru", "transformer", "nbeats", "tft"]

print("\n" + "=" * 100)
print("  COMMODITY FEATURE-CONTRACT AUDIT   (scaler width vs model input width)")
print("=" * 100)
hdr = f"{'commodity':<12}{'hz':>4}  {'scaler':>6} |" + "".join(f"{k[:6]:>8}" for k in KINDS)
print(hdr)
print("-" * 100)

broken = 0
total = 0
mismatch_rows = []

for c in COMMODITIES:
    for h in HORIZONS:
        sc = width(c, h, "scaler")
        widths = {k: width(c, h, k) for k in KINDS}
        present = {k: v for k, v in widths.items() if v is not None}
        if not present:
            continue
        total += 1
        bad = [k for k, v in present.items() if sc is not None and v != sc]
        if bad:
            broken += 1
            mismatch_rows.append((c, h, sc, {k: present[k] for k in bad}))
        row = f"{c:<12}{h:>4}  {str(sc):>6} |"
        for k in KINDS:
            v = present.get(k)
            cell = "-" if v is None else (f"{v}" if v == sc else f"{v}*")
            row += f"{cell:>8}"
        print(row)

print("-" * 100)
print(f"  * = model width disagrees with its scaler")
print(f"  {broken} of {total} commodity/horizon combinations contain at least one mismatch")
print("=" * 100)

if mismatch_rows:
    print("\n  Distinct mismatch shapes:")
    seen = {}
    for c, h, sc, bad in mismatch_rows:
        for k, v in bad.items():
            seen.setdefault((sc, v), []).append(f"{c}/{h}d/{k}")
    for (sc, v), examples in sorted(seen.items()):
        print(f"    scaler={sc} -> model={v}   ({len(examples)} artifacts)  e.g. {examples[0]}")
    print()
