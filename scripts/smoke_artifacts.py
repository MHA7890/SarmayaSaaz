"""
Artifact load smoke test.

Verifies every serialization format in models/ can be deserialized under the
pinned environment, and that sklearn pickles load without version drift.
Run: uv run python scripts/smoke_artifacts.py
"""
import glob
import os
import warnings

import joblib
import numpy as np

os.chdir(os.path.join(os.path.dirname(__file__), ".."))

results = []


def check(label, fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            detail = fn()
            warns = [w for w in caught if "version" in str(w.message).lower()]
            if warns:
                results.append(("WARN", label, str(warns[0].message)[:90]))
            else:
                results.append(("OK", label, detail))
        except Exception as e:
            results.append(("FAIL", label, f"{type(e).__name__}: {str(e)[:90]}"))


# --- sklearn pickles (the version-sensitive ones) ---------------------------
def _scaler(path):
    def inner():
        s = joblib.load(path)
        return f"n_features={getattr(s, 'n_features_in_', '?')}"
    return inner


for p in [
    "models/mufap/Equity_scaler.pkl",
    "models/stocks/Technology_scaler.pkl" if os.path.exists("models/stocks/Technology_scaler.pkl")
    else glob.glob("models/stocks/*_scaler.pkl")[0],
    glob.glob("models/crypto/*_scaler.pkl")[0] if glob.glob("models/crypto/*_scaler.pkl") else None,
    glob.glob("models/commodities/models_production/*_scaler_*.pkl")[0]
    if glob.glob("models/commodities/models_production/*_scaler_*.pkl") else None,
]:
    if p:
        check(f"sklearn scaler  {os.path.basename(p)}", _scaler(p))

rf = glob.glob("models/commodities/models_production/*_randomforest_60d.pkl")
if rf:
    check(f"sklearn RF      {os.path.basename(rf[0])}",
          lambda: f"n_features={joblib.load(rf[0]).n_features_in_}")


# --- XGBoost ---------------------------------------------------------------
def _xgb():
    from xgboost import XGBRegressor
    m = XGBRegressor()
    m.load_model("models/commodities/models_production/gold_xgboost_60d.json")
    return f"n_features={m.n_features_in_}"


check("xgboost json    gold_xgboost_60d", _xgb)


# --- LightGBM --------------------------------------------------------------
def _lgb():
    from lightgbm import Booster
    b = Booster(model_file="models/commodities/models_production/gold_lightgbm_60d.txt")
    return f"n_features={b.num_feature()}"


check("lightgbm txt    gold_lightgbm_60d", _lgb)


# --- CatBoost --------------------------------------------------------------
def _cat():
    from catboost import CatBoostRegressor
    m = CatBoostRegressor()
    m.load_model("models/commodities/models_production/gold_catboost_60d.cbm")
    return f"n_features={len(m.feature_names_)}"


check("catboost cbm    gold_catboost_60d", _cat)


# --- PyTorch ---------------------------------------------------------------
def _torch():
    import torch
    sd = torch.load("models/commodities/models_production/gold_lstm_60d.pt",
                    map_location="cpu", weights_only=True)
    key = next(k for k in sd if "weight_ih_l0" in k)
    return f"input_size={sd[key].shape[1]}, keys={len(sd)}"


check("torch state     gold_lstm_60d", _torch)


# --- Joblib-pickled tree models in crypto/mufap/stocks ---------------------
for pat, label in [
    ("models/crypto/Cluster_0_60d_XGBoost.pkl", "crypto  XGB pkl"),
    ("models/mufap/Equity_60d_*.pkl", "mufap   tree pkl"),
    ("models/stocks/Technology_60d_*.pkl", "stocks  tree pkl"),
]:
    hits = glob.glob(pat)
    if hits:
        check(f"{label}  {os.path.basename(hits[0])}",
              lambda h=hits[0]: type(joblib.load(h)).__name__)


# --- Report ----------------------------------------------------------------
print("\n" + "=" * 78)
print("  ARTIFACT LOAD SMOKE TEST")
print("=" * 78)
w = max(len(r[1]) for r in results)
for status, label, detail in results:
    icon = {"OK": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[status]
    print(f"{icon}  {label:<{w}}  {detail}")

n_fail = sum(1 for r in results if r[0] == "FAIL")
n_warn = sum(1 for r in results if r[0] == "WARN")
print("=" * 78)
print(f"  {len(results) - n_fail - n_warn} passed | {n_warn} warned | {n_fail} failed")
print("=" * 78)
raise SystemExit(1 if n_fail else 0)
