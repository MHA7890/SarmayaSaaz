"""
Commodities master training: data-ready/commodities/*.csv -> models-new/
commodities/ + results-new/commodities/stage2_{ml,dl}_metrics_{h}d.json.

Reproduces the production generators (src/generators/create_ml_nb_{h}d.py,
create_dl_nb_{h}d.py): per-asset (not clustered/pooled) training, 4 tree
models (XGBoost/LightGBM/CatBoost/RandomForest) + 5 DL models (LSTM/GRU/
Transformer/N-BEATS/TFT) per (commodity, horizon), same hyperparameters as
the "champion" config stage5_update_hyperparams.py patched in (max_depth=5,
reg_lambda=3.0, l2_leaf_reg=5.0, min_samples_leaf=15), SEQ_LENGTH=10,
HIDDEN_SIZE=32, targets clipped to [-0.8, 1.5].

Leakage: chronological 70/10/20 split, same as the original - but this
script additionally applies a purge/embargo gap of `horizon` days at each
split boundary, which the original notebooks did not. That's a deliberate
strengthening, not a fidelity gap: without a gap, a training row whose
target window is h days wide can span into the validation period, letting
the model see (through the target) information from dates it's meant to be
evaluated on blind. Per explicit instruction ("ensure no data leakage")
this script closes that gap instead of reproducing it. StandardScaler is
fit on the training fold only, per horizon.

Per explicit instruction every DL model trains up to 100 epochs with early
stopping (patience 22), not the original's 150/20.

Metrics match calculate_metrics() from the notebook generators exactly:
MAE/RMSE/MAPE/R2 on reconstructed price, Dir_Acc on sign-of-move vs. the
prior close, Naive_MAE/Improvement_Pct vs. a "no change" baseline.

Run:
    uv run python training-scripts-new/commodities/02_train_all.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "training-scripts-new"))
from common import architectures as arch  # noqa: E402
from common import dl_train  # noqa: E402
from common.metrics import price_metrics  # noqa: E402
from common.progress import StageProgress  # noqa: E402
from common.splits import chronological_split  # noqa: E402

DATA_DIR = ROOT / "data-ready" / "commodities"
MODELS_DIR = ROOT / "models-new" / "commodities"
RESULTS_DIR = ROOT / "results-new" / "commodities"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [7, 14, 28, 42, 60, 90, 120]
SEQ_LENGTH = 10
HIDDEN_SIZE = 32
MIN_ROWS = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TREE_MODELS = ["XGBoost", "LightGBM", "CatBoost", "RandomForest"]
DL_MODELS = ["LSTM", "GRU", "Transformer", "N-BEATS", "TFT"]
ARTIFACT_STEM = {
    "XGBoost": ("xgboost", "json"), "LightGBM": ("lightgbm", "txt"),
    "CatBoost": ("catboost", "cbm"), "RandomForest": ("randomforest", "pkl"),
    "LSTM": ("lstm", "pt"), "GRU": ("gru", "pt"), "Transformer": ("transformer", "pt"),
    "N-BEATS": ("nbeats", "pt"), "TFT": ("tft", "pt"),
}


def non_feature_columns(df: pd.DataFrame) -> set[str]:
    base = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    targets = {c for c in df.columns if c.startswith("Target_")}
    return base | targets


def make_sequences(X: np.ndarray, y: np.ndarray, current_price: np.ndarray, seq_len: int):
    if len(X) <= seq_len:
        empty = np.empty((0, seq_len, X.shape[1]))
        return empty, np.empty((0,)), np.empty((0,))
    xs = np.stack([X[i : i + seq_len] for i in range(len(X) - seq_len + 1)])
    ys = y[seq_len - 1 :]
    cp = current_price[seq_len - 1 :]
    return xs, ys, cp


def train_tree(model_name: str, X_train, y_train, X_val, y_val):
    common_kw = dict(random_state=42)
    if model_name == "XGBoost":
        model = XGBRegressor(n_estimators=200, learning_rate=0.03, max_depth=5, subsample=0.8,
                              colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=3.0,
                              early_stopping_rounds=20, n_jobs=-1, **common_kw)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif model_name == "LightGBM":
        model = LGBMRegressor(n_estimators=200, learning_rate=0.03, max_depth=5, subsample=0.8,
                               colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=3.0,
                               verbose=-1, n_jobs=-1, **common_kw)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])
    elif model_name == "CatBoost":
        model = CatBoostRegressor(iterations=200, learning_rate=0.03, depth=5, l2_leaf_reg=5.0,
                                   random_seed=42, verbose=0, early_stopping_rounds=20)
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
    else:  # RandomForest - bagged, no boosting rounds, no eval_set
        model = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=15,
                                       max_features="sqrt", n_jobs=-1, **common_kw)
        model.fit(X_train, y_train)
    return model


def build_dl_model(model_name: str, n_features: int) -> torch.nn.Module:
    if model_name == "LSTM":
        return arch.CommodityLSTM(n_features, HIDDEN_SIZE)
    if model_name == "GRU":
        return arch.CommodityGRU(n_features, HIDDEN_SIZE)
    if model_name == "Transformer":
        return arch.CommodityTransformer(n_features, HIDDEN_SIZE)
    if model_name == "N-BEATS":
        return arch.CommodityNBeats(n_features, SEQ_LENGTH, hidden=128)
    if model_name == "TFT":
        return arch.CommodityTFT(n_features, HIDDEN_SIZE)
    raise ValueError(model_name)


def run():
    names = sorted(p.stem for p in DATA_DIR.glob("*.csv"))
    print(f"Commodities training: {len(names)} assets from {DATA_DIR}")

    ml_metrics: dict[int, dict[str, dict]] = {h: {} for h in HORIZONS}
    dl_metrics: dict[int, dict[str, dict]] = {h: {} for h in HORIZONS}

    total_cells = len(names) * len(HORIZONS)
    progress = StageProgress("Commodities training (asset x horizon cells)", total_cells)

    for name in names:
        df = pd.read_csv(DATA_DIR / f"{name}.csv", index_col="Date", parse_dates=True).sort_index()
        exclude = non_feature_columns(df)
        feature_cols = [c for c in df.columns if c not in exclude]

        for h in HORIZONS:
            target_col = f"Target_Return_{h}d"
            if target_col not in df.columns:
                progress.step(f"{name}/{h}d: SKIPPED (no target column)")
                continue

            df_h = df.dropna(subset=[target_col] + feature_cols)
            if len(df_h) < MIN_ROWS:
                progress.step(f"{name}/{h}d: SKIPPED ({len(df_h)} rows < {MIN_ROWS})")
                continue

            # 70/10/20 chronological, purge-gapped at both boundaries. `val` is
            # used only for early stopping / best-epoch selection during
            # training; reported metrics come from the held-out `test` split
            # so early-stopping decisions can't leak into the numbers a model
            # is graded on.
            split = chronological_split(df_h, horizon_days=h, train_frac=0.7, val_frac=0.1, purge=True)
            train_df, val_df, test_df = split.train, split.val, split.test
            if len(train_df) < MIN_ROWS or len(val_df) < 20 or len(test_df) < 20:
                progress.step(
                    f"{name}/{h}d: SKIPPED (train={len(train_df)}, val={len(val_df)}, test={len(test_df)})"
                )
                continue

            X_train_raw = train_df[feature_cols].to_numpy(dtype=np.float64)
            y_train = train_df[target_col].to_numpy(dtype=np.float64)
            X_val_raw = val_df[feature_cols].to_numpy(dtype=np.float64)
            y_val = val_df[target_col].to_numpy(dtype=np.float64)
            X_test_raw = test_df[feature_cols].to_numpy(dtype=np.float64)
            y_test = test_df[target_col].to_numpy(dtype=np.float64)
            cp_test = test_df["Close"].to_numpy(dtype=np.float64)

            scaler = StandardScaler().fit(X_train_raw)
            X_train = scaler.transform(X_train_raw)
            X_val = scaler.transform(X_val_raw)
            X_test = scaler.transform(X_test_raw)
            joblib.dump(scaler, MODELS_DIR / f"{name}_scaler_{h}d.pkl")

            notes = []

            # --- tree models: tabular, last-row-per-sample (no sequence) ---
            for model_name in TREE_MODELS:
                try:
                    model = train_tree(model_name, X_train, y_train, X_val, y_val)
                    stem, ext = ARTIFACT_STEM[model_name]
                    path = MODELS_DIR / f"{name}_{stem}_{h}d.{ext}"
                    if model_name == "XGBoost":
                        model.save_model(str(path))
                    elif model_name == "LightGBM":
                        model.booster_.save_model(str(path))
                    elif model_name == "CatBoost":
                        model.save_model(str(path))
                    else:
                        joblib.dump(model, path)

                    preds = model.predict(X_test)
                    m = price_metrics(cp_test, y_test, preds)
                    ml_metrics[h].setdefault(name, {})[model_name] = m
                    notes.append(f"{model_name}=MAE{m['MAE']:.3f}/Dir{m['Dir_Acc']:.0f}%")
                except Exception as e:
                    notes.append(f"{model_name}=FAIL({e})")

            # --- DL models: sequences of SEQ_LENGTH rows ---
            X_train_seq, y_train_seq, _ = make_sequences(X_train, y_train, np.zeros(len(X_train)), SEQ_LENGTH)
            X_val_seq, y_val_seq, _ = make_sequences(X_val, y_val, np.zeros(len(X_val)), SEQ_LENGTH)
            X_test_seq, y_test_seq, cp_test_seq = make_sequences(X_test, y_test, cp_test, SEQ_LENGTH)
            n_features = len(feature_cols)

            if len(X_train_seq) >= 30 and len(X_val_seq) >= 10 and len(X_test_seq) >= 10:
                for model_name in DL_MODELS:
                    try:
                        model = build_dl_model(model_name, n_features)
                        desc = f"{name}/{h}d/{model_name}"
                        model, hist = dl_train.train_torch_model(
                            model, X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                            device=DEVICE, desc=desc, batch_size=32, lr=1e-3, weight_decay=1e-5,
                            loss_fn=torch.nn.MSELoss(),
                        )
                        stem, ext = ARTIFACT_STEM[model_name]
                        torch.save(model.state_dict(), MODELS_DIR / f"{name}_{stem}_{h}d.{ext}")

                        preds = dl_train.predict(model, X_test_seq, DEVICE)
                        m = price_metrics(cp_test_seq, y_test_seq, preds)
                        dl_metrics[h].setdefault(name, {})[model_name] = m
                        notes.append(f"{model_name}=MAE{m['MAE']:.3f}/Dir{m['Dir_Acc']:.0f}%({hist['epochs_run']}ep)")
                    except Exception as e:
                        notes.append(f"{model_name}=FAIL({e})")
            else:
                notes.append("DL=SKIPPED(too few sequence rows)")

            progress.step(f"{name}/{h}d: " + ", ".join(notes))

            with open(RESULTS_DIR / f"stage2_ml_metrics_{h}d.json", "w") as f:
                json.dump(ml_metrics[h], f, indent=2)
            with open(RESULTS_DIR / f"stage2_dl_metrics_{h}d.json", "w") as f:
                json.dump(dl_metrics[h], f, indent=2)

    progress.close()
    print(f"Artifacts: {MODELS_DIR}")
    print(f"Metrics: {RESULTS_DIR}")


if __name__ == "__main__":
    run()
