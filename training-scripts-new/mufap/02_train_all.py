"""
MUFAP master training: data-ready/mufap/<cluster>/*.csv -> models-new/mufap/
+ results-new/mufap/ensemble_weights.json.

Reproduces src/mufap/stage4_master_training.py: per cluster, every fund's
rows are pooled into one training matrix, 3 candidate models per (cluster,
horizon) - XGBoost, LightGBM, and a "tabular" LSTM (seq_len=1, same shape as
the PSX one) - single lowest-validation-MAE model wins each cell.

No data leakage: one global 80/20 split date per cluster (by unique
calendar date, not row count), reused across all 7 horizons, with training
rows restricted to more than `horizon` days before the split date so no
training row's forward-looking target window crosses into validation - same
mechanism as the original. StandardScaler fit on the pre-split-date rows
only.

Per explicit instruction the LSTM trains up to 100 epochs with early
stopping (patience 22), not the original's 150/15.

Run:
    uv run python training-scripts-new/mufap/02_train_all.py
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
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "training-scripts-new"))
from common import architectures as arch  # noqa: E402
from common import dl_train  # noqa: E402
from common.progress import StageProgress  # noqa: E402

DATA_DIR = ROOT / "data-ready" / "mufap"
MODELS_DIR = ROOT / "models-new" / "mufap"
RESULTS_DIR = ROOT / "results-new" / "mufap"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [7, 14, 28, 42, 60, 90, 120]
MIN_ROWS = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cluster_matrix(cluster_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p, index_col="Date", parse_dates=True) for p in sorted(cluster_dir.glob("*.csv"))]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()


def run():
    cluster_dirs = sorted(p for p in DATA_DIR.iterdir() if p.is_dir())
    print(f"MUFAP training: {len(cluster_dirs)} clusters from {DATA_DIR}")

    ensemble_weights: dict[str, dict[str, dict]] = {}
    total_cells = len(cluster_dirs) * len(HORIZONS)
    progress = StageProgress("MUFAP training (cluster x horizon cells)", total_cells)

    for cluster_dir in cluster_dirs:
        cluster = cluster_dir.name
        matrix = load_cluster_matrix(cluster_dir)
        ensemble_weights[cluster] = {}
        if matrix.empty or len(matrix) < MIN_ROWS:
            for h in HORIZONS:
                progress.step(f"{cluster}/{h}d: SKIPPED (insufficient data)")
            continue

        target_cols = [f"Target_{h}d" for h in HORIZONS]
        feature_cols = [c for c in matrix.columns if c not in target_cols]

        unique_dates = np.sort(matrix.index.unique())
        split_idx = int(len(unique_dates) * 0.8)
        split_date = pd.Timestamp(unique_dates[split_idx])

        train_scaler_df = matrix[matrix.index < split_date]
        if len(train_scaler_df) < 100:
            for h in HORIZONS:
                progress.step(f"{cluster}/{h}d: SKIPPED (too little pre-split data)")
            continue
        scaler = StandardScaler().fit(train_scaler_df[feature_cols].to_numpy(dtype=np.float64))
        joblib.dump(scaler, MODELS_DIR / f"{cluster}_scaler.pkl")

        for h in HORIZONS:
            target_col = f"Target_{h}d"
            df_h = matrix.dropna(subset=[target_col])
            train_mask = df_h.index < (split_date - pd.Timedelta(days=h))
            val_mask = df_h.index >= split_date

            train_df, val_df = df_h[train_mask], df_h[val_mask]
            if len(train_df) < MIN_ROWS or len(val_df) < 50:
                progress.step(f"{cluster}/{h}d: SKIPPED (train={len(train_df)}, val={len(val_df)})")
                continue

            X_train = scaler.transform(train_df[feature_cols].to_numpy(dtype=np.float64))
            y_train = train_df[target_col].to_numpy(dtype=np.float64)
            X_val = scaler.transform(val_df[feature_cols].to_numpy(dtype=np.float64))
            y_val = val_df[target_col].to_numpy(dtype=np.float64)

            metrics = {}

            xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=42,
                                early_stopping_rounds=50, n_jobs=-1)
            xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            joblib.dump(xgb, MODELS_DIR / f"{cluster}_{h}d_XGBoost.pkl")
            metrics["XGBoost"] = mean_absolute_error(y_val, xgb.predict(X_val))

            lgbm = LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42, n_jobs=-1)
            lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
            joblib.dump(lgbm, MODELS_DIR / f"{cluster}_{h}d_LightGBM.pkl")
            metrics["LightGBM"] = mean_absolute_error(y_val, lgbm.predict(X_val))

            lstm = arch.TabularLSTM(input_dim=len(feature_cols))
            lstm, hist = dl_train.train_torch_model(
                lstm, X_train, y_train, X_val, y_val,
                device=DEVICE, desc=f"{cluster}/{h}d/LSTM", batch_size=1024,
                lr=1e-3, weight_decay=1e-4,
            )
            torch.save(lstm.state_dict(), MODELS_DIR / f"{cluster}_{h}d_LSTM.pt")
            preds = dl_train.predict(lstm, X_val, DEVICE)
            metrics["LSTM"] = mean_absolute_error(y_val, preds)

            winner = min(metrics, key=metrics.get)
            ext = "pt" if winner == "LSTM" else "pkl"
            ensemble_weights[cluster][f"{h}d"] = {
                "Winning_Model": winner,
                "MAE": round(float(metrics[winner]), 6),
                "Path": f"{cluster}_{h}d_{winner}.{ext}",
            }

            progress.step(
                f"{cluster}/{h}d: XGB={metrics['XGBoost']:.4f} LGBM={metrics['LightGBM']:.4f} "
                f"LSTM={metrics['LSTM']:.4f} ({hist['epochs_run']}ep) -> winner={winner}"
            )

            with open(RESULTS_DIR / "ensemble_weights.json", "w") as f:
                json.dump(ensemble_weights, f, indent=2)

    progress.close()
    print(f"Artifacts: {MODELS_DIR}")
    print(f"Routing table: {RESULTS_DIR / 'ensemble_weights.json'}")


if __name__ == "__main__":
    run()
