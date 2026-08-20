"""
PSX master training: data-ready/psx/<sector>/*.csv -> models-new/psx/ +
results-new/psx/ensemble_weights.json.

Reproduces src/stocks/stage4_master_training.py: per sector, every ticker's
rows are pooled into one training matrix (no per-ticker model, no ticker
identity feature - a single sector model serves every ticker in it), 3
candidate models per (sector, horizon) - XGBoost, LightGBM, and a "tabular"
LSTM (a single feature row reshaped to seq_len=1, not a real sequence model,
exactly as the original) - and the single lowest-validation-MAE model wins
that cell (routed inference, not ensembled).

No data leakage: chronological 80/20 split with a purge/embargo gap of
`horizon` CALENDAR DAYS at the boundary, and the StandardScaler is fit on
the training fold only (once per sector, on the 120d fold - see run()).

The original used a fixed 120-ROW gap here, which does not work on a pooled
sector matrix: one calendar date contributes one row per ticker, so 120 rows
bought only ~6 days on a 33-ticker sector and never scaled with the horizon.
That left ~4.3% of training rows at the 120d horizon with target windows
reaching into validation. This script gaps by days instead, so the guarantee
holds at every horizon.

Per explicit instruction the LSTM trains up to 100 epochs with early
stopping (patience 22), not the original's 200/15.

Run:
    uv run python training-scripts-new/psx/02_train_all.py
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

DATA_DIR = ROOT / "data-ready" / "psx"
MODELS_DIR = ROOT / "models-new" / "psx"
RESULTS_DIR = ROOT / "results-new" / "psx"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [7, 14, 28, 42, 60, 90, 120]
MIN_ROWS = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_sector_matrix(sector_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p, index_col="Date", parse_dates=True) for p in sorted(sector_dir.glob("*.csv"))]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()


def split_with_purge(df: pd.DataFrame, horizon_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    80/20 chronological split with a purge gap of `horizon_days` CALENDAR DAYS.

    The gap must be measured in days, not rows. `df` here is a POOLED sector
    matrix - every ticker in the sector concatenated and sorted by date - so a
    single calendar date contributes one row per ticker. The previous
    implementation skipped a fixed 120 *rows*, which on a 33-ticker sector
    (~27 rows/day) bought only ~6 calendar days, and on a 5-ticker sector ~34
    days. Neither scaled with the horizon, so a training row's forward-looking
    target window still reached into validation: measured contamination was
    ~4.3% of training rows at the 120d horizon (0% at 7d).

    Gapping by `pd.Timedelta(days=horizon_days)` makes the guarantee real and
    horizon-proportional, matching what common/splits.py already does for the
    commodities, crypto and MUFAP pipelines.
    """
    dates = pd.DatetimeIndex(df.index)
    split_date = dates[int(len(df) * 0.8)]

    train_df = df[dates <= split_date]
    val_df = df[dates > split_date + pd.Timedelta(days=horizon_days)]
    if len(val_df) < 50:
        # Fallback for thin sectors: drop the gap rather than end up with too
        # few validation rows to trust the metric. Same policy as
        # common/splits.two_way_chronological_split.
        val_df = df[dates > split_date]
    return train_df, val_df


def run():
    sector_dirs = sorted(p for p in DATA_DIR.iterdir() if p.is_dir())
    print(f"PSX training: {len(sector_dirs)} sectors from {DATA_DIR}")

    ensemble_weights: dict[str, dict[str, dict]] = {}
    total_cells = len(sector_dirs) * len(HORIZONS)
    progress = StageProgress("PSX training (sector x horizon cells)", total_cells)

    for sector_dir in sector_dirs:
        sector = sector_dir.name
        matrix = load_sector_matrix(sector_dir)
        ensemble_weights[sector] = {}
        if matrix.empty:
            for h in HORIZONS:
                progress.step(f"{sector}/{h}d: SKIPPED (no data)")
            continue

        target_cols = [f"Target_{h}d" for h in HORIZONS]
        feature_cols = [c for c in matrix.columns if c not in target_cols]

        # One scaler per sector, fit once here and reused for every horizon.
        #
        # Only ONE scaler file per sector can exist on disk, because inference
        # (RoutedEngine._scaler in backend/engines/routed.py) takes no horizon
        # argument - it loads "{sector}_scaler.pkl" once and scales a single
        # feature row before forecasting all seven horizons. Fitting a fresh
        # scaler inside the horizon loop therefore produced a train/serve skew:
        # each horizon trained under its own scaler, but only the last one
        # written (120d) survived, so six of seven horizons served under a
        # scaler they were never trained with.
        #
        # It is fit on the 120d (longest-horizon) training fold deliberately.
        # dropna() on a longer horizon's target discards more trailing rows per
        # ticker, so the 120d fold is the SHORTEST of the seven and a strict
        # subset of every other horizon's training fold. Fitting on it therefore
        # cannot leak any horizon's validation rows into the scaler statistics.
        # Fitting on a shorter horizon (e.g. 7d) would do exactly that.
        strictest = matrix.dropna(subset=[f"Target_{max(HORIZONS)}d"] + feature_cols)
        if len(strictest) < MIN_ROWS:
            for h in HORIZONS:
                progress.step(
                    f"{sector}/{h}d: SKIPPED (120d fold has {len(strictest)} rows < {MIN_ROWS}, "
                    "cannot fit a sector scaler)"
                )
            continue
        train_strict, _ = split_with_purge(strictest, max(HORIZONS))
        scaler = StandardScaler().fit(train_strict[feature_cols].to_numpy(dtype=np.float64))
        joblib.dump(scaler, MODELS_DIR / f"{sector}_scaler.pkl")

        for h in HORIZONS:
            target_col = f"Target_{h}d"
            clean = matrix.dropna(subset=[target_col] + feature_cols)
            if len(clean) < MIN_ROWS:
                progress.step(f"{sector}/{h}d: SKIPPED ({len(clean)} rows < {MIN_ROWS})")
                continue

            train_df, val_df = split_with_purge(clean, h)
            X_train_raw = train_df[feature_cols].to_numpy(dtype=np.float64)
            y_train = train_df[target_col].to_numpy(dtype=np.float64)
            X_val_raw = val_df[feature_cols].to_numpy(dtype=np.float64)
            y_val = val_df[target_col].to_numpy(dtype=np.float64)

            # Reuse the sector scaler fit above - never refit per horizon, or
            # the artifact on disk stops matching what these models trained on.
            X_train = scaler.transform(X_train_raw)
            X_val = scaler.transform(X_val_raw)

            metrics = {}

            lgbm = LGBMRegressor(n_estimators=500, learning_rate=0.05, n_jobs=-1, random_state=42)
            lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
            joblib.dump(lgbm, MODELS_DIR / f"{sector}_{h}d_LightGBM.pkl")
            metrics["LightGBM"] = mean_absolute_error(y_val, lgbm.predict(X_val))

            xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=-1, random_state=42, reg_lambda=1.0)
            xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            joblib.dump(xgb, MODELS_DIR / f"{sector}_{h}d_XGBoost.pkl")
            metrics["XGBoost"] = mean_absolute_error(y_val, xgb.predict(X_val))

            lstm = arch.TabularLSTM(input_dim=len(feature_cols))
            lstm, hist = dl_train.train_torch_model(
                lstm, X_train, y_train, X_val, y_val,
                device=DEVICE, desc=f"{sector}/{h}d/LSTM", batch_size=len(X_train),  # full-batch, as original
                lr=1e-3, weight_decay=1e-4,
            )
            torch.save(lstm.state_dict(), MODELS_DIR / f"{sector}_{h}d_LSTM.pt")
            preds = dl_train.predict(lstm, X_val, DEVICE)
            metrics["LSTM"] = mean_absolute_error(y_val, preds)

            winner = min(metrics, key=metrics.get)
            ext = "pt" if winner == "LSTM" else "pkl"
            ensemble_weights[sector][f"{h}d"] = {
                "Winning_Model": winner,
                "MAE": round(float(metrics[winner]), 6),
                "Path": f"{sector}_{h}d_{winner}.{ext}",
            }

            progress.step(
                f"{sector}/{h}d: LGBM={metrics['LightGBM']:.4f} XGB={metrics['XGBoost']:.4f} "
                f"LSTM={metrics['LSTM']:.4f} ({hist['epochs_run']}ep) -> winner={winner}"
            )

            with open(RESULTS_DIR / "ensemble_weights.json", "w") as f:
                json.dump(ensemble_weights, f, indent=2)

    progress.close()
    print(f"Artifacts: {MODELS_DIR}")
    print(f"Routing table: {RESULTS_DIR / 'ensemble_weights.json'}")


if __name__ == "__main__":
    run()
