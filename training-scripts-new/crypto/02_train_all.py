"""
Crypto master training: data-ready/crypto/*.csv -> models-new/crypto/ +
results-new/crypto/ensemble_weights.json + cluster_map.json.

Reproduces src/crypto/stage4_master_training.py's structure - 4 K-Means
clusters (on [BTC_Volatility_30d.mean(), Target_7d.mean()] per asset,
random_state=42, exactly as the original), 9 model families per
(cluster, horizon): XGBoost/LightGBM/CatBoost/RandomForest (Optuna, 3
trials) + LSTM/GRU/Transformer/NBEATS_Lite/TFT_Lite (PyTorch, SEQ_LEN=30).

Per explicit instruction, every DL model here trains up to 100 epochs with
early stopping (patience 22) instead of the original's fixed 15-epoch loop -
this script does NOT reproduce that overfitting-prone shortcut.

No data leakage: chronological 80/20 split per cluster with a purge/embargo
gap equal to the horizon, sequences never cross a ticker boundary, and the
StandardScaler is fit on the training fold only.

Run:
    uv run python training-scripts-new/crypto/02_train_all.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "training-scripts-new"))
from common import architectures as arch  # noqa: E402
from common import dl_train  # noqa: E402
from common.metrics import return_mae, win_rate_from_mae  # noqa: E402
from common.progress import StageProgress  # noqa: E402

DATA_DIR = ROOT / "data-ready" / "crypto"
MODELS_DIR = ROOT / "models-new" / "crypto"
RESULTS_DIR = ROOT / "results-new" / "crypto"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [7, 14, 28, 42, 60, 90, 120]
SEQ_LEN = 30
N_CLUSTERS = 4
TRIALS = 3
TREE_MODELS = ["XGBoost", "LightGBM", "CatBoost", "RandomForest"]
DL_MODELS = ["LSTM", "GRU", "Transformer", "NBEATS_Lite", "TFT_Lite"]
ALL_MODELS = TREE_MODELS + DL_MODELS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_all() -> dict[str, pd.DataFrame]:
    data = {}
    for path in sorted(DATA_DIR.glob("*.csv")):
        df = pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()
        if len(df) >= 200:
            data[path.stem] = df
    return data


def cluster_assets(data: dict[str, pd.DataFrame]) -> dict[str, int]:
    rows = []
    for ticker, df in data.items():
        vol = df["BTC_Volatility_30d"].mean() if "BTC_Volatility_30d" in df else 0.0
        ret = df["Target_7d"].mean() if "Target_7d" in df else 0.0
        rows.append({"Ticker": ticker, "Vol": vol, "Ret": ret})
    stats = pd.DataFrame(rows).set_index("Ticker")
    labels = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10).fit_predict(
        StandardScaler().fit_transform(stats)
    )
    return dict(zip(stats.index, (int(x) for x in labels), strict=True))


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not c.startswith("Target_")]


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    if len(X) <= seq_len:
        return np.empty((0, seq_len, X.shape[1])), np.empty((0,))
    xs = np.stack([X[i : i + seq_len] for i in range(len(X) - seq_len + 1)])
    ys = y[seq_len - 1 :]
    return xs, ys


def prepare_cluster_horizon(members: dict[str, pd.DataFrame], horizon: int):
    target_col = f"Target_{horizon}d"
    all_dates = sorted(set().union(*[set(df.index) for df in members.values()]))
    split_date = pd.Timestamp(all_dates[int(len(all_dates) * 0.8)])
    val_start = split_date + pd.Timedelta(days=horizon)

    feat_cols = feature_columns(next(iter(members.values())))

    train_by_ticker, val_by_ticker = {}, {}
    for ticker, df in members.items():
        d = df.dropna(subset=[target_col])
        train_by_ticker[ticker] = d[d.index < split_date]
        val_by_ticker[ticker] = d[d.index >= val_start]

    train_all = pd.concat(train_by_ticker.values()) if train_by_ticker else pd.DataFrame(columns=feat_cols + [target_col])
    if len(train_all) < 100:
        return None

    scaler = StandardScaler().fit(train_all[feat_cols].to_numpy(dtype=np.float64))

    def build(frames_by_ticker: dict[str, pd.DataFrame]):
        xs, ys = [], []
        for ticker, d in frames_by_ticker.items():
            if len(d) <= SEQ_LEN:
                continue
            Xs = scaler.transform(d[feat_cols].to_numpy(dtype=np.float64))
            y = d[target_col].to_numpy(dtype=np.float64)
            xseq, yseq = make_sequences(Xs, y, SEQ_LEN)
            if len(xseq):
                xs.append(xseq)
                ys.append(yseq)
        if not xs:
            return np.empty((0, SEQ_LEN, len(feat_cols))), np.empty((0,))
        return np.concatenate(xs), np.concatenate(ys)

    X_train_seq, y_train = build(train_by_ticker)
    X_val_seq, y_val = build(val_by_ticker)

    if len(X_train_seq) < 50 or len(X_val_seq) < 20:
        return None

    return {
        "scaler": scaler,
        "feat_cols": feat_cols,
        "X_train_seq": X_train_seq, "y_train": y_train,
        "X_val_seq": X_val_seq, "y_val": y_val,
        "X_train_2d": X_train_seq[:, -1, :], "X_val_2d": X_val_seq[:, -1, :],
    }


def train_tree(model_name: str, X_train, y_train, X_val, y_val) -> tuple[object, float]:
    def objective(trial: optuna.Trial) -> float:
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        max_depth = trial.suggest_int("max_depth", 3, 7)
        if model_name == "XGBoost":
            m = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        elif model_name == "LightGBM":
            m = LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, verbose=-1, n_jobs=-1)
        elif model_name == "CatBoost":
            m = CatBoostRegressor(iterations=n_estimators, depth=max_depth, random_seed=42, verbose=0)
        else:  # RandomForest
            m = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        m.fit(X_train, y_train)
        return return_mae(y_val, m.predict(X_val))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=TRIALS, show_progress_bar=False)

    best = study.best_params
    if model_name == "XGBoost":
        model = XGBRegressor(n_estimators=best["n_estimators"], max_depth=best["max_depth"], random_state=42, n_jobs=-1)
    elif model_name == "LightGBM":
        model = LGBMRegressor(n_estimators=best["n_estimators"], max_depth=best["max_depth"], random_state=42, verbose=-1, n_jobs=-1)
    elif model_name == "CatBoost":
        model = CatBoostRegressor(iterations=best["n_estimators"], depth=best["max_depth"], random_seed=42, verbose=0)
    else:
        model = RandomForestRegressor(n_estimators=best["n_estimators"], max_depth=best["max_depth"], random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    mae = return_mae(y_val, model.predict(X_val))
    return model, mae


def build_dl_model(model_name: str, n_features: int) -> nn.Module:
    if model_name == "LSTM":
        return arch.CryptoLSTM(n_features)
    if model_name == "GRU":
        return arch.CryptoGRU(n_features)
    if model_name == "Transformer":
        return arch.CryptoTransformer(n_features)
    if model_name == "NBEATS_Lite":
        return arch.NBeatsLite(n_features, SEQ_LEN)
    if model_name == "TFT_Lite":
        return arch.TFTLite(n_features)
    raise ValueError(model_name)


def train_dl(model_name: str, n_features: int, X_train, y_train, X_val, y_val, desc: str):
    model = build_dl_model(model_name, n_features)
    model, history = dl_train.train_torch_model(
        model, X_train, y_train, X_val, y_val,
        device=DEVICE, desc=desc, batch_size=2048, lr=1e-3,
    )
    preds = dl_train.predict(model, X_val, DEVICE)
    mae = return_mae(y_val, preds)
    return model, mae, history


def run():
    print("Loading data-ready/crypto/*.csv ...")
    data = load_all()
    print(f"Loaded {len(data)} tickers")
    if not data:
        print("No data found - run 01_feature_engineering.py first.")
        return

    print("Clustering (KMeans, k=4, random_state=42, on [BTC_Volatility_30d, Target_7d])...")
    cluster_map = cluster_assets(data)
    clusters: dict[int, dict[str, pd.DataFrame]] = {c: {} for c in range(N_CLUSTERS)}
    for ticker, c in cluster_map.items():
        clusters[c][ticker] = data[ticker]
    for c in range(N_CLUSTERS):
        print(f"  Cluster_{c}: {len(clusters[c])} assets")

    with open(RESULTS_DIR / "cluster_map.json", "w") as f:
        json.dump({"clusters": cluster_map}, f, indent=2)

    ensemble_weights: dict[str, dict[str, dict[str, float]]] = {f"Cluster_{c}": {} for c in range(N_CLUSTERS)}

    total_cells = N_CLUSTERS * len(HORIZONS)
    progress = StageProgress("Crypto training (cluster x horizon cells)", total_cells)

    for c in range(N_CLUSTERS):
        members = clusters[c]
        if len(members) < 2:
            print(f"  Cluster_{c} has <2 members, skipping")
            for h in HORIZONS:
                progress.step(f"Cluster_{c}/{h}d SKIPPED (too few members)")
            continue

        for h in HORIZONS:
            cluster_key, horizon_key = f"Cluster_{c}", f"{h}d"
            prepped = prepare_cluster_horizon(members, h)
            if prepped is None:
                ensemble_weights[cluster_key][horizon_key] = {m: 0.0 for m in ALL_MODELS}
                progress.step(f"{cluster_key}/{horizon_key}: SKIPPED (insufficient data)")
                continue

            n_features = len(prepped["feat_cols"])
            scaler_path = MODELS_DIR / f"Cluster_{c}_{h}d_scaler.pkl"
            joblib.dump(prepped["scaler"], scaler_path)

            weights = {}
            notes = []
            for model_name in TREE_MODELS:
                try:
                    model, mae = train_tree(
                        model_name, prepped["X_train_2d"], prepped["y_train"],
                        prepped["X_val_2d"], prepped["y_val"],
                    )
                    joblib.dump(model, MODELS_DIR / f"Cluster_{c}_{h}d_{model_name}.pkl")
                    weights[model_name] = win_rate_from_mae(mae)
                    notes.append(f"{model_name}={weights[model_name]}")
                except Exception as e:
                    weights[model_name] = 0.0
                    notes.append(f"{model_name}=FAIL({e})")

            for model_name in DL_MODELS:
                try:
                    desc = f"C{c}/{h}d/{model_name}"
                    model, mae, hist = train_dl(
                        model_name, n_features, prepped["X_train_seq"], prepped["y_train"],
                        prepped["X_val_seq"], prepped["y_val"], desc,
                    )
                    torch.save(model.state_dict(), MODELS_DIR / f"Cluster_{c}_{h}d_{model_name}.pt")
                    weights[model_name] = win_rate_from_mae(mae)
                    notes.append(f"{model_name}={weights[model_name]}({hist['epochs_run']}ep)")
                except Exception as e:
                    weights[model_name] = 0.0
                    notes.append(f"{model_name}=FAIL({e})")

            ensemble_weights[cluster_key][horizon_key] = weights
            progress.step(f"{cluster_key}/{horizon_key}: " + ", ".join(notes))

            with open(RESULTS_DIR / "ensemble_weights.json", "w") as f:
                json.dump(ensemble_weights, f, indent=2)

    progress.close()
    print(f"Artifacts: {MODELS_DIR}")
    print(f"Weights + cluster map: {RESULTS_DIR}")


if __name__ == "__main__":
    run()
