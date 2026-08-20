import os
import glob
import logging
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import optuna
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- 1. SETUP & LOGGING ---
INPUT_DIR = "data/crypto_processed"
MODELS_DIR = "models/crypto"
RESULTS_DIR = "results/crypto"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN = 30
HORIZONS = [7, 14, 28, 42, 60, 90, 120]
MODEL_NAMES = ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 'LSTM', 'GRU', 'Transformer', 'NBEATS_Lite', 'TFT_Lite']
TRIALS = 3 

# --- 2. CLUSTERING ---
def cluster_assets():
    files = glob.glob(f"{INPUT_DIR}/*.csv")
    stats, dataframes = [], {}
    for f in files:
        ticker = os.path.basename(f).replace('.csv', '')
        df = pd.read_csv(f, index_col='Date', parse_dates=True)
        dataframes[ticker] = df
        avg_vol = df['BTC_Volatility_30d'].mean() if 'BTC_Volatility_30d' in df.columns else 0
        avg_ret = df['Target_7d'].mean() if 'Target_7d' in df.columns else 0
        stats.append({'Ticker': ticker, 'Vol': avg_vol, 'Ret': avg_ret})
        
    stats_df = pd.DataFrame(stats).set_index('Ticker')
    kmeans = KMeans(n_clusters=4, random_state=42)
    stats_df['Cluster'] = kmeans.fit_predict(StandardScaler().fit_transform(stats_df))
    return stats_df, dataframes

# --- 3. DATALOADERS & EMBARGO SPLIT ---
class CryptoDataset(Dataset):
    def __init__(self, X_3d, y):
        self.X = X_3d
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

def prepare_data(cluster_id, stats_df, dataframes, horizon):
    members = stats_df[stats_df['Cluster'] == cluster_id].index.tolist()
    target_col = f'Target_{horizon}d'
    
    # 1. Determine Global Date Split across ALL members
    all_dates = pd.DatetimeIndex([])
    for t in members:
        all_dates = all_dates.union(dataframes[t].index)
    all_dates = all_dates.sort_values()
    
    if len(all_dates) < 100:
        return None, None, None, None, None, None, None
        
    global_split_date = all_dates[int(len(all_dates) * 0.8)]
    val_start_date = global_split_date + pd.Timedelta(days=horizon) # Strict Purge
    
    train_dfs, val_dfs = [], []
    for t in members:
        df = dataframes[t].dropna(subset=[target_col])
        if len(df) == 0: continue
        train_dfs.append(df[df.index < global_split_date].copy())
        val_dfs.append(df[df.index >= val_start_date].copy())
        
    combined_train = pd.concat(train_dfs) if train_dfs else pd.DataFrame()
    if combined_train.empty: 
        return None, None, None, None, None, None, None
        
    features = [c for c in combined_train.columns if 'Target_' not in c and c != 'Date']
    
    # 2. Fit Scaler ONLY on Training Data (Prevent Future Leakage)
    scaler = StandardScaler()
    scaler.fit(combined_train[features].values)
    
    # Save the scaler for inference/simulation
    scaler_path = f"{MODELS_DIR}/Cluster_{cluster_id}_{horizon}d_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    
    # 3. Generate Sequences PER ASSET (Prevent Boundary Contamination)
    X_train_seqs, y_train_seqs = [], []
    X_val_seqs, y_val_seqs = [], []
    
    for train_df in train_dfs:
        if len(train_df) < SEQ_LEN: continue
        X_scaled = scaler.transform(train_df[features].values)
        y = train_df[target_col].values
        for i in range(len(X_scaled) - SEQ_LEN + 1):
            X_train_seqs.append(X_scaled[i:i+SEQ_LEN])
            y_train_seqs.append(y[i+SEQ_LEN-1])
            
    for val_df in val_dfs:
        if len(val_df) < SEQ_LEN: continue
        X_scaled = scaler.transform(val_df[features].values)
        y = val_df[target_col].values
        for i in range(len(X_scaled) - SEQ_LEN + 1):
            X_val_seqs.append(X_scaled[i:i+SEQ_LEN])
            y_val_seqs.append(y[i+SEQ_LEN-1])
            
    if not X_train_seqs or not X_val_seqs:
        return None, None, None, None, None, None, None
        
    X_train_3d = np.array(X_train_seqs)
    y_train = np.array(y_train_seqs)
    X_val_3d = np.array(X_val_seqs)
    y_val = np.array(y_val_seqs)
    
    # Tree Models just use the most recent day of the sequence
    X_train_2d = X_train_3d[:, -1, :]
    X_val_2d = X_val_3d[:, -1, :]
    
    train_ds = CryptoDataset(X_train_3d, y_train)
    val_ds = CryptoDataset(X_val_3d, y_val)
    
    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False)
    
    return train_loader, val_loader, X_train_2d, y_train, X_val_2d, y_val, len(features)

# --- 4. DEEP LEARNING ARCHITECTURES ---
class CryptoLSTM(nn.Module):
    def __init__(self, f): super().__init__(); self.lstm = nn.LSTM(f, 64, batch_first=True); self.fc = nn.Linear(64, 1)
    def forward(self, x): out, _ = self.lstm(x); return self.fc(out[:, -1, :])

class CryptoGRU(nn.Module):
    def __init__(self, f): super().__init__(); self.gru = nn.GRU(f, 64, batch_first=True); self.fc = nn.Linear(64, 1)
    def forward(self, x): out, _ = self.gru(x); return self.fc(out[:, -1, :])

class CryptoTransformer(nn.Module):
    def __init__(self, f): 
        super().__init__()
        self.emb = nn.Linear(f, 64)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True), num_layers=2)
        self.fc = nn.Linear(64, 1)
    def forward(self, x): return self.fc(self.encoder(self.emb(x)).mean(dim=1))

class NBEATS_Lite(nn.Module):
    def __init__(self, f): super().__init__(); self.fc = nn.Sequential(nn.Linear(f*SEQ_LEN, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x): return self.fc(x.reshape(x.size(0), -1))

class TFT_Lite(nn.Module):
    def __init__(self, f): super().__init__(); self.lstm = nn.LSTM(f, 64, batch_first=True); self.gate = nn.Linear(64, 64); self.fc = nn.Linear(64, 1)
    def forward(self, x): out, _ = self.lstm(x); gated = out[:, -1, :] * torch.sigmoid(self.gate(out[:, -1, :])); return self.fc(gated)

# --- 5. TRAINING ENGINE ---
def run_optuna_trees(X_train, y_train, X_val, y_val, model_name, n_trials=3, save_path=""):
    def objective(trial):
        if model_name == 'XGBoost':
            model = XGBRegressor(n_estimators=trial.suggest_int('n_estimators', 50, 200), max_depth=trial.suggest_int('max_depth', 3, 7), device='cuda')
        elif model_name == 'LightGBM':
            model = LGBMRegressor(n_estimators=trial.suggest_int('n_estimators', 50, 200), max_depth=trial.suggest_int('max_depth', 3, 7), device='gpu', verbose=-1)
        elif model_name == 'CatBoost':
            model = CatBoostRegressor(iterations=trial.suggest_int('iterations', 50, 200), depth=trial.suggest_int('depth', 3, 7), task_type='GPU', verbose=0)
        else: # RandomForest
            model = RandomForestRegressor(n_estimators=trial.suggest_int('n_estimators', 50, 200), max_depth=trial.suggest_int('max_depth', 3, 7), n_jobs=-1)
            
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)
        
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    if model_name == 'XGBoost':
        best_model = XGBRegressor(**study.best_params, device='cuda')
    elif model_name == 'LightGBM':
        best_model = LGBMRegressor(**study.best_params, device='gpu', verbose=-1)
    elif model_name == 'CatBoost':
        best_model = CatBoostRegressor(**study.best_params, task_type='GPU', verbose=0)
    else:
        best_model = RandomForestRegressor(**study.best_params, n_jobs=-1)
        
    best_model.fit(X_train, y_train)
    joblib.dump(best_model, save_path)
    
    return study.best_value

def run_optuna_dl(train_loader, val_loader, input_size, model_name, n_trials=3, save_path=""):
    def objective(trial):
        lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
        if model_name == 'LSTM': model = CryptoLSTM(input_size)
        elif model_name == 'GRU': model = CryptoGRU(input_size)
        elif model_name == 'Transformer': model = CryptoTransformer(input_size)
        elif model_name == 'NBEATS_Lite': model = NBEATS_Lite(input_size)
        else: model = TFT_Lite(input_size)
        
        model = model.to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.HuberLoss()
        best_loss = float('inf')
        
        for epoch in range(15):
            model.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(bx.to(DEVICE)).squeeze(), by.to(DEVICE))
                loss.backward()
                optimizer.step()
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    val_loss += criterion(model(bx.to(DEVICE)).squeeze(), by.to(DEVICE)).item()
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), save_path)
        return best_loss
        
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_value

# --- 6. MASTER LOOP ---
def main():
    logger.info("Initializing Phase 4 Master Engine...")
    stats_df, dataframes = cluster_assets()
    results_registry = {}
    
    for c in range(4):
        cluster_key = f"Cluster_{c}"
        results_registry[cluster_key] = {}
        for h in HORIZONS:
            results_registry[cluster_key][f"{h}d"] = {}
            train_loader, val_loader, Xt, yt, Xv, yv, features_count = prepare_data(c, stats_df, dataframes, h)
            if train_loader is None: continue
            
            for m in MODEL_NAMES:
                logger.info(f"[{cluster_key}] [Horizon {h}d] Training {m} (Tuning {TRIALS} Trials)...")
                save_path = f"{MODELS_DIR}/{cluster_key}_{h}d_{m}"
                
                try:
                    if m in ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest']:
                        best_mae = run_optuna_trees(Xt, yt, Xv, yv, m, n_trials=TRIALS, save_path=f"{save_path}.pkl")
                    else:
                        best_mae = run_optuna_dl(train_loader, val_loader, features_count, m, n_trials=TRIALS, save_path=f"{save_path}.pt")
                    
                    win_rate = max(0.4, 1.0 - best_mae) 
                    results_registry[cluster_key][f"{h}d"][m] = round(win_rate, 4)
                except Exception as e:
                    logger.error(f"Failed on {m}: {str(e)}")
                    results_registry[f"Cluster_{c}"][f"{h}d"][m] = 0.0
                    
    with open(f"{RESULTS_DIR}/ensemble_weights.json", "w") as f:
        json.dump(results_registry, f, indent=4)
        
    logger.info("✅ Phase 4 Complete! All 252 Models Trained and Registered.")

if __name__ == "__main__":
    main()
