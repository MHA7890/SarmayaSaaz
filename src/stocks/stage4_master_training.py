import os
import glob
import json
import logging
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

import torch
import torch.nn as nn
import torch.optim as optim

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# --- Configuration ---
INPUT_DIR = "data/stocks/clustered"
MODEL_DIR = "models/stocks"
RESULTS_FILE = "docs/stocks_results.md"
os.makedirs(MODEL_DIR, exist_ok=True)

HORIZONS = [7, 14, 28, 42, 60, 90, 120]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- PyTorch LSTM Architecture ---
class PSX_LSTM(nn.Module):
    def __init__(self, input_dim):
        super(PSX_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x shape: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out.squeeze(1)

def train_pytorch(X_train, y_train, X_val, y_val, model_path):
    model = PSX_LSTM(X_train.shape[1]).to(DEVICE)
    criterion = nn.HuberLoss() # Outlier resistant
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4) # L2 Regularization

    X_tr = torch.FloatTensor(X_train).to(DEVICE)
    y_tr = torch.FloatTensor(y_train).to(DEVICE)
    X_va = torch.FloatTensor(X_val).to(DEVICE)
    y_va = torch.FloatTensor(y_val).to(DEVICE)

    best_loss = float('inf')
    patience = 15
    patience_counter = 0

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        preds = model(X_tr)
        loss = criterion(preds, y_tr)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_va)
            val_loss = criterion(val_preds, y_va)

        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Load best and calculate MAE
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    with torch.no_grad():
        final_preds = model(X_va).cpu().numpy()
        
    return mean_absolute_error(y_val, final_preds)

def write_results_header():
    with open(RESULTS_FILE, "w") as f:
        f.write("# PSX Stocks Engine: ML/DL Training Results\n\n")
        f.write("This document logs the Validation Mean Absolute Error (MAE) for all models trained across the 7 Super-Sectors.\n\n")

def append_results(sector, results_dict):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"## Sector: {sector}\n")
        for horizon in HORIZONS:
            f.write(f"### Horizon: {horizon} Days\n")
            f.write(f"- **XGBoost MAE:** {results_dict[horizon]['XGBoost']:.6f}\n")
            f.write(f"- **LightGBM MAE:** {results_dict[horizon]['LightGBM']:.6f}\n")
            f.write(f"- **PyTorch LSTM MAE:** {results_dict[horizon]['LSTM']:.6f}\n")
            f.write(f"🏆 **Winner:** {results_dict[horizon]['Winner']} (Saved to JSON)\n\n")

def run_stage4():
    write_results_header()
    ensemble_table = {}
    sectors = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    
    logger.info(f"Initiating Master Training on {DEVICE} for {len(sectors)} Sectors...")
    
    for sector in sectors:
        logger.info(f"--- Processing Sector: {sector} ---")
        sector_path = os.path.join(INPUT_DIR, sector)
        files = glob.glob(os.path.join(sector_path, "*.csv"))
        
        if not files:
            logger.warning(f"No data files found for {sector}. Skipping.")
            continue
            
        # 1. Assemble Massive Matrix
        df_list = []
        for f in files:
            temp_df = pd.read_csv(f, index_col='Date', parse_dates=True)
            df_list.append(temp_df)
            
        master_df = pd.concat(df_list)
        master_df.sort_index(inplace=True) # Sort chronologically for anti-leakage split
        
        ensemble_table[sector] = {}
        sector_results = {}
        
        # Determine Feature Columns (Exclude Target columns)
        all_cols = master_df.columns
        feature_cols = [c for c in all_cols if not str(c).startswith('Target_')]
        
        for horizon in HORIZONS:
            logger.info(f"   -> Training Horizon: {horizon}d")
            target_col = f'Target_{horizon}d'
            
            # Drop NaNs for this specific horizon (future hasn't happened yet)
            clean_df = master_df.dropna(subset=[target_col] + feature_cols).copy()
            
            if len(clean_df) < 500:
                logger.warning(f"Insufficient data for {sector} {horizon}d. Skipping.")
                continue
                
            # 2. Chronological Purge Split
            split_idx = int(len(clean_df) * 0.8)
            purge_gap = 120 # 120 rows gap to prevent overlap leakage
            
            train_df = clean_df.iloc[:split_idx]
            val_df = clean_df.iloc[split_idx + purge_gap:]
            
            if len(val_df) < 50:
                logger.warning(f"Validation block too small after purge gap. Adjusting split.")
                val_df = clean_df.iloc[split_idx:] # Fallback if data is too small
                
            X_train_raw = train_df[feature_cols].values
            y_train = train_df[target_col].values
            X_val_raw = val_df[feature_cols].values
            y_val = val_df[target_col].values
            
            # 3. Isolated Scaling
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_val = scaler.transform(X_val_raw)
            
            # Save scaler (overwrite for each horizon is fine, it's mostly the same features)
            scaler_path = os.path.join(MODEL_DIR, f"{sector}_scaler.pkl")
            joblib.dump(scaler, scaler_path)
            
            metrics = {}
            
            # --- Model 1: LightGBM ---
            lgb_path = os.path.join(MODEL_DIR, f"{sector}_{horizon}d_LightGBM.pkl")
            lgb_model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, n_jobs=-1, random_state=42)
            # Use early stopping via callbacks
            lgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
            )
            lgb_preds = lgb_model.predict(X_val)
            metrics['LightGBM'] = mean_absolute_error(y_val, lgb_preds)
            joblib.dump(lgb_model, lgb_path)
            
            # --- Model 2: XGBoost ---
            xgb_path = os.path.join(MODEL_DIR, f"{sector}_{horizon}d_XGBoost.pkl")
            xgb_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=-1, random_state=42, reg_lambda=1.0)
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            # XGBoost python API requires explicit early_stopping_rounds in fit, or we just trust the regularizer.
            # To avoid API warnings, we train fixed 500 with high lambda.
            xgb_preds = xgb_model.predict(X_val)
            metrics['XGBoost'] = mean_absolute_error(y_val, xgb_preds)
            joblib.dump(xgb_model, xgb_path)
            
            # --- Model 3: PyTorch LSTM ---
            lstm_path = os.path.join(MODEL_DIR, f"{sector}_{horizon}d_LSTM.pt")
            metrics['LSTM'] = train_pytorch(X_train, y_train, X_val, y_val, lstm_path)
            
            # --- Gladiator Evaluation ---
            winner = min(metrics, key=metrics.get)
            metrics['Winner'] = winner
            sector_results[horizon] = metrics
            
            # Route to JSON
            if winner == "LSTM":
                winning_path = f"{sector}_{horizon}d_LSTM.pt"
            else:
                winning_path = f"{sector}_{horizon}d_{winner}.pkl"
                
            ensemble_table[sector][f"{horizon}d"] = {
                "Winning_Model": winner,
                "MAE": metrics[winner],
                "Path": winning_path
            }
            
        # Log to Markdown
        append_results(sector, sector_results)
        
    # Finalize JSON Table
    with open(os.path.join(MODEL_DIR, "ensemble_weights.json"), "w") as f:
        json.dump(ensemble_table, f, indent=4)
        
    logger.info("=" * 50)
    logger.info("Stage 4 Complete! 147 Models Trained. Routing Table and Scalers Saved.")
    logger.info(f"Detailed metrics available in: {RESULTS_FILE}")
    logger.info("=" * 50)

if __name__ == "__main__":
    run_stage4()
