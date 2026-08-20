import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

INPUT_DIR = "data/mufap_clustered"
MODEL_DIR = "models/mufap"
RESULTS_DIR = "results/mufap"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

HORIZONS = [7, 14, 28, 42, 60, 90, 120]
CLUSTERS = ['Equity', 'MoneyMarket', 'Income', 'Commodity', 'Balanced']

# DL Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1024
EPOCHS = 150
PATIENCE = 15

# --- Neural Network Architecture ---
class MUFAP_LSTM(nn.Module):
    def __init__(self, input_dim):
        super(MUFAP_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x is (Batch, Features), we need (Batch, Seq, Features) for LSTM. We treat it as Seq=1 for this feature vector.
        x = x.unsqueeze(1) 
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Take last hidden state
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out.squeeze(1)

def train_lstm(X_train, y_train, X_val, y_val, model_path):
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = MUFAP_LSTM(X_train.shape[1]).to(DEVICE)
    criterion = nn.HuberLoss() # Robust to outliers
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) # L2 Regularization
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break # Early Stopping

    # Load best weights to calculate MAE
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    with torch.no_grad():
        preds = model(X_val_t).cpu().numpy()
        
    return mean_absolute_error(y_val, preds)

def train_master_models():
    results_map = {}

    for cluster in CLUSTERS:
        cluster_dir = os.path.join(INPUT_DIR, cluster)
        files = glob.glob(os.path.join(cluster_dir, "*.csv"))
        if not files:
            continue
            
        logger.info(f"--- Loading Data for {cluster} Cluster ({len(files)} funds) ---")
        dfs = []
        for f in files:
            df = pd.read_csv(f)
            # Ensure dates are parsed
            if 'Validity_Date' in df.columns:
                df['Validity_Date'] = pd.to_datetime(df['Validity_Date'])
            dfs.append(df)
            
        cluster_df = pd.concat(dfs, ignore_index=True)
        cluster_df.sort_values('Validity_Date', inplace=True)
        
        # Calculate 80% split date globally
        unique_dates = cluster_df['Validity_Date'].unique()
        split_idx = int(len(unique_dates) * 0.8)
        split_date = unique_dates[split_idx]
        logger.info(f"Global Split Date for {cluster}: {split_date}")
        
        # Base Feature Columns
        drop_cols = ['Validity_Date', 'Fund'] + [f'Target_{h}d' for h in HORIZONS]
        feature_cols = [c for c in cluster_df.columns if c not in drop_cols and cluster_df[c].dtype != 'object']
        
        # Isolate Scaler on Training Data ONLY (Anti-Leakage)
        train_scaler_df = cluster_df[cluster_df['Validity_Date'] < split_date].copy()
        scaler = StandardScaler()
        scaler.fit(train_scaler_df[feature_cols])
        
        # Save exact scaler for live inference
        scaler_path = os.path.join(MODEL_DIR, f"{cluster}_scaler.pkl")
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved perfectly isolated scaler: {scaler_path}")
        
        results_map[cluster] = {}
        
        for horizon in HORIZONS:
            logger.info(f"Training {cluster} | Horizon: {horizon}d")
            
            target_col = f'Target_{horizon}d'
            df_h = cluster_df.dropna(subset=[target_col]).copy()
            
            # The Purge Gap
            train_mask = df_h['Validity_Date'] < (split_date - pd.Timedelta(days=horizon))
            val_mask = df_h['Validity_Date'] >= split_date
            
            train_df = df_h[train_mask]
            val_df = df_h[val_mask]
            
            if len(train_df) == 0 or len(val_df) == 0:
                logger.warning(f"Skipping {horizon}d due to insufficient data after purge.")
                continue
                
            X_train = scaler.transform(train_df[feature_cols])
            y_train = train_df[target_col].values
            
            X_val = scaler.transform(val_df[feature_cols])
            y_val = val_df[target_col].values
            
            best_mae = float('inf')
            best_model_name = ""
            
            # 1. XGBoost
            xgb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, early_stopping_rounds=50, random_state=42)
            xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            xgb_preds = xgb_model.predict(X_val)
            xgb_mae = mean_absolute_error(y_val, xgb_preds)
            
            if xgb_mae < best_mae:
                best_mae = xgb_mae
                best_model_name = "XGBoost"
                joblib.dump(xgb_model, os.path.join(MODEL_DIR, f"{cluster}_{horizon}d_XGBoost.pkl"))
                
            # 2. LightGBM
            lgb_model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42)
            # Use callbacks for early stopping in modern lightgbm
            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
            lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
            lgb_preds = lgb_model.predict(X_val)
            lgb_mae = mean_absolute_error(y_val, lgb_preds)
            
            if lgb_mae < best_mae:
                best_mae = lgb_mae
                best_model_name = "LightGBM"
                joblib.dump(lgb_model, os.path.join(MODEL_DIR, f"{cluster}_{horizon}d_LightGBM.pkl"))
                
            # 3. LSTM
            lstm_path = os.path.join(MODEL_DIR, f"{cluster}_{horizon}d_LSTM.pt")
            lstm_mae = train_lstm(X_train, y_train, X_val, y_val, lstm_path)
            
            if lstm_mae < best_mae:
                best_mae = lstm_mae
                best_model_name = "LSTM"
            else:
                # Discard LSTM if it didn't win to save space
                if os.path.exists(lstm_path): os.remove(lstm_path)
            
            logger.info(f"   -> Winner: {best_model_name} (MAE: {best_mae:.5f})")
            
            results_map[cluster][f"{horizon}d"] = {
                "Winning_Model": best_model_name,
                "MAE": float(best_mae),
                "Path": f"{cluster}_{horizon}d_{best_model_name}.{'pt' if best_model_name == 'LSTM' else 'pkl'}"
            }
            
    # Save the Master Routing Table
    routing_path = os.path.join(RESULTS_DIR, "ensemble_weights.json")
    with open(routing_path, "w") as f:
        json.dump(results_map, f, indent=4)
        
    logger.info(f"🎉 Stage 4 Master Training Complete! Routing table saved to {routing_path}")

if __name__ == "__main__":
    train_master_models()
