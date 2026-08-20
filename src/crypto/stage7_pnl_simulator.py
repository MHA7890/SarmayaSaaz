import os
import glob
import json
import logging
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Import architectures
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from stage4_master_training import CryptoLSTM, CryptoGRU, CryptoTransformer, NBEATS_Lite, TFT_Lite

# --- Setup & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

INPUT_DIR = "data/crypto_processed"
MODELS_DIR = "models/crypto"
RESULTS_DIR = "results/crypto"
TICKER = "BTC"  # We will simulate the flagship asset
HORIZON = 7
SEQ_LEN = 30

INITIAL_CAPITAL = 100000.0
FEE_RATE = 0.001  # 0.1% exchange fee

def get_cluster_mapping():
    files = glob.glob(f"{INPUT_DIR}/*.csv")
    stats = []
    for f in files:
        ticker = os.path.basename(f).replace('.csv', '')
        df = pd.read_csv(f, index_col='Date')
        vol = df['BTC_Volatility_30d'].mean() if 'BTC_Volatility_30d' in df.columns else 0
        ret = df['Target_7d'].mean() if 'Target_7d' in df.columns else 0
        stats.append({'Ticker': ticker, 'Vol': vol, 'Ret': ret})
    
    stats_df = pd.DataFrame(stats).set_index('Ticker')
    stats_df['Cluster'] = KMeans(n_clusters=4, random_state=42).fit_predict(StandardScaler().fit_transform(stats_df))
    return stats_df

def load_models_into_memory(cluster_key, horizon_key, features_count, weights_dict):
    """Pre-loads all models above the 0.4 threshold to avoid loading them 365 times in the loop."""
    loaded_models = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for model_name, weight in weights_dict.items():
        if weight <= 0.4:
            continue
            
        file_path = f"{MODELS_DIR}/{cluster_key}_{horizon_key}_{model_name}"
        
        if model_name in ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest']:
            path = f"{file_path}.pkl"
            if os.path.exists(path):
                loaded_models[model_name] = joblib.load(path)
        else:
            path = f"{file_path}.pt"
            if os.path.exists(path):
                if model_name == 'LSTM': model = CryptoLSTM(features_count)
                elif model_name == 'GRU': model = CryptoGRU(features_count)
                elif model_name == 'Transformer': model = CryptoTransformer(features_count)
                elif model_name == 'NBEATS_Lite': model = NBEATS_Lite(features_count)
                else: model = TFT_Lite(features_count)
                
                model.load_state_dict(torch.load(path, map_location=device))
                model.to(device)
                model.eval()
                loaded_models[model_name] = model
                
    return loaded_models

def main():
    logger.info(f"Initializing Phase 7 Advanced PnL Simulator for {TICKER}...")
    
    file_path = f"{INPUT_DIR}/{TICKER}.csv"
    if not os.path.exists(file_path):
        logger.error(f"Data for {TICKER} not found.")
        return
        
    # Get cluster
    cluster_map = get_cluster_mapping()
    cluster_id = cluster_map.loc[TICKER, 'Cluster']
    cluster_key = f"Cluster_{cluster_id}"
    horizon_key = f"{HORIZON}d"
    
    # Read ensemble weights
    try:
        with open(f"{RESULTS_DIR}/ensemble_weights.json", "r") as f:
            ensemble_weights = json.load(f)
    except:
        logger.error("Could not load ensemble_weights.json")
        return
        
    model_weights = ensemble_weights.get(cluster_key, {}).get(horizon_key, {})
    if not model_weights:
        logger.error(f"No weights found for {cluster_key} {horizon_key}")
        return
    
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    features = [c for c in df.columns if 'Target_' not in c and c != 'Date']
    features_count = len(features)
    
    # Load all models for this asset's cluster
    logger.info(f"Loading {cluster_key} ensemble models for {horizon_key}...")
    loaded_models = load_models_into_memory(cluster_key, horizon_key, features_count, model_weights)
    logger.info(f"Successfully loaded {len(loaded_models)} deep learning / ensemble models into memory.")
    
    # We will backtest over the last 365 days
    backtest_days = 365
    if len(df) < backtest_days + SEQ_LEN:
        logger.error("Not enough data to backtest 365 days.")
        return
        
    # Need actual daily returns to calculate PnL
    import yfinance as yf
    asset = yf.Ticker(f"{TICKER}-USD")
    raw_df = asset.history(period="2y").reset_index()
    raw_df['Date'] = pd.to_datetime(raw_df['Date']).dt.tz_localize(None)
    
    # Simulate PnL
    equity_curve = []
    strategy_capital = INITIAL_CAPITAL
    buy_hold_capital = INITIAL_CAPITAL
    
    position = 0 # 0 = Cash, 1 = Long
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("Simulating Historical Trading Engine with 8.5-Year Sentiment AI...")
    
    start_idx = len(df) - backtest_days
    
    for i in range(start_idx, len(df)):
        # Data available up to day i
        historical_df = df.iloc[:i]
        date = historical_df['Date'].iloc[-1]
        
        # Scale using the perfectly purged scaler from Phase 4
        scaler_path = f"{MODELS_DIR}/{cluster_key}_{horizon_key}_scaler.pkl"
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            X_today = scaler.transform(historical_df[features].values)[-SEQ_LEN:]
        else:
            X_today = StandardScaler().fit_transform(historical_df[features].values)[-SEQ_LEN:]
        
        predictions = []
        weights_used = []
        
        for name, model in loaded_models.items():
            weight = model_weights[name]
            if name in ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest']:
                X_single = X_today[-1].reshape(1, -1)
                pred = model.predict(X_single)[0]
                predictions.append(float(pred) * weight)
            else:
                X_tensor = torch.tensor(X_today, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model(X_tensor).item()
                predictions.append(pred * weight)
            weights_used.append(weight)
            
        final_prediction = sum(predictions) / sum(weights_used) if sum(weights_used) > 0 else 0.0
        
        # Find daily return for the *next* day to evaluate the decision
        next_date = df['Date'].iloc[i]
        match = raw_df[raw_df['Date'] == next_date]
        daily_ret = match['Close'].pct_change().iloc[0] if len(match) > 1 else (match['Close'].values[0] / raw_df[raw_df['Date'] == date]['Close'].values[0] - 1 if len(match) > 0 and len(raw_df[raw_df['Date'] == date]) > 0 else 0)
        
        # Buy & Hold simply absorbs the daily return
        buy_hold_capital *= (1 + daily_ret)
        
        # Strategy Logic
        if position == 1:
            strategy_capital *= (1 + daily_ret)
        
        # Trade Decision (Executes for next day)
        # Tighter thresholds because we are using the powerful ensemble now
        if final_prediction > 0.02 and position == 0:
            position = 1
            strategy_capital *= (1 - FEE_RATE) # Pay fee to enter
        elif final_prediction < -0.01 and position == 1:
            position = 0
            strategy_capital *= (1 - FEE_RATE) # Pay fee to exit
            
        equity_curve.append({
            'Date': next_date.strftime('%Y-%m-%d'),
            'Strategy_Equity': round(strategy_capital, 2),
            'Buy_Hold_Equity': round(buy_hold_capital, 2)
        })
        
    equity_df = pd.DataFrame(equity_curve)
    
    # Calculate Metrics
    strat_return = ((strategy_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    bh_return = ((buy_hold_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    # Drawdown
    equity_df['Peak'] = equity_df['Strategy_Equity'].cummax()
    equity_df['Drawdown'] = (equity_df['Strategy_Equity'] - equity_df['Peak']) / equity_df['Peak']
    max_dd = equity_df['Drawdown'].min() * 100
    
    csv_path = f"{RESULTS_DIR}/equity_curve.csv"
    equity_df.drop(columns=['Peak', 'Drawdown']).to_csv(csv_path, index=False)
    
    logger.info(f"✅ Simulation Complete!")
    logger.info(f"Strategy ROI: {strat_return:+.2f}% | Buy & Hold ROI: {bh_return:+.2f}%")
    logger.info(f"Max Drawdown: {max_dd:.2f}%")
    logger.info(f"Equity curve saved to {csv_path}")

if __name__ == "__main__":
    main()
