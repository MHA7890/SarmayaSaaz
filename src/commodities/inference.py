import os
import joblib
import torch
import yfinance as yf
import pandas as pd
import numpy as np

# Import our feature engineering logic so we don't repeat code
from stage2_feature_engineering import engineer_features
# Import the network architectures so we can load the weights
from stage2_dl_models import LSTMModel, TransformerModel

COMMODITY_TICKERS = {
    'gold': 'GC=F',
    'silver': 'SI=F',
    'copper': 'HG=F',
    'natural_gas': 'NG=F',
    'crude_oil': 'CL=F',
    'wheat': 'ZW=F'
}

def fetch_live_data(commodity_name, days=150):
    ticker = COMMODITY_TICKERS.get(commodity_name)
    if not ticker:
        raise ValueError(f"Unknown commodity: {commodity_name}")
    
    # Download live data
    df = yf.download(ticker, period=f"{days}d", progress=False)
    
    # Handle yfinance multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    base_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    
    # Ensure all required columns are present
    for col in base_cols:
        if col not in df.columns:
            df[col] = df['Close'] if col == 'Adj Close' else 0
            
    df = df[base_cols]
    
    # Handle negative prices as we did in Stage 0
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
        df.loc[df[col] <= 0, col] = 0.01
        
    # Forward fill missing values
    df.ffill(inplace=True)
    return df

def run_inference(commodity_name, model_type):
    # 1. Fetch live data
    df = fetch_live_data(commodity_name, days=150) # Need enough days for 50-day MAs
    
    # 2. Engineer features
    df = engineer_features(df)
    
    drop_cols = ['Target_Close_Next', 'Target_Return_Next', 'Target_Direction', 
                 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    features = [col for col in df.columns if col not in drop_cols]
    
    # Drop rows that have NaN in the FEATURES (due to rolling windows).
    # We do NOT want to dropna on the whole dataframe because the very last row 
    # will naturally have NaN for 'Target_Close_Next' (since tomorrow hasn't happened yet).
    df.dropna(subset=features, inplace=True)
    
    today_close = df['Close'].iloc[-1]
    last_date = df.index[-1].strftime('%Y-%m-%d')
    
    models_dir = './models'
    
    if model_type in ['XGBoost', 'LightGBM']:
        # For tree models, we just need the single most recent row
        X_live = df[features].iloc[-1:]
        
        if model_type == 'XGBoost':
            model_path = os.path.join(models_dir, f'xgb_return_{commodity_name}.pkl')
        else:
            model_path = os.path.join(models_dir, f'lgb_return_{commodity_name}.pkl')
            
        model = joblib.load(model_path)
        pred_return = float(model.predict(X_live)[0])
        
    elif model_type in ['LSTM', 'Transformer']:
        # For DL models, we need the last 10 rows for the sequence
        SEQ_LENGTH = 10
        if len(df) < SEQ_LENGTH:
            raise ValueError("Not enough data to form a sequence.")
            
        X_live_raw = df[features].iloc[-SEQ_LENGTH:].values
        
        # Load the fitted StandardScaler
        scaler_path = os.path.join(models_dir, f'scaler_{commodity_name}.pkl')
        scaler = joblib.load(scaler_path)
        
        X_live_scaled = scaler.transform(X_live_raw)
        
        # Reshape to [batch_size=1, seq_length=10, num_features]
        X_live_tensor = torch.tensor(X_live_scaled, dtype=torch.float32).unsqueeze(0)
        
        input_size = len(features)
        HIDDEN_SIZE = 32
        
        if model_type == 'LSTM':
            model = LSTMModel(input_size, HIDDEN_SIZE)
            model_path = os.path.join(models_dir, f'lstm_{commodity_name}.pt')
        else:
            model = TransformerModel(input_size, HIDDEN_SIZE)
            model_path = os.path.join(models_dir, f'transformer_{commodity_name}.pt')
            
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        with torch.no_grad():
            pred_return = float(model(X_live_tensor).item())
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Convert predicted return to price
    pred_price = today_close * (1 + pred_return)
    pred_direction = "Up" if pred_return > 0 else "Down"
    
    return {
        'Commodity': commodity_name.upper(),
        'Model': model_type,
        'Last_Date': last_date,
        'Last_Close': round(today_close, 4),
        'Predicted_Return_Pct': round(pred_return * 100, 2),
        'Predicted_Price': round(pred_price, 4),
        'Direction': pred_direction
    }

if __name__ == "__main__":
    # Quick test to ensure everything works
    print("Running live inference test for Gold (XGBoost)...")
    result = run_inference('gold', 'XGBoost')
    for k, v in result.items():
        print(f"{k}: {v}")
