import os
import yfinance as yf
import pandas as pd
import numpy as np
import ta

DATA_DIR = './data'

# Ticker mapping for yfinance
ASSET_TICKERS = {
    'gold': 'GC=F',
    'silver': 'SI=F',
    'copper': 'HG=F',
    'crude_oil': 'CL=F',
    'natural_gas': 'NG=F',
    'wheat': 'ZW=F'
}

MACRO_TICKERS = {
    'USD_Index': 'DX-Y.NYB',
    'US_10Y_Yield': '^TNX',
    'VIX': '^VIX'
}

def get_live_features(asset_key):
    """
    Fetches 1 year of live data, calculates technicals, merges fast macros, 
    fills slow macros from history, and returns the final feature row perfectly aligned.
    """
    ticker = ASSET_TICKERS.get(asset_key)
    if not ticker:
        raise ValueError(f"No ticker found for {asset_key}")
        
    # 1. Fetch 1 year of asset data
    df_live = yf.download(ticker, period='1y', progress=False)
    if df_live.empty:
        raise ValueError(f"Failed to fetch live data for {ticker}")
        
    if isinstance(df_live.columns, pd.MultiIndex):
        df_live.columns = df_live.columns.droplevel(1)
        
    df_live = df_live[['High', 'Low', 'Close', 'Volume']].copy()
    
    # 2. Compute Technical Features (using exact formulas from training)
    df_live['Return'] = df_live['Close'].pct_change()
    df_live['Log_Return'] = np.log(df_live['Close'] / df_live['Close'].shift(1))
    
    df_live['Ret_Lag_1'] = df_live['Return'].shift(1)
    df_live['Ret_Lag_2'] = df_live['Return'].shift(2)
    df_live['Ret_Lag_3'] = df_live['Return'].shift(3)
    df_live['Ret_Lag_5'] = df_live['Return'].shift(5)
    df_live['Ret_Lag_10'] = df_live['Return'].shift(10)
    
    df_live['Daily_Range_Pct'] = (df_live['High'] - df_live['Low']) / df_live['Close']
    
    df_live['Close_to_MA_5'] = df_live['Close'] / df_live['Close'].rolling(window=5).mean()
    df_live['Close_to_MA_10'] = df_live['Close'] / df_live['Close'].rolling(window=10).mean()
    df_live['Close_to_MA_20'] = df_live['Close'] / df_live['Close'].rolling(window=20).mean()
    df_live['Close_to_MA_50'] = df_live['Close'] / df_live['Close'].rolling(window=50).mean()
    
    df_live['Close_to_EMA_10'] = df_live['Close'] / df_live['Close'].ewm(span=10, adjust=False).mean()
    df_live['Close_to_EMA_20'] = df_live['Close'] / df_live['Close'].ewm(span=20, adjust=False).mean()
    
    df_live['Rolling_Std_10'] = df_live['Return'].rolling(window=10).std()
    df_live['Rolling_Std_20'] = df_live['Return'].rolling(window=20).std()
    
    df_live['RSI_14'] = ta.momentum.RSIIndicator(close=df_live['Close'], window=14).rsi()
    
    macd_indicator = ta.trend.MACD(close=df_live['Close'])
    df_live['MACD_Pct'] = macd_indicator.macd_diff() / df_live['Close']
    
    vol_ma_10 = df_live['Volume'].rolling(window=10).mean()
    df_live['Volume_to_MA_10'] = df_live['Volume'] / (vol_ma_10 + 1e-9)
    
    # 3. Fetch Fast Macros
    for macro_name, macro_ticker in MACRO_TICKERS.items():
        macro_df = yf.download(macro_ticker, period='1y', progress=False)
        if not macro_df.empty:
            if isinstance(macro_df.columns, pd.MultiIndex):
                macro_df.columns = macro_df.columns.droplevel(1)
            df_live[macro_name] = macro_df['Close']
        
    df_live = df_live.ffill() # Forward fill any missing macro days
    
    # 4. Extract target row and columns order from historical CSV
    hist_path = os.path.join(DATA_DIR, f"{asset_key}.csv")
    df_hist = pd.read_csv(hist_path, index_col='Date', parse_dates=True)
    df_hist = df_hist.ffill()
    
    # Expected features based exactly on what the scaler/model expects
    drop_cols = ['Target_Close_Next', 'Target_Return_Next', 'Target_Direction', 
                 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    expected_features = [col for col in df_hist.columns if col not in drop_cols and not col.startswith('Target_')]
    
    # Final live dataframe (last 10 rows for DL sequences)
    df_live_seq = df_live.iloc[-10:].copy()
    hist_row = df_hist.iloc[-1].copy()
    
    # 5. Build the perfectly aligned sequence dataframe
    for col in expected_features:
        if col not in df_live_seq.columns:
            # Fallback to historical (slow macros like CPI, Rig Count, etc)
            df_live_seq[col] = hist_row[col]
            
    df_live_seq = df_live_seq[expected_features]
    
    current_price = df_live['Close'].iloc[-1]
    
    # Return as DataFrame so it can easily be transformed by scaler, and current price
    return df_live_seq, current_price
