import os
import yfinance as yf
import pandas as pd
import numpy as np
import ta

DATA_DIR = './data'

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
    ticker = ASSET_TICKERS.get(asset_key)
    if not ticker:
        raise ValueError(f"No ticker found for {asset_key}")
        
    print(f"Downloading live data for {ticker}...")
    df_live = yf.download(ticker, period='1y', progress=False)
    if isinstance(df_live.columns, pd.MultiIndex):
        df_live.columns = df_live.columns.droplevel(1)
        
    df_live = df_live[['High', 'Low', 'Close', 'Volume']].copy()
    
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
    
    for macro_name, macro_ticker in MACRO_TICKERS.items():
        print(f"Downloading macro: {macro_ticker}...")
        macro_df = yf.download(macro_ticker, period='1y', progress=False)
        if isinstance(macro_df.columns, pd.MultiIndex):
            macro_df.columns = macro_df.columns.droplevel(1)
        df_live[macro_name] = macro_df['Close']
        
    df_live = df_live.ffill() 
    
    hist_path = os.path.join(DATA_DIR, f"{asset_key}.csv")
    df_hist = pd.read_csv(hist_path, index_col='Date', parse_dates=True)
    df_hist = df_hist.ffill()
    
    drop_cols = ['Target_Close_Next', 'Target_Return_Next', 'Target_Direction', 
                 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    expected_features = [col for col in df_hist.columns if col not in drop_cols and not col.startswith('Target_')]
    
    live_row = df_live.iloc[-1].copy()
    hist_row = df_hist.iloc[-1].copy()
    
    final_features = {}
    for col in expected_features:
        if col in live_row:
            final_features[col] = live_row[col]
        else:
            final_features[col] = hist_row[col]
            
    df_final = pd.DataFrame([final_features], columns=expected_features)
    print("\nGenerated Final Live Features:")
    print(df_final.T)
    return df_final

if __name__ == "__main__":
    get_live_features('crude_oil')
