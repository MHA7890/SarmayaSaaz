import pandas as pd
import numpy as np
import os
import ta

data_dir = './data'
commodities = ['gold', 'silver', 'copper', 'natural_gas', 'crude_oil', 'wheat']

def engineer_features(df):
    # Ensure data is sorted by index just in case
    df = df.sort_index()
    
    # ---------------------------
    # Price-based Features (Stationary)
    # ---------------------------
    df['Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Lags of RETURNS (instead of absolute prices)
    df['Ret_Lag_1'] = df['Return'].shift(1)
    df['Ret_Lag_2'] = df['Return'].shift(2)
    df['Ret_Lag_3'] = df['Return'].shift(3)
    df['Ret_Lag_5'] = df['Return'].shift(5)
    df['Ret_Lag_10'] = df['Return'].shift(10)
    
    # Daily range as percentage of Close
    df['Daily_Range_Pct'] = (df['High'] - df['Low']) / df['Close']
    
    # ---------------------------
    # Rolling / Moving Averages (Relative)
    # ---------------------------
    # Calculate ratio of Close price to Moving Average instead of raw absolute MA
    df['Close_to_MA_5'] = df['Close'] / df['Close'].rolling(window=5).mean()
    df['Close_to_MA_10'] = df['Close'] / df['Close'].rolling(window=10).mean()
    df['Close_to_MA_20'] = df['Close'] / df['Close'].rolling(window=20).mean()
    df['Close_to_MA_50'] = df['Close'] / df['Close'].rolling(window=50).mean()
    
    df['Close_to_EMA_10'] = df['Close'] / df['Close'].ewm(span=10, adjust=False).mean()
    df['Close_to_EMA_20'] = df['Close'] / df['Close'].ewm(span=20, adjust=False).mean()
    
    # Rolling standard deviation of RETURNS (already stationary)
    df['Rolling_Std_10'] = df['Return'].rolling(window=10).std()
    df['Rolling_Std_20'] = df['Return'].rolling(window=20).std()
    
    # ---------------------------
    # Momentum / Technical Indicators (Stationary)
    # ---------------------------
    # RSI 14 (already stationary: bounded 0-100)
    df['RSI_14'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    
    # MACD Percentage (MACD difference normalized by Close price)
    macd_indicator = ta.trend.MACD(close=df['Close'])
    df['MACD_Pct'] = macd_indicator.macd_diff() / df['Close']
    
    # Volume normalized by its 10-day moving average
    vol_ma_10 = df['Volume'].rolling(window=10).mean()
    df['Volume_to_MA_10'] = df['Volume'] / (vol_ma_10 + 1e-9) # add epsilon to avoid div by zero
    
    # ---------------------------
    # Target Columns (What we predict)
    # ---------------------------
    df['Target_Close_Next'] = df['Close'].shift(-1)
    df['Target_Return_Next'] = df['Return'].shift(-1)
    
    df['Target_Direction'] = (df['Target_Return_Next'] > 0).astype(int)
    # Correct the last row where Target_Return_Next is NaN
    df.loc[df['Target_Return_Next'].isna(), 'Target_Direction'] = np.nan
    
    return df

def main():
    # Since we are modifying the base features, we should start fresh from the stage 0 raw data
    # But since stage 0 data is overwritten, we just run on the existing CSVs which still have 
    # Open, High, Low, Close, Adj Close, Volume columns intact.
    for name in commodities:
        file_path = os.path.join(data_dir, f"{name}.csv")
        if not os.path.exists(file_path):
            print(f"Skipping {name}: Data file not found.")
            continue
            
        print(f"Processing {name}...")
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        
        # Keep only the base raw columns in case this script is run multiple times
        base_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        df = df[base_cols]
        
        # Engineer the new stationary features
        df = engineer_features(df)
        
        # Check rows before drop
        initial_rows = len(df)
        
        # Drop rows with NaN (due to rolling windows, lags, and shift(-1) target)
        df.dropna(inplace=True)
        
        final_rows = len(df)
        print(f"Dropped {initial_rows - final_rows} rows due to NaN. Final shape: {df.shape}")
        
        # Save back to CSV, overwriting with new features
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        df.to_csv(file_path)
        print(f"Saved {name}.csv with completely stationary technical features.")
        print("-" * 50)

if __name__ == "__main__":
    main()
