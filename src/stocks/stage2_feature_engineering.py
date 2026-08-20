import os
import glob
import pandas as pd
import numpy as np
import ta
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# --- Configuration ---
INPUT_DIR = "data/stocks/raw"
OUTPUT_DIR = "data/stocks/features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def engineer_features(df):
    """
    Applies mathematical transformations to raw stock data to make it stationary and predictive.
    """
    # 1. Trend & Momentum (Using Adj Close to account for splits/dividends)
    df['SMA_20'] = ta.trend.sma_indicator(df['Adj Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Adj Close'], window=50)
    df['SMA_200'] = ta.trend.sma_indicator(df['Adj Close'], window=200)
    
    # Ratios (Stationary)
    df['Close_to_SMA_20'] = df['Adj Close'] / df['SMA_20']
    df['Close_to_SMA_50'] = df['Adj Close'] / df['SMA_50']
    df['Close_to_SMA_200'] = df['Adj Close'] / df['SMA_200']
    
    # RSI & MACD
    df['RSI_14'] = ta.momentum.rsi(df['Adj Close'], window=14)
    macd = ta.trend.MACD(df['Adj Close'])
    df['MACD_Diff'] = macd.macd_diff()
    
    # Log Returns
    for days in [1, 3, 7, 14]:
        df[f'Log_Return_{days}d'] = np.log(df['Adj Close'] / df['Adj Close'].shift(days))
        
    # 2. Volatility
    bollinger = ta.volatility.BollingerBands(df['Adj Close'], window=20, window_dev=2)
    df['BB_Width'] = (bollinger.bollinger_hband() - bollinger.bollinger_lband()) / df['SMA_20']
    
    atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Adj Close'], window=14)
    df['ATR_Normalized'] = atr.average_true_range() / df['Adj Close']
    
    df['Volatility_30d'] = df['Log_Return_1d'].rolling(window=30).std()
    
    # 3. Volume Profiling (The Stock Secret)
    df['SMA_Volume_20'] = df['Volume'].rolling(window=20).mean()
    # Replace 0s with 1s to prevent division by zero in illiquid stocks
    safe_sma_volume = df['SMA_Volume_20'].replace(0, 1)
    df['Volume_Surge'] = df['Volume'] / safe_sma_volume
    
    # On-Balance Volume (OBV) - We must normalize it because raw OBV goes into millions
    obv = ta.volume.OnBalanceVolumeIndicator(df['Adj Close'], df['Volume']).on_balance_volume()
    # Normalize OBV by creating an OBV SMA Ratio
    obv_sma = obv.rolling(window=20).mean().replace(0, 1)
    df['OBV_Momentum'] = obv / obv_sma

    # Clean up intermediate absolute columns to prevent leakage
    drop_cols = [
        'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
        'SMA_20', 'SMA_50', 'SMA_200', 'SMA_Volume_20'
    ]
    df.drop(columns=drop_cols, inplace=True)
    
    return df

def run_stage2():
    files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    logger.info(f"Initiating Stage 2 Feature Engineering for {len(files)} stocks...")
    
    success_count = 0
    
    for file_path in files:
        ticker = os.path.basename(file_path)
        
        try:
            df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            
            # Apply Engineering
            df = engineer_features(df)
            
            # Drop the first 200 rows (NaNs from the SMA_200 calculation)
            # Actually, to be safe, dropna on features (keeping targets with potential NaNs at the end)
            feature_cols = [c for c in df.columns if not c.startswith('Target_')]
            df.dropna(subset=feature_cols, inplace=True)
            
            if len(df) < 50:
                logger.warning(f"   -> SKIPPED: {ticker} has less than 50 days of data after SMA_200 purge.")
                continue
                
            # Save Output
            output_path = os.path.join(OUTPUT_DIR, ticker)
            df.to_csv(output_path)
            success_count += 1
            
        except Exception as e:
            logger.error(f"   -> CRITICAL ERROR on {ticker}: {str(e)}")
            
    logger.info("=" * 50)
    logger.info(f"Stage 2 Complete! Successfully engineered {success_count} isolated, leakage-free matrices.")
    logger.info("=" * 50)

if __name__ == "__main__":
    run_stage2()
