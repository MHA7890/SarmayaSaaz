import os
import glob
import pandas as pd
import numpy as np
import ta
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

DATA_DIR = "data/mufap"

def engineer_features():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    logger.info(f"Starting Stage 2 Feature Engineering for {len(files)} mutual funds...")
    
    funds_processed = 0
    funds_dropped = 0
    
    for file_path in files:
        df = pd.read_csv(file_path)
        
        # Ensure chronological order and proper index
        if 'Validity_Date' in df.columns:
            df['Validity_Date'] = pd.to_datetime(df['Validity_Date'])
            df = df.sort_values('Validity_Date')
            df.set_index('Validity_Date', inplace=True)
            
        # STEP 1: The Age Filter
        if len(df) < 200:
            logger.debug(f"Dropping {os.path.basename(file_path)}: Only {len(df)} rows (<200 days).")
            os.remove(file_path)
            funds_dropped += 1
            continue
            
        # Keep base columns for tracking if needed
        # Expected columns from Stage 1: NAV, Offer, Repurchase, Spread_Ratio, Target_7d...
        
        # STEP 2: Stationary Returns
        df['Daily_Return'] = df['NAV'].pct_change()
        df['Log_Return'] = np.log(df['NAV'] / df['NAV'].shift(1))
        
        df['Ret_Lag_1'] = df['Log_Return'].shift(1)
        df['Ret_Lag_2'] = df['Log_Return'].shift(2)
        df['Ret_Lag_3'] = df['Log_Return'].shift(3)
        df['Ret_Lag_5'] = df['Log_Return'].shift(5)
        df['Ret_Lag_10'] = df['Log_Return'].shift(10)
        
        # STEP 3: Relative Moving Averages
        df['NAV_to_SMA_20'] = df['NAV'] / df['NAV'].rolling(window=20).mean()
        df['NAV_to_SMA_50'] = df['NAV'] / df['NAV'].rolling(window=50).mean()
        df['NAV_to_SMA_200'] = df['NAV'] / df['NAV'].rolling(window=200).mean()
        df['NAV_to_EMA_20'] = df['NAV'] / df['NAV'].ewm(span=20, adjust=False).mean()
        
        # STEP 4: Technical Momentum Indicators
        # RSI 14
        df['RSI_14'] = ta.momentum.RSIIndicator(close=df['NAV'], window=14).rsi()
        
        # MACD Percentage
        macd = ta.trend.MACD(close=df['NAV'])
        df['MACD_Pct'] = macd.macd_diff() / df['NAV']
        
        # Volatility
        df['Rolling_Std_10'] = df['Log_Return'].rolling(window=10).std()
        df['Rolling_Std_30'] = df['Log_Return'].rolling(window=30).std()
        
        # STEP 7: NaN Eradication (Before dropping NAV so we safely target the 200 MA tail)
        # We must drop rows where SMA_200 is NaN (the first 199 rows)
        # We do NOT drop based on Target columns (which are NaN at the very end of the dataset)
        df.dropna(subset=['NAV_to_SMA_200', 'Rolling_Std_30', 'RSI_14', 'MACD_Pct', 'Ret_Lag_10'], inplace=True)
        
        # STEP 6: Purging Absolute Data (Anti-Leakage)
        cols_to_drop = ['NAV', 'Offer', 'Repurchase', 'Front_End_Load', 'Back_End_Load', 'Contingent_Load', 'FundType']
        # Also drop Sector, Category, CategoryID, Fund string if they are still in the columns as they are static metadata
        metadata_cols = ['Sector', 'Category', 'CategoryID', 'Fund', 'AMC']
        cols_to_drop.extend(metadata_cols)
        
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True, errors='ignore')
        
        # Save finalized feature matrix
        df.to_csv(file_path)
        funds_processed += 1
        
    logger.info(f"✅ Stage 2 Complete!")
    logger.info(f"Funds Processed Successfully: {funds_processed}")
    logger.info(f"Funds Destroyed (Age Filter < 200 Days): {funds_dropped}")

if __name__ == "__main__":
    engineer_features()
