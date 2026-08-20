import os
import yfinance as yf
import pandas as pd
import numpy as np
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# --- Configuration ---
OUTPUT_DIR = "data/stocks/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TICKERS = [
    "ABL", "ABOT", "AGP", "AHCL", "AICL", "AIRLINK", "AKBL", "APL", "ATLH", "ATRL", 
    "BAFL", "BAHL", "BNWM", "BOP", "BWCL", "CHCC", "CNERGY", "COLG", "CPHL", "DGKC", 
    "EFERT", "ENGROH", "FABL", "FATIMA", "FCCL", "FFC", "FFL", "GADT", "GAL", "GHGL", 
    "GHNI", "GLAXO", "HALEON", "HBL", "HCAR", "HINOON", "HMB", "HUBC", "HUMNL", "IBFL", 
    "ILP", "INDU", "INIL", "ISL", "JDWS", "JVDC", "KAPCO", "KEL", "KOHC", "KTML", 
    "LCI", "LOTCHEM", "LUCK", "MARI", "MCB", "MEBL", "MEHT", "MLCF", "MTL", "MUREB", 
    "NATF", "NBP", "NESTLE", "NETSOL", "NML", "OGDC", "PABC", "PAEL", "PAKT", "PGLC", 
    "PIBTL", "PIOC", "PKGS", "POL", "POWER", "PPL", "PSEL", "PSO", "PSX", "PTC", 
    "RMPL", "SAZEW", "SCBPL", "SEARL", "SHFA", "SNGP", "SRVI", "SSGC", "SSOM", "SYS", 
    "TGL", "THALL", "TRG", "UBL", "UNITY", "UPFL", "YOUW"
]

HORIZONS = [7, 14, 28, 42, 60, 90, 120]

def remove_outliers(df):
    """
    PSX has strict daily circuit breakers (historically 5% to 7.5%).
    Any single-day return exceeding +/- 30% is a mathematical anomaly or yfinance data glitch.
    """
    initial_len = len(df)
    daily_return = df['Adj Close'].pct_change()
    
    # Identify Glitches
    glitch_mask = (daily_return > 0.30) | (daily_return < -0.30)
    
    # Drop Glitches
    df = df[~glitch_mask].copy()
    
    removed = initial_len - len(df)
    if removed > 0:
        logger.warning(f"   -> Removed {removed} extreme outlier rows (Data Glitches).")
    
    return df

def run_stage1():
    logger.info(f"Initiating PSX Download for {len(TICKERS)} tickers...")
    
    success_count = 0
    fail_count = 0
    
    for base_ticker in TICKERS:
        ticker = f"{base_ticker}.KA"
        logger.info(f"Downloading {ticker}...")
        
        try:
            # Download 10 years of data
            df = yf.download(ticker, start="2014-01-01", end="2026-08-01", progress=False)
            
            if df.empty:
                logger.error(f"   -> FAILED: No data found for {ticker}")
                fail_count += 1
                continue
                
            # Handle yfinance multi-index columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
                
            # Fallback for international tickers missing Adj Close
            if 'Adj Close' not in df.columns:
                df['Adj Close'] = df['Close']
                
            # Basic Columns
            df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
            
            # --- 1. Null Value Cleaning ---
            # Forward fill small gaps (e.g., trading halts)
            df.ffill(inplace=True)
            # Drop any remaining unfillable NaNs (like missing start data)
            df.dropna(inplace=True)
            
            # --- 2. Outlier Removal ---
            df = remove_outliers(df)
            
            if len(df) < 200:
                logger.warning(f"   -> SKIPPED: {ticker} has less than 200 days of data ({len(df)} rows).")
                fail_count += 1
                continue
                
            # --- 3. Target Engineering ---
            # Using Adj Close to prevent splits/dividends from looking like crashes/spikes
            for h in HORIZONS:
                target_col = f'Target_{h}d'
                df[target_col] = (df['Adj Close'].shift(-h) - df['Adj Close']) / df['Adj Close']
                
            # Save to disk
            output_path = os.path.join(OUTPUT_DIR, f"{base_ticker}.csv")
            df.to_csv(output_path)
            success_count += 1
            
        except Exception as e:
            logger.error(f"   -> CRITICAL ERROR on {ticker}: {str(e)}")
            fail_count += 1
            
    logger.info("=" * 50)
    logger.info(f"Stage 1 Complete! Successfully downloaded & cleaned {success_count} stocks.")
    if fail_count > 0:
        logger.warning(f"Failed / Skipped: {fail_count} stocks.")
    logger.info("=" * 50)

if __name__ == "__main__":
    run_stage1()
