import os
import glob
import pandas as pd
import numpy as np
import yfinance as yf
import logging
import ta

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# --- Configuration ---
INPUT_DIR = "data/stocks/features"
OUTPUT_BASE_DIR = "data/stocks/clustered"
START_DATE = "2014-01-01"
END_DATE = "2026-08-01"

# --- 7 Super-Sectors Mapping ---
SECTOR_MAP = {
    'Financials': ['ABL', 'AHCL', 'AICL', 'AKBL', 'BAFL', 'BAHL', 'BOP', 'FABL', 'HBL', 'HMB', 'MCB', 'MEBL', 'NBP', 'PSX', 'SCBPL', 'UBL'],
    'Energy_Power': ['APL', 'ATRL', 'CNERGY', 'HUBC', 'KAPCO', 'KEL', 'MARI', 'OGDC', 'POL', 'PPL', 'PSO', 'SNGP', 'SSGC'],
    'Cement_Construction': ['BWCL', 'CHCC', 'DGKC', 'FCCL', 'KOHC', 'LUCK', 'MLCF', 'PIOC', 'POWER', 'THALL', 'ISL', 'INIL'],
    'Tech_Telecom': ['AIRLINK', 'NETSOL', 'PTC', 'SYS', 'TRG'],
    'Pharmaceuticals': ['ABOT', 'AGP', 'CPHL', 'GLAXO', 'HALEON', 'HINOON', 'IBFL', 'SEARL', 'SHFA'],
    'Fertilizers_Chemicals': ['COLG', 'EFERT', 'ENGROH', 'FATIMA', 'FFC', 'GHGL', 'LCI', 'LOTCHEM', 'PABC', 'EPCL'],
    'Consumer_Autos': ['ATLH', 'BNWM', 'FFL', 'GADT', 'GAL', 'GHNI', 'HCAR', 'HUMNL', 'ILP', 'INDU', 'JDWS', 'JVDC', 'KTML', 'MEHT', 'MTL', 'MUREB', 'NATF', 'NESTLE', 'NML', 'PAEL', 'PAKT', 'PGLC', 'PIBTL', 'PKGS', 'PSEL', 'RMPL', 'SAZEW', 'SRVI', 'SSOM', 'TGL', 'UNITY', 'UPFL', 'YOUW']
}

# Reverse mapping for O(1) lookup
TICKER_TO_SECTOR = {}
for sector, tickers in SECTOR_MAP.items():
    for ticker in tickers:
        TICKER_TO_SECTOR[ticker] = sector

# --- Macro Engineering Function ---
def download_and_engineer_macro(ticker_symbol, prefix):
    logger.info(f"Downloading Global Macro Proxy: {ticker_symbol}...")
    df = yf.download(ticker_symbol, start=START_DATE, end=END_DATE, progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']
        
    # Engineer anti-leakage features
    df[f'{prefix}_Log_Return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    
    sma_200 = ta.trend.sma_indicator(df['Adj Close'], window=200)
    df[f'{prefix}_SMA_200_Ratio'] = df['Adj Close'] / sma_200
    
    df[f'{prefix}_Volatility_30d'] = df[f'{prefix}_Log_Return'].rolling(window=30).std()
    
    # Keep only the engineered features
    features = [f'{prefix}_Log_Return', f'{prefix}_SMA_200_Ratio', f'{prefix}_Volatility_30d']
    return df[features]

def run_stage3():
    # 1. Download and Engineer Macros
    df_pkr = download_and_engineer_macro("PKR=X", "Currency")
    df_oil = download_and_engineer_macro("CL=F", "Oil")
    df_nasdaq = download_and_engineer_macro("^IXIC", "Tech")
    
    # 2. Iterate through all processed stocks
    files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    logger.info(f"Initiating Stage 3 Macro Merging for {len(files)} stocks...")
    
    success_count = 0
    
    for file_path in files:
        filename = os.path.basename(file_path) # e.g., 'SYS.csv'
        base_ticker = filename.replace('.csv', '')
        
        # Identify Sector
        sector = TICKER_TO_SECTOR.get(base_ticker, "Diversified")
        
        # Load the stock data
        stock_df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        initial_rows = len(stock_df)
        
        # --- The Merging Logic ---
        # 1. Everyone gets Currency (PKR=X) because devaluation affects all PSX stocks
        stock_df = stock_df.join(df_pkr, how='left')
        
        # 2. Targeted Sector Injection
        if sector == "Energy_Power":
            stock_df = stock_df.join(df_oil, how='left')
        elif sector == "Tech_Telecom":
            stock_df = stock_df.join(df_nasdaq, how='left')
            
        # 3. Handle Mismatched Holidays using Forward Fill
        # (e.g. US markets open on Pakistani Eid -> data exists in macro but not locally. 
        # But since we use left join on the PSX calendar, we only care about PSX trading days.
        # If PSX is open but US is closed (e.g. July 4), the left join puts NaN. We ffill() it.)
        stock_df.ffill(inplace=True)
        
        # Drop the initial rows where the Macro SMA_200 caused NaNs
        stock_df.dropna(subset=['Currency_SMA_200_Ratio'], inplace=True)
        
        # Ensure directory exists
        sector_dir = os.path.join(OUTPUT_BASE_DIR, sector)
        os.makedirs(sector_dir, exist_ok=True)
        
        # Save
        out_path = os.path.join(sector_dir, filename)
        stock_df.to_csv(out_path)
        success_count += 1
        
    logger.info("=" * 50)
    logger.info(f"Stage 3 Complete! Successfully clustered and macro-injected {success_count} stocks across 7 Sectors.")
    logger.info("=" * 50)

if __name__ == "__main__":
    run_stage3()
