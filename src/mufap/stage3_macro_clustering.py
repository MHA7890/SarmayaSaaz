import os
import glob
import pandas as pd
import numpy as np
import yfinance as yf
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

RAW_FILE = "MUFAP_Historical_NAV.csv"
INPUT_DIR = "data/mufap"
OUTPUT_DIR = "data/mufap_clustered"

CLUSTERS = ['Equity', 'MoneyMarket', 'Income', 'Commodity', 'Balanced']

for c in CLUSTERS:
    os.makedirs(os.path.join(OUTPUT_DIR, c), exist_ok=True)

def fetch_and_process_macro(ticker, prefix):
    logger.info(f"Downloading {ticker} macro data...")
    df = yf.download(ticker, start="2015-01-01", end="2027-01-01", progress=False)
    
    # Handle multi-level columns if present in newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    df = df[['Close']].copy()
    
    # Engineer Macro Features
    df[f'{prefix}_Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df[f'{prefix}_SMA_200_Ratio'] = df['Close'] / df['Close'].rolling(window=200).mean()
    df[f'{prefix}_Volatility_30d'] = df[f'{prefix}_Log_Return'].rolling(window=30).std()
    
    # Clean up raw price
    df = df.drop(columns=['Close'])
    
    # Make sure index is timezone naive for merging
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
        
    return df

def run_stage3():
    # 1. Build Routing Map
    logger.info("Building Fund to Category routing map...")
    raw_df = pd.read_csv(RAW_FILE)
    fund_to_cat = dict(zip(raw_df['Fund'].dropna(), raw_df['Category'].dropna()))
    
    # 2. Fetch Macro Data
    macro_equity = fetch_and_process_macro('PAK', 'PAK')
    macro_income = fetch_and_process_macro('PKR=X', 'PKR')
    macro_commodity = fetch_and_process_macro('GC=F', 'GC')
    
    # 3. Process each fund
    files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    logger.info(f"Processing {len(files)} mature funds...")
    
    cluster_counts = {c: 0 for c in CLUSTERS}
    
    for file_path in files:
        # Extract original fund name
        basename = os.path.basename(file_path).replace('.csv', '')
        # Convert underscores back to slash if we had replaced it (though usually it's direct match, let's look up roughly)
        
        # Exact match or find the closest fund
        fund_name = None
        for k in fund_to_cat.keys():
            safe_k = str(k).replace('/', '_').replace('\\', '_')
            if safe_k == basename:
                fund_name = k
                break
                
        if not fund_name:
            logger.warning(f"Could not map {basename} to a category. Defaulting to Balanced.")
            category = "Balanced"
        else:
            category = fund_to_cat[fund_name]
            
        # Determine Cluster
        cat_lower = str(category).lower()
        if 'equity' in cat_lower or 'index' in cat_lower or 'etf' in cat_lower:
            cluster = 'Equity'
            macro_df = macro_equity
        elif 'money market' in cat_lower:
            cluster = 'MoneyMarket'
            macro_df = macro_income
        elif 'income' in cat_lower or 'fixed' in cat_lower or 'debt' in cat_lower:
            cluster = 'Income'
            macro_df = macro_income
        elif 'commodit' in cat_lower or 'gold' in cat_lower:
            cluster = 'Commodity'
            macro_df = macro_commodity
        else:
            cluster = 'Balanced' # Asset Allocation, Capital Protected, etc.
            # Blend both equity and income macros for balanced funds
            macro_df = pd.merge(macro_equity, macro_income, left_index=True, right_index=True, how='outer')
            
        # 4. Load Fund Data
        df = pd.read_csv(file_path, index_col='Validity_Date', parse_dates=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        # 5. Merge Macro Data
        # Left merge keeps all mutual fund dates
        merged_df = df.join(macro_df, how='left')
        
        # 6. Forward Fill Weekend Gaps
        merged_df.ffill(inplace=True)
        
        # 7. Drop rows where Macro Features are NaN (the first 200 days of the macro data)
        # We only drop rows if the actual engineered macro features are NaN
        macro_cols = macro_df.columns.tolist()
        merged_df.dropna(subset=macro_cols, inplace=True)
        
        # 8. Save
        output_path = os.path.join(OUTPUT_DIR, cluster, os.path.basename(file_path))
        merged_df.to_csv(output_path)
        cluster_counts[cluster] += 1
        
    logger.info("✅ Stage 3 Complete! Final Clustering Distribution:")
    for c, count in cluster_counts.items():
        logger.info(f"   - {c}: {count} funds")

if __name__ == "__main__":
    run_stage3()
