import os
import pandas as pd
import numpy as np
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# --- Configuration ---
INPUT_FILE = "data/mufap/raw/MUFAP_Historical_NAV.csv"
OUTPUT_DIR = "data/mufap"
HORIZONS = [7, 14, 28, 42, 60, 90, 120]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_and_process():
    logger.info(f"Loading dataset: {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    initial_rows = len(df)
    
    # 1. Missing Values (Drop missing Validity_Date)
    df = df.dropna(subset=['Validity_Date'])
    
    # Convert Validity_Date to datetime and set as index
    df['Validity_Date'] = pd.to_datetime(df['Validity_Date'], errors='coerce')
    df = df.dropna(subset=['Validity_Date']) # Drop any rows that failed to parse
    
    # 2. Critical Outliers
    # Drop impossible NAVs
    df = df[df['NAV'] > 0]
    
    # Drop impossible Offer/Repurchase prices (Glitch filtering)
    # Offer price should never be more than 10% higher than NAV (max load is ~6%)
    df = df[(df['Offer'] > 0) & (df['Offer'] <= df['NAV'] * 1.2)]
    df = df[(df['Repurchase'] > 0) & (df['Repurchase'] <= df['NAV'] * 1.2)]
    
    # 3. Columns to Remove
    cols_to_drop = ['Trustee', 'Scrape_Time', 'FundID', 'AMC', 'Market_Price']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    # Sort chronologically before splitting
    df = df.sort_values(by=['Fund', 'Validity_Date'])
    
    # 4. Process each fund individually to calculate targets and features
    funds = df['Fund'].dropna().unique()
    logger.info(f"Purged {initial_rows - len(df)} bad rows. Splitting into {len(funds)} clean assets...")
    
    success_count = 0
    
    for fund_name in funds:
        fund_df = df[df['Fund'] == fund_name].copy()
        
        # Ensure chronological order
        fund_df = fund_df.sort_values('Validity_Date')
        fund_df.set_index('Validity_Date', inplace=True)
        
        # Add Spread Ratio
        fund_df['Spread_Ratio'] = (fund_df['Offer'] - fund_df['NAV']) / fund_df['NAV']
        
        # Handle division by zero or infinite values in spread
        fund_df.replace([np.inf, -np.inf], 0, inplace=True)
        
        # Add Target Columns (Future Returns)
        for h in HORIZONS:
            # We predict the percentage return of the NAV 'h' days into the future
            fund_df[f'Target_{h}d'] = (fund_df['NAV'].shift(-h) - fund_df['NAV']) / fund_df['NAV']
            
        # Drop rows where target couldn't be calculated (the tail of the dataset)
        # However, for pure feature generation we will keep them as NaN and drop during training
        
        # Clean the filename string
        safe_fund_name = str(fund_name).replace('/', '_').replace('\\', '_')
        output_path = os.path.join(OUTPUT_DIR, f"{safe_fund_name}.csv")
        
        fund_df.to_csv(output_path)
        success_count += 1
        
    logger.info(f"✅ Phase 1 Complete! Successfully saved {success_count} clean mutual fund files to '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    clean_and_process()
