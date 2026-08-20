import pandas as pd
import numpy as np
import os

data_dir = './data'

# Configuration mapping to handle the inconsistencies in the group project files
MACRO_FILES = {
    'gold': {'file': 'gold-macro.csv', 'dayfirst': False},
    'silver': {'file': 'silver-macro.csv', 'dayfirst': False},
    'copper': {'file': 'copper-macro.xlsx', 'dayfirst': False},
    'natural_gas': {'file': 'natural gas-macro.csv', 'dayfirst': True},
    'crude_oil': {'file': 'crude-macro.xlsx', 'dayfirst': False},
    'wheat': {'file': 'wheat-macro.csv', 'dayfirst': False}
}

def main():
    for comm, config in MACRO_FILES.items():
        base_path = os.path.join(data_dir, f"{comm}.csv")
        macro_path = os.path.join(data_dir, config['file'])
        
        if not os.path.exists(base_path):
            print(f"[{comm.upper()}] Base OHLCV file not found, skipping...")
            continue
            
        if not os.path.exists(macro_path):
            print(f"[{comm.upper()}] Macro file not found at {macro_path}, skipping...")
            continue
            
        print(f"\n--- Merging Macro Data for {comm.upper()} ---")
        
        # 1. Load Base Feature Table (OHLCV + Technicals)
        df_base = pd.read_csv(base_path)
        df_base['Date'] = pd.to_datetime(df_base['Date'])
        df_base.set_index('Date', inplace=True)
        df_base.sort_index(inplace=True)
        initial_rows = len(df_base)
        
        # 2. Load Macro Data
        if macro_path.endswith('.csv'):
            df_macro = pd.read_csv(macro_path)
        else:
            df_macro = pd.read_excel(macro_path)
            
        # 3. Clean Macro Dates and Column Names
        if 'date' in df_macro.columns:
            df_macro.rename(columns={'date': 'Date'}, inplace=True)
            
        # Normalize column names from Excel files to match the expected schema
        rename_map = {
            'USD_Index_DXY': 'USD_Index',
            'cpi_inflation_yoy': 'CPI_Inflation',
            'cpi_index': 'CPI_Index'
        }
        df_macro.rename(columns=rename_map, inplace=True)
        
        # Drop redundant/metadata columns if they exist
        drop_redundant = ['Copper_Price', 'Crude_Oil_Price', 'ingestion_time_utc', 'CPI_Index']
        df_macro.drop(columns=[col for col in drop_redundant if col in df_macro.columns], inplace=True)
            
        df_macro['Date'] = pd.to_datetime(df_macro['Date'], dayfirst=config['dayfirst'], errors='coerce')
        df_macro.dropna(subset=['Date'], inplace=True) # Drop rows with invalid dates
        df_macro.set_index('Date', inplace=True)
        df_macro.sort_index(inplace=True)
        
        # Drop duplicates if multiple rows have the same date
        df_macro = df_macro[~df_macro.index.duplicated(keep='last')]
        
        # 4. Left Join Macro onto Base (Aligned by Date)
        # Drop any intersecting columns from base (in case the script is run multiple times)
        overlapping_cols = df_base.columns.intersection(df_macro.columns)
        if len(overlapping_cols) > 0:
            df_base.drop(columns=overlapping_cols, inplace=True)
            
        df_merged = df_base.join(df_macro, how='left')
        
        # 5. Forward Fill (ffill) the Macro Data
        # This is CRITICAL. Monthly data (like CPI) is only reported on one day.
        # We must carry that value forward for every subsequent day until the next release.
        # We only ffill the macro columns, not the base columns.
        macro_cols = df_macro.columns.tolist()
        df_merged[macro_cols] = df_merged[macro_cols].ffill()
        
        # 6. Drop Initial NaNs (Look-ahead bias prevention)
        # The base OHLCV data likely starts in 2000, but macro data starts in ~2014.
        # We cannot backfill (that's look-ahead bias), so we must drop the early years.
        df_merged.dropna(inplace=True)
        
        final_rows = len(df_merged)
        
        print(f"Base rows: {initial_rows}")
        print(f"Macro features added: {len(macro_cols)}")
        print(f"Rows dropped (due to missing historical macro data): {initial_rows - final_rows}")
        print(f"Final shape ready for training: {df_merged.shape}")
        
        # 7. Overwrite the base file with the fully merged table
        df_merged.to_csv(base_path)
        print(f"Successfully saved {base_path}")

if __name__ == "__main__":
    main()
