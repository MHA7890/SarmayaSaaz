import os
import pandas as pd
import numpy as np

INPUT_DIR = "data/crypto"
OUTPUT_DIR = "data/crypto_processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HORIZONS = [7, 14, 28, 42, 60, 90, 120]

def process_file(filepath, filename):
    try:
        df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        
        # 1. Target Engineering (Forward-Looking Returns)
        for h in HORIZONS:
            df[f'Target_{h}d'] = (df['Close'].shift(-h) - df['Close']) / df['Close']
            
        # 2. Stationarity Fix (Convert Absolute Swing Prices to Ratios)
        # We replace 0 values in the denominator to avoid division by zero (though extremely rare for prices)
        df['Last_Swing_High_Price'] = df['Last_Swing_High_Price'].replace(0, np.nan)
        df['Last_Swing_Low_Price'] = df['Last_Swing_Low_Price'].replace(0, np.nan)
        
        df['Dist_to_Swing_High'] = (df['Close'] - df['Last_Swing_High_Price']) / df['Last_Swing_High_Price']
        df['Dist_to_Swing_Low'] = (df['Close'] - df['Last_Swing_Low_Price']) / df['Last_Swing_Low_Price']
        
        # Fill missing initial distances with 0 or drop them. 
        # Since we want a completely clean dataframe, we will let dropna() handle it later.
        
        # 3. Absolute Price Purge
        columns_to_drop = ['Close', 'Volume', 'Last_Swing_High_Price', 'Last_Swing_Low_Price']
        df = df.drop(columns=columns_to_drop, errors='ignore')
        
        # Handle any possible infinite values caused by extreme memecoin math
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # 4. The Sacrifice (Drop rows where Targets are NaN, plus any indicator NaNs)
        df = df.dropna()
        
        # Save processed data
        output_path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(output_path)
        
        return True, len(df)
        
    except Exception as e:
        return False, str(e)

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Directory {INPUT_DIR} not found.")
        return
        
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
    print(f"Found {len(files)} files in {INPUT_DIR}. Starting Phase 2 Preprocessing...")
    
    success_count = 0
    total_rows_saved = 0
    
    for f in files:
        filepath = os.path.join(INPUT_DIR, f)
        success, result = process_file(filepath, f)
        
        if success:
            ticker = f.replace('.csv', '')
            print(f"  -> Processed {ticker} successfully. Rows: {result}")
            success_count += 1
            total_rows_saved += result
        else:
            print(f"  -> Failed to process {f}: {result}")
            
    print(f"\\nPhase 2 Complete! Successfully processed {success_count}/{len(files)} assets.")
    print(f"Total clean, stationary, predictive rows generated: {total_rows_saved:,}")

if __name__ == "__main__":
    main()
