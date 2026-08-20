import yfinance as yf
import pandas as pd
import os
import numpy as np

# Define commodities and tickers based on the table in project.md
commodities = {
    'gold': 'GC=F',
    'silver': 'SI=F',
    'copper': 'HG=F',
    'natural_gas': 'NG=F',
    'crude_oil': 'CL=F',
    'wheat': 'ZW=F'
}

# Date range: ~10 years
start_date = '2014-01-01'
end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

data_dir = './data'
os.makedirs(data_dir, exist_ok=True)

def clean_and_align_data(df, name):
    if df.empty:
        print(f"Warning: No data for {name}")
        return df
        
    df.index = pd.to_datetime(df.index)
    
    # Remove timezone if any to avoid issues when reindexing
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Reindex to all business days in the range to ensure no gaps
    all_bussiness_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
    df = df.reindex(all_bussiness_days)
    
    # Forward fill missing values (e.g., holidays)
    df.ffill(inplace=True)
    
    # Drop any remaining NaNs (e.g., if the first day was a holiday)
    df.dropna(inplace=True)
    
    # Explicitly handle negative price outliers (like Crude Oil in April 2020)
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
    
    # Check for negatives
    negatives = (df[numeric_cols] < 0).any(axis=1)
    if negatives.any():
        print(f"[{name}] Detected negative prices! Resolving by capping at 0.01.")
        # Apply capping
        for col in numeric_cols:
            df.loc[df[col] < 0, col] = 0.01
            
    # Also handle Volume: shouldn't be negative
    if (df['Volume'] < 0).any():
         print(f"[{name}] Detected negative volume! Setting to 0.")
         df.loc[df['Volume'] < 0, 'Volume'] = 0
         
    # Make sure index is named Date
    df.index.name = 'Date'
    
    return df

def main():
    for name, ticker in commodities.items():
        print(f"Downloading data for {name} ({ticker})...")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # Flatten MultiIndex columns if present (yfinance sometimes returns multiindex)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Select the desired columns
        columns_order = ['Open', 'High', 'Low', 'Close', 'Volume']
        if 'Adj Close' in df.columns:
            columns_order.insert(4, 'Adj Close')
        else:
            # If Adj Close is missing, duplicate Close to satisfy OHLCV with Adj Close requirement
            df['Adj Close'] = df['Close']
            columns_order.insert(4, 'Adj Close')

        if not all(col in df.columns for col in columns_order):
            print(f"Error: Missing columns in {name}. Available columns: {df.columns}")
            continue
            
        df = df[columns_order]
        
        # Clean and align data
        df_cleaned = clean_and_align_data(df, name)
        
        # Save to CSV in data directory
        output_path = os.path.join(data_dir, f"{name}.csv")
        df_cleaned.to_csv(output_path)
        print(f"Saved {name}.csv to {output_path} (Shape: {df_cleaned.shape})")
        print("-" * 50)

if __name__ == "__main__":
    main()
