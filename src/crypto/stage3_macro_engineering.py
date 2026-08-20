import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import warnings
warnings.filterwarnings('ignore')

INPUT_DIR = "data/crypto_processed"

def get_fear_and_greed():
    print("Fetching Fear & Greed Index from alternative.me API...")
    url = "https://api.alternative.me/fng/?limit=0"
    response = requests.get(url)
    data = response.json()['data']
    
    df = pd.DataFrame(data)
    # Convert Unix timestamp to localized Date
    df['Date'] = pd.to_datetime(df['timestamp'], unit='s').dt.tz_localize(None).dt.normalize()
    df['Fear_Greed_Score'] = df['value'].astype(float)
    
    return df[['Date', 'Fear_Greed_Score']].set_index('Date').sort_index()

def get_sp500():
    print("Fetching S&P 500 Data...")
    sp500 = yf.download("^GSPC", period="10y", interval="1d", progress=False)
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = [col[0] for col in sp500.columns]
    sp500.index = sp500.index.tz_localize(None).normalize()
    
    sp500['SP500_Return_7d'] = np.log(sp500['Close'] / sp500['Close'].shift(7))
    return sp500[['SP500_Return_7d']]

def get_btc_volatility():
    print("Fetching BTC Volatility Data...")
    btc = yf.download("BTC-USD", period="10y", interval="1d", progress=False)
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = [col[0] for col in btc.columns]
    btc.index = btc.index.tz_localize(None).normalize()
    
    btc['Log_Return_1d'] = np.log(btc['Close'] / btc['Close'].shift(1))
    btc['BTC_Volatility_30d'] = btc['Log_Return_1d'].rolling(window=30).std() * np.sqrt(365)
    return btc[['BTC_Volatility_30d']]

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: {INPUT_DIR} does not exist.")
        return
        
    fng_df = get_fear_and_greed()
    sp500_df = get_sp500()
    btc_vol_df = get_btc_volatility()
    
    # Combine macro features
    macro_df = pd.concat([fng_df, sp500_df, btc_vol_df], axis=1)
    
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
    print(f"\\nMerging Macro Features into {len(files)} assets...")
    
    success_count = 0
    total_rows = 0
    
    for f in files:
        try:
            filepath = os.path.join(INPUT_DIR, f)
            df = pd.read_csv(filepath)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            
            # Left join macro features to existing asset dates
            df = df.join(macro_df, how='left')
            
            # 1. Forward fill S&P 500 over weekends
            df['SP500_Return_7d'] = df['SP500_Return_7d'].ffill()
            
            # 2. Forward fill any other minor gaps
            df['BTC_Volatility_30d'] = df['BTC_Volatility_30d'].ffill()
            
            # 3. Handle Fear & Greed (API only goes back to Feb 2018)
            # We fill missing pre-2018 values with 50 (Neutral) to preserve the 2016-2017 dataset
            df['Fear_Greed_Score'] = df['Fear_Greed_Score'].fillna(50)
            
            # Drop any lingering NaNs (e.g. the first 30 days where BTC volatility was NaN)
            df = df.dropna()
            
            # Save
            df.to_csv(filepath)
            success_count += 1
            total_rows += len(df)
            print(f"  -> Successfully merged macro data into {f} ({len(df)} rows)")
            
        except Exception as e:
            print(f"  -> Failed on {f}: {str(e)}")
            
    print(f"\\nPhase 3 Complete! Macro data injected into {success_count}/{len(files)} assets.")
    print(f"Total model-ready rows remaining: {total_rows:,}")

if __name__ == "__main__":
    main()
