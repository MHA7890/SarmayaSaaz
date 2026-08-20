import os
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import warnings
warnings.filterwarnings('ignore')

# Curated List of Crypto Assets (31 Assets across 4 Clusters)
TICKERS = [
    # Cluster 1: Macro Majors
    "BTC", "ETH", "SOL", "BNB", "AVAX", "ADA", "NEAR", "SUI", "APT", "TRX",
    # Cluster 2: DeFi Blue Chips
    "LINK", "AAVE", "UNI", "CRV", "LDO", "MKR", "PENDLE", "SNX",
    # Cluster 3: AI, Gaming, DePIN
    "TAO", "FET", "RNDR", "FIL", "GRT", "INJ", "IMX",
    # Cluster 4: Memecoins
    "DOGE", "SHIB", "PEPE", "WIF", "BONK", "FLOKI"
]

OUTPUT_DIR = "data/crypto"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_technical_indicators(df):
    """Calculates stationary Technical Indicators."""
    # 1. Log Returns (Expanded to 120d as per user request)
    periods = [1, 7, 14, 30, 60, 90, 120]
    for p in periods:
        df[f'Log_Return_{p}d'] = np.log(df['Close'] / df['Close'].shift(p))
        
    # 2. Momentum
    df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['MACD_Hist'] = macd.macd_diff()
    
    # 3. Volatility
    df['ATR_14'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    
    # Normalized ATR (Stationary)
    df['ATR_14_Pct'] = df['ATR_14'] / df['Close']
    
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Width'] = bb.bollinger_wband()
    
    df['Volatility_30d'] = df['Log_Return_1d'].rolling(window=30).std() * np.sqrt(365)
    
    # 4. Moving Average Ratios (Stationary)
    for w in [20, 50, 200]:
        sma = ta.trend.SMAIndicator(df['Close'], window=w).sma_indicator()
        df[f'Close_to_SMA{w}'] = (df['Close'] - sma) / sma

    # 5. Volume
    df['Volume_Change_1d'] = df['Volume'].pct_change()
    
    return df

def calculate_smc(df):
    """Calculates Smart Money Concepts (SMC)."""
    # 1. Fair Value Gaps (FVG)
    # Bullish FVG: Low of candle 3 is higher than High of candle 1
    df['Bullish_FVG'] = (df['Low'] > df['High'].shift(2)) & (df['Close'].shift(1) > df['Open'].shift(1))
    # Bearish FVG: High of candle 3 is lower than Low of candle 1
    df['Bearish_FVG'] = (df['High'] < df['Low'].shift(2)) & (df['Close'].shift(1) < df['Open'].shift(1))
    
    # 2. Swing Highs / Lows (Lookback=3, Lookforward=3 to confirm)
    # Note: Using shift ensures no lookahead bias at the time of calculation
    df['Swing_High'] = (df['High'].shift(3) > df['High'].shift(4)) & \
                       (df['High'].shift(3) > df['High'].shift(5)) & \
                       (df['High'].shift(3) > df['High'].shift(6)) & \
                       (df['High'].shift(3) > df['High'].shift(2)) & \
                       (df['High'].shift(3) > df['High'].shift(1)) & \
                       (df['High'].shift(3) > df['High'])
                       
    df['Swing_Low'] = (df['Low'].shift(3) < df['Low'].shift(4)) & \
                      (df['Low'].shift(3) < df['Low'].shift(5)) & \
                      (df['Low'].shift(3) < df['Low'].shift(6)) & \
                      (df['Low'].shift(3) < df['Low'].shift(2)) & \
                      (df['Low'].shift(3) < df['Low'].shift(1)) & \
                      (df['Low'].shift(3) < df['Low'])
    
    # Forward fill the last confirmed swing high/low price
    df['Last_Swing_High_Price'] = np.where(df['Swing_High'], df['High'].shift(3), np.nan)
    df['Last_Swing_High_Price'] = df['Last_Swing_High_Price'].ffill()
    
    df['Last_Swing_Low_Price'] = np.where(df['Swing_Low'], df['Low'].shift(3), np.nan)
    df['Last_Swing_Low_Price'] = df['Last_Swing_Low_Price'].ffill()
    
    # 3. Market Structure (Break of Structure - BOS)
    # Price closes above the last confirmed swing high
    df['Bullish_BOS'] = (df['Close'] > df['Last_Swing_High_Price']) & (df['Close'].shift(1) <= df['Last_Swing_High_Price'])
    df['Bearish_BOS'] = (df['Close'] < df['Last_Swing_Low_Price']) & (df['Close'].shift(1) >= df['Last_Swing_Low_Price'])

    return df

def run_pipeline():
    print(f"Starting Data Collection for {len(TICKERS)} assets...")
    
    for ticker in TICKERS:
        yfinance_ticker = f"{ticker}-USD"
        print(f"Fetching data for {yfinance_ticker}...")
        
        try:
            # Download 10 years of daily data
            df = yf.download(yfinance_ticker, period="10y", interval="1d", progress=False)
            
            if df.empty or len(df) < 200:
                print(f"  -> Skipping {ticker}: Insufficient data ({len(df)} rows).")
                continue
            
            # yfinance returns MultiIndex columns sometimes, flatten them
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
                
            df.index.name = 'Date'
            
            # Forward fill missing OHLCV data from yfinance before math
            df = df.ffill().bfill()
            
            # Feature Engineering
            df = calculate_technical_indicators(df)
            df = calculate_smc(df)
            
            # Target Engineering (Will be done in Phase 2, but we can prep the raw data here)
            
            # Clean up: Drop original raw prices (except Close for now, as it might be needed for targets in Phase 2)
            # We will drop Close in Phase 2 after targets are created.
            df = df.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'ATR_14'], errors='ignore')
            
            # Drop NaN rows created by the 200-day SMA
            df = df.dropna()
            
            # Save
            file_path = os.path.join(OUTPUT_DIR, f"{ticker}.csv")
            df.to_csv(file_path)
            print(f"  -> Successfully saved {ticker} ({len(df)} rows)")
            
        except Exception as e:
            print(f"  -> Failed to process {ticker}: {str(e)}")

if __name__ == "__main__":
    run_pipeline()
    print("\\nPhase 1 Data Collection Complete!")
