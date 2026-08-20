import os
import pandas as pd
import numpy as np
import shutil

DATA_DIR = "./data"
COMMODITIES = ['gold', 'silver', 'copper', 'crude_oil', 'natural_gas', 'wheat']
DECAY_FACTOR = 0.95

def process_sentiment_features():
    print("Starting Horizon-Aware Feature Engineering & Pipeline Merging...")

    for commodity in COMMODITIES:
        print(f"\nProcessing {commodity}...")

        # 1. Load Scored Sentiment Data
        hist_file = os.path.join(DATA_DIR, f"{commodity}_news_historical_scored.csv")
        live_file = os.path.join(DATA_DIR, f"{commodity}_news_live_scored.csv")

        dfs_to_concat = []
        if os.path.exists(hist_file):
            dfs_to_concat.append(pd.read_csv(hist_file))
        if os.path.exists(live_file):
            dfs_to_concat.append(pd.read_csv(live_file))

        if not dfs_to_concat:
            print(f"  No scored data found for {commodity}. Skipping.")
            continue

        news_df = pd.concat(dfs_to_concat, ignore_index=True)
        
        # 2. Time-Shift Logic (Prevent Look-Ahead Bias)
        if 'Time' in news_df.columns:
            # If article published after 16:00:00 (4 PM), shift it to the next day
            news_df['Date'] = pd.to_datetime(news_df['Date'])
            news_df['Time'] = pd.to_datetime(news_df['Time'], format='%H:%M:%S', errors='coerce').dt.time
            
            cutoff_time = pd.to_datetime('16:00:00', format='%H:%M:%S').time()
            # Find rows where time is not null and is after cutoff
            mask = news_df['Time'].notnull() & (news_df['Time'] > cutoff_time)
            
            if mask.any():
                news_df.loc[mask, 'Date'] = news_df.loc[mask, 'Date'] + pd.Timedelta(days=1)
                
            news_df['Date'] = news_df['Date'].dt.strftime('%Y-%m-%d')

        # 3. Aggregate to Daily Level
        daily_sentiment = news_df.groupby('Date').agg(
            News_Volume=('Headline', 'count'),
            Daily_Bullish=('Weighted_Bullish', 'mean'),
            Daily_Bearish=('Weighted_Bearish', 'mean')
        ).reset_index()

        daily_sentiment['Daily_NSS'] = daily_sentiment['Daily_Bullish'] - daily_sentiment['Daily_Bearish']

        # 4. Load Master OHLCV Data
        master_file = os.path.join(DATA_DIR, f"{commodity}.csv")
        if not os.path.exists(master_file):
            print(f"  Master file {master_file} not found. Skipping merge.")
            continue

        master_df = pd.read_csv(master_file)
        # Create a backup before modifying
        backup_file = os.path.join(DATA_DIR, f"{commodity}_backup_pre_sentiment.csv")
        if not os.path.exists(backup_file): # Only backup once
            shutil.copy(master_file, backup_file)

        # 5. Merge Sentiment into Master Data (Left Join)
        master_df = pd.merge(master_df, daily_sentiment, on='Date', how='left')

        # 6. Handle Weekends & "No News" Days with Decay Factor
        master_df['News_Volume'] = master_df['News_Volume'].fillna(0)
        
        # Create a forward-filled version
        ffill_nss = master_df['Daily_NSS'].ffill()
        ffill_bull = master_df['Daily_Bullish'].ffill()
        ffill_bear = master_df['Daily_Bearish'].ffill()

        # Calculate days since last news event
        # Create a boolean series where True means we have data
        has_data = master_df['Daily_NSS'].notnull()
        # Group by the cumulative sum of has_data. The cumcount gives us days since the last True
        days_since = master_df.groupby(has_data.cumsum()).cumcount()

        # Apply exponential decay: value * (0.95 ^ days_since)
        decay_multiplier = DECAY_FACTOR ** days_since
        
        master_df['Daily_NSS'] = ffill_nss * decay_multiplier
        master_df['Daily_Bullish'] = ffill_bull * decay_multiplier
        master_df['Daily_Bearish'] = ffill_bear * decay_multiplier
        
        # Fill any remaining NaNs at the very beginning of the dataset with 0
        master_df['Daily_NSS'] = master_df['Daily_NSS'].fillna(0)
        master_df['Daily_Bullish'] = master_df['Daily_Bullish'].fillna(0)
        master_df['Daily_Bearish'] = master_df['Daily_Bearish'].fillna(0)

        # 7. Horizon-Aware Feature Engineering
        # Short-Term Spikes
        master_df['Sentiment_EMA_7d'] = master_df['Daily_NSS'].ewm(span=7, adjust=False).mean()
        master_df['Sentiment_Spike'] = master_df['Daily_NSS'] - master_df['Sentiment_EMA_7d']

        # Long-Term Averages
        master_df['Sentiment_EMA_30d'] = master_df['Daily_NSS'].ewm(span=30, adjust=False).mean()
        master_df['Sentiment_EMA_60d'] = master_df['Daily_NSS'].ewm(span=60, adjust=False).mean()

        # 8. Regime-Contextual Interaction Feature
        # Calculate a quick 50-day SMA for the trend interaction if it doesn't exist
        if 'Close' in master_df.columns:
            sma_50 = master_df['Close'].rolling(window=50).mean()
            close_to_ma = (master_df['Close'] - sma_50) / sma_50
            master_df['Sentiment_x_Trend'] = master_df['Daily_NSS'] * close_to_ma.fillna(0)
        else:
            master_df['Sentiment_x_Trend'] = 0.0

        # Save back to master CSV
        master_df.to_csv(master_file, index=False)
        print(f"  Successfully integrated sentiment into {master_file}")
        print(f"  New columns added: News_Volume, Daily_NSS, Daily_Bullish, Daily_Bearish, Sentiment_Spike, Sentiment_EMA_60d, Sentiment_x_Trend")

if __name__ == "__main__":
    process_sentiment_features()
