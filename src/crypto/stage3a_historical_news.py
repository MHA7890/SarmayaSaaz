"""
Phase 3.5a: Multi-Source Historical Sentiment Acquisition
=========================================================
Downloads two complementary data sources:
  1. CoinDesk News Dataset (229K articles, Oct 2019 - Feb 2025) for FinBERT scoring.
  2. Alternative.me Fear & Greed Index (3,100+ daily scores, Feb 2018 - Today).
"""
import os
import logging
import requests
import pandas as pd
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

RAW_DIR = "data/crypto_raw"
os.makedirs(RAW_DIR, exist_ok=True)


def fetch_coindesk_news():
    """Downloads 229K CoinDesk crypto news articles from HuggingFace (Oct 2019 - Feb 2025)."""
    logger.info("Source 1: Downloading CoinDesk News Dataset from HuggingFace...")
    
    ds = load_dataset("maryamfakhari/crypto-news-coindesk-2020-2025", split="train")
    df = ds.to_pandas()
    logger.info(f"Raw CoinDesk dataset: {len(df)} articles, columns: {df.columns.tolist()}")
    
    # Extract date, headline, and URL (URL is for dashboard display only, not training)
    df = df[['published_on', 'title', 'url']].copy()
    df.rename(columns={'published_on': 'Date', 'title': 'Headline', 'url': 'URL'}, inplace=True)
    
    # Convert timestamps to date strings
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    
    # Drop any rows with missing headlines
    df = df.dropna(subset=['Headline'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    output_path = os.path.join(RAW_DIR, "coindesk_news.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Source 1 Complete: {len(df)} headlines saved ({df['Date'].min()} → {df['Date'].max()})")
    return df


def fetch_fear_greed_index():
    """Downloads the full historical Fear & Greed Index from Alternative.me (Feb 2018 - Today)."""
    logger.info("Source 2: Downloading Fear & Greed Index from Alternative.me API...")
    
    url = "https://api.alternative.me/fng/?limit=0&format=json&date_format=cn"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    data = response.json()['data']
    logger.info(f"API returned {len(data)} daily data points.")
    
    records = []
    for entry in data:
        records.append({
            'Date': entry['timestamp'],
            'FearGreed_Raw': int(entry['value']),           # 0-100 scale
            'FearGreed_Label': entry['value_classification'] # e.g. "Fear", "Greed"
        })
    
    df = pd.DataFrame(records)
    df = df.sort_values('Date').reset_index(drop=True)
    
    output_path = os.path.join(RAW_DIR, "fear_greed_index.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Source 2 Complete: {len(df)} daily scores saved ({df['Date'].min()} → {df['Date'].max()})")
    return df


def main():
    logger.info("=" * 60)
    logger.info("Phase 3.5a: Multi-Source Historical Sentiment Acquisition")
    logger.info("=" * 60)
    
    df_news = fetch_coindesk_news()
    df_fng = fetch_fear_greed_index()
    
    logger.info("=" * 60)
    logger.info(f"ACQUISITION COMPLETE")
    logger.info(f"  CoinDesk News:  {len(df_news):>8,} headlines  ({df_news['Date'].min()} → {df_news['Date'].max()})")
    logger.info(f"  Fear & Greed:   {len(df_fng):>8,} daily scores ({df_fng['Date'].min()} → {df_fng['Date'].max()})")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
