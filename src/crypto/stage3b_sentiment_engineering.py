"""
Phase 3.5b: Multi-Source Sentiment Scoring, Fusion & Feature Engineering
========================================================================
1. Scores 229K CoinDesk headlines with FinBERT (GPU batched inference).
2. Loads the Fear & Greed Index and normalizes it to [-1, +1].
3. Fuses both sources into a unified daily sentiment table.
4. Engineers 7 temporal features (EMAs, Volume).
5. Merges all features into the 29 asset CSVs in data/crypto_processed/.
"""
import os
import glob
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

RAW_DIR = "data/crypto_raw"
PROCESSED_DIR = "data/crypto_processed"
MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 512


class NewsDataset(Dataset):
    """PyTorch Dataset wrapper for batched FinBERT tokenization."""
    def __init__(self, texts, tokenizer):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def score_headlines_with_finbert(df_news):
    """Runs FinBERT batched inference on all headlines. Returns df with 'Score' column."""
    logger.info(f"Loading {MODEL_NAME} for batched inference...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    
    texts = df_news['Headline'].tolist()
    dataset = NewsDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    all_scores = []
    
    logger.info(f"Running FinBERT over {len(texts):,} headlines in {len(dataloader)} batches...")
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT labels: [Positive(0), Negative(1), Neutral(2)]
            net_scores = probs[:, 0] - probs[:, 1]
            all_scores.extend(net_scores.cpu().numpy())
    
    df_news['Score'] = all_scores
    return df_news


def build_finbert_daily(df_news):
    """Aggregates FinBERT headline scores into daily Sentiment_Score and News_Volume."""
    daily = df_news.groupby('Date').agg(
        Sentiment_Score=('Score', 'mean'),
        News_Volume=('Score', 'count')
    ).reset_index()
    daily['Date'] = pd.to_datetime(daily['Date'])
    daily = daily.sort_values('Date')
    return daily


def build_fear_greed_daily():
    """Loads the Fear & Greed CSV and normalizes from 0-100 to [-1, +1]."""
    path = os.path.join(RAW_DIR, "fear_greed_index.csv")
    df = pd.read_csv(path)
    
    # Normalize: 0 → -1.0, 50 → 0.0, 100 → +1.0
    df['FearGreed_Score'] = (df['FearGreed_Raw'] - 50) / 50.0
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'FearGreed_Score']].sort_values('Date')
    return df


def fuse_and_engineer(finbert_daily, feargreed_daily):
    """Fuses both sentiment sources and engineers all 7 temporal features."""
    
    # Create a complete date range spanning both sources
    min_date = min(finbert_daily['Date'].min(), feargreed_daily['Date'].min())
    max_date = max(finbert_daily['Date'].max(), feargreed_daily['Date'].max())
    full_dates = pd.DataFrame({'Date': pd.date_range(min_date, max_date, freq='D')})
    
    # Left-join both sources onto the full date range
    fused = full_dates.merge(finbert_daily, on='Date', how='left')
    fused = fused.merge(feargreed_daily, on='Date', how='left')
    
    # Forward-fill FinBERT gaps (weekends with no articles), then fill remaining with 0
    fused['Sentiment_Score'] = fused['Sentiment_Score'].ffill().fillna(0.0)
    fused['News_Volume'] = fused['News_Volume'].ffill().fillna(0)
    
    # Forward-fill Fear & Greed gaps, then fill remaining with 0
    fused['FearGreed_Score'] = fused['FearGreed_Score'].ffill().fillna(0.0)
    
    # Engineer EMAs
    fused['Sentiment_EMA_7d'] = fused['Sentiment_Score'].ewm(span=7, adjust=False).mean()
    fused['Sentiment_EMA_30d'] = fused['Sentiment_Score'].ewm(span=30, adjust=False).mean()
    fused['FearGreed_EMA_7d'] = fused['FearGreed_Score'].ewm(span=7, adjust=False).mean()
    fused['FearGreed_EMA_30d'] = fused['FearGreed_Score'].ewm(span=30, adjust=False).mean()
    
    logger.info(f"Fused sentiment table: {len(fused)} days, {fused.columns.tolist()}")
    logger.info(f"  Coverage: {fused['Date'].min().date()} → {fused['Date'].max().date()}")
    
    return fused


def merge_into_assets(fused):
    """Merges the 7 sentiment features into all 29 asset CSVs."""
    csv_files = glob.glob(f"{PROCESSED_DIR}/*.csv")
    
    sentiment_cols = [
        'Sentiment_Score', 'News_Volume',
        'Sentiment_EMA_7d', 'Sentiment_EMA_30d',
        'FearGreed_Score', 'FearGreed_EMA_7d', 'FearGreed_EMA_30d'
    ]
    
    fused_for_merge = fused[['Date'] + sentiment_cols].copy()
    
    logger.info(f"Merging 7 sentiment features into {len(csv_files)} asset CSVs...")
    
    for file in csv_files:
        asset_df = pd.read_csv(file)
        
        # Drop any pre-existing sentiment columns to avoid duplicates
        existing_sent_cols = [c for c in sentiment_cols if c in asset_df.columns]
        if existing_sent_cols:
            asset_df.drop(columns=existing_sent_cols, inplace=True)
        
        asset_df['Date'] = pd.to_datetime(asset_df['Date'])
        
        merged = pd.merge(asset_df, fused_for_merge, on='Date', how='left')
        
        # Forward-fill then zero-fill any remaining NaNs in sentiment columns
        merged[sentiment_cols] = merged[sentiment_cols].ffill().fillna(0)
        
        merged.to_csv(file, index=False)
    
    logger.info("✅ All 29 asset CSVs enriched with 7 sentiment features.")


def main():
    logger.info("=" * 60)
    logger.info("Phase 3.5b: Multi-Source Sentiment Scoring & Engineering")
    logger.info("=" * 60)
    
    # --- Step 1: Score CoinDesk headlines with FinBERT ---
    news_path = os.path.join(RAW_DIR, "coindesk_news.csv")
    if not os.path.exists(news_path):
        logger.error(f"CoinDesk news not found at {news_path}. Run stage3a first.")
        return
    
    df_news = pd.read_csv(news_path)
    logger.info(f"Loaded {len(df_news):,} CoinDesk headlines for FinBERT scoring.")
    
    df_scored = score_headlines_with_finbert(df_news)
    finbert_daily = build_finbert_daily(df_scored)
    logger.info(f"FinBERT daily table: {len(finbert_daily)} unique days.")
    
    # --- Step 2: Load Fear & Greed Index ---
    feargreed_daily = build_fear_greed_daily()
    logger.info(f"Fear & Greed daily table: {len(feargreed_daily)} unique days.")
    
    # --- Step 3: Fuse & Engineer ---
    fused = fuse_and_engineer(finbert_daily, feargreed_daily)
    
    # --- Step 4: Merge into asset CSVs ---
    merge_into_assets(fused)
    
    logger.info("=" * 60)
    logger.info("Phase 3.5b COMPLETE. Ready for Phase 4 Master Training.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
