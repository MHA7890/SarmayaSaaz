"""
Phase 5: Live Sentiment Pipeline (For Real-Time Inference)
==========================================================
Fetches today's live news + Fear & Greed Index, scores with FinBERT,
and outputs all 7 sentiment features matching the training data format.
"""
import os
import glob
import json
import logging
import requests
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import urllib.request
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

INPUT_DIR = "data/crypto_processed"
RESULTS_DIR = "results/crypto"
os.makedirs(RESULTS_DIR, exist_ok=True)
MODEL_NAME = "ProsusAI/finbert"


def fetch_rss_news():
    """Fetches live breaking crypto news from Cointelegraph RSS."""
    headlines = []
    try:
        req = urllib.request.Request('https://cointelegraph.com/rss', headers={'User-Agent': 'Mozilla/5.0'})
        xml_data = urllib.request.urlopen(req).read()
        root = ET.fromstring(xml_data)
        for item in root.findall('.//item'):
            title = item.find('title')
            if title is not None and title.text:
                headlines.append(title.text)
        logger.info(f"Successfully scraped {len(headlines)} breaking headlines from Cointelegraph.")
        return headlines
    except Exception as e:
        logger.error(f"Failed to fetch RSS: {str(e)}")
        return []


def fetch_live_fear_greed():
    """Fetches today's Fear & Greed Index score from Alternative.me."""
    try:
        url = "https://api.alternative.me/fng/?limit=1&format=json&date_format=cn"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        value = int(response.json()['data'][0]['value'])
        # Normalize 0-100 to [-1, +1]
        normalized = (value - 50) / 50.0
        logger.info(f"Live Fear & Greed Index: {value}/100 (normalized: {normalized:+.4f})")
        return normalized
    except Exception as e:
        logger.warning(f"Failed to fetch Fear & Greed Index: {str(e)}. Defaulting to 0.0")
        return 0.0


def analyze_sentiment(headlines, tokenizer, model, device):
    """Scores headlines with FinBERT and returns average Net Sentiment Score."""
    if not headlines:
        return 0.0
    scores = []
    model.eval()
    with torch.no_grad():
        for text in headlines:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            net_score = probs[0][0].item() - probs[0][1].item()
            scores.append(net_score)
    return sum(scores) / len(scores)


def main():
    logger.info("Initializing Phase 5: Live Multi-Source Sentiment Pipeline...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    
    # --- Source 1: FinBERT on live RSS headlines ---
    global_headlines = fetch_rss_news()
    news_volume = len(global_headlines)
    sentiment_score = analyze_sentiment(global_headlines, tokenizer, model, device) if global_headlines else 0.0
    
    # --- Source 2: Live Fear & Greed Index ---
    feargreed_score = fetch_live_fear_greed()
    
    # --- Calculate EMAs from the most recent historical values ---
    sample_csv = [f for f in glob.glob(f"{INPUT_DIR}/*.csv") if 'historical' not in f][0]
    df_sample = pd.read_csv(sample_csv)
    
    # Read the last known EMAs from the training data
    last_sent_ema7 = df_sample['Sentiment_EMA_7d'].iloc[-1] if 'Sentiment_EMA_7d' in df_sample.columns else 0.0
    last_sent_ema30 = df_sample['Sentiment_EMA_30d'].iloc[-1] if 'Sentiment_EMA_30d' in df_sample.columns else 0.0
    last_fg_ema7 = df_sample['FearGreed_EMA_7d'].iloc[-1] if 'FearGreed_EMA_7d' in df_sample.columns else 0.0
    last_fg_ema30 = df_sample['FearGreed_EMA_30d'].iloc[-1] if 'FearGreed_EMA_30d' in df_sample.columns else 0.0
    
    # EMA formula: EMA_today = (value * k) + (EMA_yesterday * (1 - k))
    k7 = 2 / (7 + 1)
    k30 = 2 / (30 + 1)
    
    live_features = {
        "Sentiment_Score": round(sentiment_score, 4),
        "News_Volume": news_volume,
        "Sentiment_EMA_7d": round((sentiment_score * k7) + (last_sent_ema7 * (1 - k7)), 4),
        "Sentiment_EMA_30d": round((sentiment_score * k30) + (last_sent_ema30 * (1 - k30)), 4),
        "FearGreed_Score": round(feargreed_score, 4),
        "FearGreed_EMA_7d": round((feargreed_score * k7) + (last_fg_ema7 * (1 - k7)), 4),
        "FearGreed_EMA_30d": round((feargreed_score * k30) + (last_fg_ema30 * (1 - k30)), 4),
    }
    
    output_path = f"{RESULTS_DIR}/live_sentiment.json"
    with open(output_path, "w") as f:
        json.dump(live_features, f, indent=4)
    
    logger.info(f"✅ Phase 5 Complete! All 7 live features saved to {output_path}")
    for k, v in live_features.items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
