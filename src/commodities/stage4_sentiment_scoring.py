import os
import glob
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import logging

logging.set_verbosity_error()

DATA_DIR = "./data"

# Topics for Zero-Shot Classification (Advanced Alpha Strategy #2)
TOPIC_LABELS = ['Supply', 'Macro', 'Geopolitics']

# Entities that require sentiment inversion for specific commodities (Advanced Alpha Strategy #1)
# If Gold/Silver news is primarily about strong Dollar/Yields/Fed, "Bullish" means Bullish for Dollar, which is Bearish for Gold.
INVERSE_KEYWORDS = ['dollar', 'dxy', 'fomc', 'fed', 'yield', 'rate hike', 'powell']
INVERSE_TARGETS = ['gold', 'silver']

# Source credibility weights (Advanced Alpha Strategy #3)
def get_source_weight(publisher_name):
    pub_lower = str(publisher_name).lower()
    tier_1 = ['reuters', 'bloomberg', 'wsj', 'wall street journal', 'fed', 'federal reserve']
    tier_2 = ['cnbc', 'yahoo', 'marketwatch', 'seeking alpha', 'motley fool', 'unknown']
    
    if any(t1 in pub_lower for t1 in tier_1):
        return 1.5
    elif any(t2 in pub_lower for t2 in tier_2):
        return 1.0
    else:
        return 0.5 # Unknown/Blogs are penalized

def process_and_score():
    print("Loading AI Models... This may take a minute.")
    
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

    # Load FinBERT
    finbert_name = "ProsusAI/finbert"
    sentiment_pipe = pipeline("text-classification", model=finbert_name, tokenizer=finbert_name, top_k=None, device=device)

    # Load Zero-Shot Classifier
    zero_shot_name = "facebook/bart-large-mnli"
    zero_shot_pipe = pipeline("zero-shot-classification", model=zero_shot_name, device=device)

    # Find all historical and live news CSVs
    news_files = glob.glob(os.path.join(DATA_DIR, "*_news_historical.csv")) + \
                 glob.glob(os.path.join(DATA_DIR, "*_news_live.csv"))

    if not news_files:
        print("No news files found to score!")
        return

    for file_path in news_files:
        filename = os.path.basename(file_path)
        commodity = filename.split('_news_')[0]
        file_type = "live" if "live" in filename else "historical"
        
        print(f"\nProcessing [{commodity}] {file_type} news... ({filename})")
        df = pd.read_csv(file_path)
        
        if df.empty:
            print("File is empty, skipping.")
            continue

        bullish_scores = []
        bearish_scores = []
        topics = []
        weights = []

        headlines = df['Headline'].tolist()
        publishers = df['Publisher'].tolist() if 'Publisher' in df.columns else ['Unknown'] * len(df)

        for i, (headline, publisher) in enumerate(zip(headlines, publishers)):
            if i % 50 == 0 and i > 0:
                print(f"  Scored {i}/{len(df)} headlines...")

            # 1. Source Credibility Weighting
            weight = get_source_weight(publisher)
            weights.append(weight)

            # 2. Zero-Shot Topic Classification
            try:
                topic_res = zero_shot_pipe(headline, candidate_labels=TOPIC_LABELS)
                best_topic = topic_res['labels'][0] # Topic with highest probability
                topics.append(best_topic)
            except Exception as e:
                topics.append('Macro') # fallback

            # 3. FinBERT Sentiment Scoring
            try:
                sent_res = sentiment_pipe(headline)[0]
                
                bull_val = 0.0
                bear_val = 0.0
                
                # FinBERT returns [{'label': 'positive', 'score': 0.8}, ...]
                for r in sent_res:
                    if r['label'] == 'positive': bull_val = r['score']
                    if r['label'] == 'negative': bear_val = r['score']
                
                # 4. Entity Inversion Check (Loss Aversion & Inverse correlation)
                # If the commodity is Gold/Silver and the headline is about the Fed/Dollar raising rates
                is_inverse = False
                if commodity in INVERSE_TARGETS:
                    if any(inv_kw in headline.lower() for inv_kw in INVERSE_KEYWORDS):
                        is_inverse = True
                
                if is_inverse:
                    # Invert the logic: Bullish for Dollar = Bearish for Gold
                    temp = bull_val
                    bull_val = bear_val
                    bear_val = temp
                
                bullish_scores.append(bull_val)
                bearish_scores.append(bear_val)
                
            except Exception as e:
                bullish_scores.append(0.0)
                bearish_scores.append(0.0)

        # Apply weights to the scores
        df['Topic'] = topics
        df['Source_Weight'] = weights
        df['Raw_Bullish'] = bullish_scores
        df['Raw_Bearish'] = bearish_scores
        
        # Calculate Weighted Asymmetric Scores
        df['Weighted_Bullish'] = df['Raw_Bullish'] * df['Source_Weight']
        df['Weighted_Bearish'] = df['Raw_Bearish'] * df['Source_Weight']
        
        # Calculate standard NSS just in case we need a single feature baseline
        df['NSS'] = df['Weighted_Bullish'] - df['Weighted_Bearish']

        # Save Scored Data
        output_filename = filename.replace(".csv", "_scored.csv")
        output_path = os.path.join(DATA_DIR, output_filename)
        df.to_csv(output_path, index=False)
        print(f"Saved scored data to {output_filename}")

if __name__ == "__main__":
    process_and_score()
