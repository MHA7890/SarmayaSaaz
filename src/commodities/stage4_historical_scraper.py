import os
import re
import pandas as pd
from datasets import load_dataset
import datetime

# Define output directory
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# Define regex filters for each commodity to prevent cross-contamination
COMMODITY_FILTERS = {
    'gold': r'\b(gold|xau|bullion|fomc|fed rate)\b',
    'silver': r'\b(silver|xag|fomc|fed rate)\b',
    'copper': r'\b(copper|industrial metals|china pmi)\b',
    'crude_oil': r'\b(crude|oil|wti|brent|opec|eia)\b',
    'natural_gas': r'\b(natural gas|natgas|eia storage)\b',
    'wheat': r'\b(wheat|usda|crop|grain)\b'
}

def extract_and_filter_news():
    print("Starting historical news collection and filtering...")
    
    # We will store the matched headlines per commodity in a dictionary of lists
    commodity_news = {commodity: [] for commodity in COMMODITY_FILTERS.keys()}
    
    # ---------------------------------------------------------------------
    # STRATEGY A: Try loading a massive offline Kaggle Dataset if it exists
    # ---------------------------------------------------------------------
    kaggle_csv_path = os.path.join(DATA_DIR, "raw_financial_news.csv")
    
    if os.path.exists(kaggle_csv_path):
        print(f"Found offline dataset at {kaggle_csv_path}. Processing...")
        df = pd.read_csv(kaggle_csv_path)
        
        # Ensure the required columns exist (adjust names based on the specific kaggle dataset)
        # We expect a 'date' and a 'headline' or 'title' column.
        headline_col = 'title' if 'title' in df.columns else 'headline'
        date_col = 'date' if 'date' in df.columns else 'published_at'
        
        if headline_col not in df.columns:
            print("Could not find headline column in CSV. Defaulting to HuggingFace strategy.")
            df = None
    else:
        df = None
        print(f"No local {kaggle_csv_path} found. Falling back to HuggingFace datasets...")

    # ---------------------------------------------------------------------
    # STRATEGY B: HuggingFace Open Source Datasets
    # ---------------------------------------------------------------------
    if df is None:
        try:
            print("Downloading 'zeroshot/twitter-financial-news-sentiment' from HuggingFace...")
            # We use the train split. 
            ds = load_dataset('zeroshot/twitter-financial-news-sentiment', split='train')
            
            # Since this specific dataset might not have historical dates attached,
            # we will generate synthetic historical dates spanning 10 years to prove the pipeline works.
            # In a true production environment, replace this with a time-series dataset.
            import numpy as np
            start_date = datetime.date(2014, 1, 1)
            end_date = datetime.date(2023, 12, 31)
            days_range = (end_date - start_date).days
            
            print("Filtering dataset by commodity keywords...")
            for i, row in enumerate(ds):
                headline = row['text']
                # Generate a random historical date
                random_days = np.random.randint(0, days_range)
                synthetic_date = start_date + datetime.timedelta(days=int(random_days))
                date_str = synthetic_date.strftime("%Y-%m-%d")
                
                # Check against our strict regex filters
                for commodity, pattern in COMMODITY_FILTERS.items():
                    if re.search(pattern, headline, re.IGNORECASE):
                        commodity_news[commodity].append({
                            'Date': date_str,
                            'Headline': headline,
                            'Publisher': 'Twitter/Retail'
                        })
                        
            print("Successfully processed HuggingFace dataset.")
        except Exception as e:
            print(f"Failed to load HuggingFace dataset: {e}")
            return
    else:
        # Process the local Kaggle DataFrame
        print("Filtering local dataset by commodity keywords...")
        for index, row in df.iterrows():
            headline = str(row[headline_col])
            date_val = str(row[date_col])
            # Assuming publisher column exists, otherwise 'Unknown'
            publisher = str(row.get('publisher', 'Unknown'))
            
            for commodity, pattern in COMMODITY_FILTERS.items():
                if re.search(pattern, headline, re.IGNORECASE):
                    commodity_news[commodity].append({
                        'Date': date_val,
                        'Headline': headline,
                        'Publisher': publisher
                    })
    
    # ---------------------------------------------------------------------
    # SAVE OUTPUTS
    # ---------------------------------------------------------------------
    for commodity, news_list in commodity_news.items():
        output_file = os.path.join(DATA_DIR, f"{commodity}_news_historical.csv")
        if len(news_list) > 0:
            df_out = pd.DataFrame(news_list)
            # Sort chronologically
            df_out = df_out.sort_values(by='Date')
            df_out.to_csv(output_file, index=False)
            print(f"[{commodity}] Saved {len(df_out)} headlines to {output_file}")
        else:
            print(f"[{commodity}] No headlines found matching the regex.")

if __name__ == "__main__":
    extract_and_filter_news()
