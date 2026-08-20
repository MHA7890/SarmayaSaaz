import os
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# Strict queries to prevent cross-contamination
COMMODITY_QUERIES = {
    'gold': "(gold OR xau OR bullion) AND (price OR market OR fed)",
    'silver': "(silver OR xag) AND (price OR market OR fed)",
    'copper': "(copper) AND (industrial OR china OR price)",
    'crude_oil': "(crude OR oil OR WTI OR brent) AND (OPEC OR supply OR prices)",
    'natural_gas': "(natural gas OR natgas) AND (EIA OR weather OR price)",
    'wheat': "(wheat OR grain) AND (USDA OR crop OR weather)"
}

def fetch_live_news():
    print("Starting Live API News Collection...")
    
    if not NEWS_API_KEY or NEWS_API_KEY == "your_news_api_key_here":
        print("ERROR: NEWS_API_KEY is not set in .env file.")
        print("Please sign up at newsapi.org to get a free API key and add it to .env")
        return

    url = "https://newsapi.org/v2/everything"

    for commodity, query in COMMODITY_QUERIES.items():
        print(f"Fetching news for {commodity}...")
        
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 50, # Free tier limits
            "apiKey": NEWS_API_KEY
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if data.get('status') != 'ok':
                print(f"API Error for {commodity}: {data.get('message', 'Unknown error')}")
                continue
                
            articles = data.get('articles', [])
            if not articles:
                print(f"  No articles found for {commodity}.")
                continue
                
            parsed_data = []
            for article in articles:
                # Extract and parse datetime (e.g. "2024-10-24T14:30:00Z")
                pub_str = article.get('publishedAt')
                if not pub_str: continue
                
                try:
                    # Convert to UTC datetime
                    dt_obj = datetime.strptime(pub_str, "%Y-%m-%dT%H:%M:%SZ")
                    dt_obj = dt_obj.replace(tzinfo=timezone.utc)
                    
                    date_str = dt_obj.strftime("%Y-%m-%d")
                    time_str = dt_obj.strftime("%H:%M:%S")
                except Exception as e:
                    print(f"  Date parsing error: {e}")
                    continue
                    
                headline = article.get('title', '')
                source_name = article.get('source', {}).get('name', 'Unknown')
                article_url = article.get('url', '')
                
                if headline:
                    parsed_data.append({
                        'Date': date_str,
                        'Time': time_str,
                        'Headline': headline,
                        'Publisher': source_name,
                        'URL': article_url
                    })
                    
            if parsed_data:
                df_new = pd.DataFrame(parsed_data)
                
                # We append to a live file (or create it)
                file_path = os.path.join(DATA_DIR, f"{commodity}_news_live.csv")
                
                if os.path.exists(file_path):
                    df_existing = pd.read_csv(file_path)
                    # Concat and drop duplicates based on Headline and Date
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    df_combined = df_combined.drop_duplicates(subset=['Headline', 'Date'], keep='last')
                else:
                    df_combined = df_new
                    
                # Sort chronologically
                df_combined = df_combined.sort_values(by=['Date', 'Time'])
                df_combined.to_csv(file_path, index=False)
                
                print(f"  Successfully saved {len(df_new)} new articles to {file_path}")
                
        except Exception as e:
            print(f"Request failed for {commodity}: {e}")

if __name__ == "__main__":
    fetch_live_news()
