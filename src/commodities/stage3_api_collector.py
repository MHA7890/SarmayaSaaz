import os
import pandas as pd
import requests
import quandl
from dotenv import load_dotenv

load_dotenv()

# Securely load API keys
EIA_API_KEY = os.getenv('EIA_API_KEY')
NASDAQ_API_KEY = os.getenv('NASDAQ_API_KEY')

data_dir = './data'

def fetch_eia_opec_production():
    """Fetches OPEC Crude Oil Production from the EIA API."""
    print("Fetching OPEC Crude Oil Production from EIA...")
    url = f"https://api.eia.gov/v2/international/data/?api_key={EIA_API_KEY}&frequency=monthly&data[0]=value&facets[activityId][]=1&facets[productId][]=53&facets[countryRegionId][]=OPEC&sort[0][column]=period&sort[0][direction]=desc"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if 'response' in data and 'data' in data['response']:
            records = data['response']['data']
            df = pd.DataFrame(records)
            
            # The 'period' column is YYYY-MM
            df['Date'] = pd.to_datetime(df['period'], format='%Y-%m')
            
            # Since production is reported monthly, assign to the end of the month
            df['Date'] = df['Date'] + pd.offsets.MonthEnd(0)
            
            # Convert 'value' to numeric (Thousand Barrels Per Day -> Millions of Barrels for clean scaling)
            df['OPEC_Production'] = pd.to_numeric(df['value'], errors='coerce') / 1000.0
            
            df = df[['Date', 'OPEC_Production']]
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            out_path = os.path.join(data_dir, 'opec_production.csv')
            df.to_csv(out_path)
            print(f"Successfully saved {len(df)} records to {out_path}")
            return df
        else:
            print("EIA API Response Error:", data)
            return None
    except Exception as e:
        print(f"Failed to fetch EIA data: {e}")
        return None

def fetch_nasdaq_lme_copper():
    """Fetches LME Copper Inventory from Nasdaq Data Link."""
    print("Fetching LME Copper Inventory from Nasdaq Data Link...")
    quandl.ApiConfig.api_key = NASDAQ_API_KEY
    try:
        # LME/PR_CU contains LME Copper Prices and Inventory
        df = quandl.get('LME/PR_CU')
        # The column for inventory is usually 'Stock' or similar. 
        # For LME/PR_CU it might be 'Stock' or 'Inventory'.
        # Let's inspect it dynamically
        print("LME Copper Columns:", df.columns.tolist())
        
        # We will look for a column containing 'Stock' or 'Inventory'
        inv_col = [col for col in df.columns if 'Stock' in col or 'Inventory' in col or 'stock' in col.lower()]
        
        if inv_col:
            df_inv = df[[inv_col[0]]].copy()
            df_inv.rename(columns={inv_col[0]: 'LME_Copper_Inventory'}, inplace=True)
            df_inv.index.name = 'Date'
            
            out_path = os.path.join(data_dir, 'lme_copper.csv')
            df_inv.to_csv(out_path)
            print(f"Successfully saved {len(df_inv)} records to {out_path}")
            return df_inv
        else:
            print("Could not locate an Inventory column in the dataset. Available columns:", df.columns)
            return None
    except Exception as e:
        print(f"Failed to fetch Nasdaq Data Link data: {e}")
        return None

def merge_new_macro_features():
    """Merges the newly fetched API data and the parsed China PMI into the master CSVs."""
    print("\n--- Merging new API features into master datasets ---")
    
    # --- 1. Crude Oil Merge ---
    crude_path = os.path.join(data_dir, 'crude_oil.csv')
    opec_path = os.path.join(data_dir, 'opec_production.csv')
    
    if os.path.exists(crude_path) and os.path.exists(opec_path):
        df_crude = pd.read_csv(crude_path, index_col='Date', parse_dates=True)
        df_opec = pd.read_csv(opec_path, index_col='Date', parse_dates=True)
        
        # Drop overlapping to make idempotent
        overlapping = df_crude.columns.intersection(df_opec.columns)
        if len(overlapping) > 0:
            df_crude.drop(columns=overlapping, inplace=True)
            
        df_merged_crude = df_crude.join(df_opec, how='left')
        df_merged_crude['OPEC_Production'] = df_merged_crude['OPEC_Production'].ffill()
        df_merged_crude.dropna(inplace=True)
        df_merged_crude.to_csv(crude_path)
        print(f"Successfully injected OPEC Production into {crude_path}. New shape: {df_merged_crude.shape}")
        
    # --- 2. Copper Merge ---
    copper_path = os.path.join(data_dir, 'copper.csv')
    china_pmi_path = os.path.join(data_dir, 'china_pmi.csv')
    lme_path = os.path.join(data_dir, 'lme_copper.csv')
    
    if os.path.exists(copper_path):
        df_copper = pd.read_csv(copper_path, index_col='Date', parse_dates=True)
        
        # Merge China PMI
        if os.path.exists(china_pmi_path):
            df_china = pd.read_csv(china_pmi_path, index_col='Date', parse_dates=True)
            overlapping = df_copper.columns.intersection(df_china.columns)
            if len(overlapping) > 0:
                df_copper.drop(columns=overlapping, inplace=True)
            df_copper = df_copper.join(df_china, how='left')
            df_copper['China_PMI'] = df_copper['China_PMI'].ffill()
            
        # Merge LME
        if os.path.exists(lme_path):
            df_lme = pd.read_csv(lme_path, index_col='Date', parse_dates=True)
            overlapping = df_copper.columns.intersection(df_lme.columns)
            if len(overlapping) > 0:
                df_copper.drop(columns=overlapping, inplace=True)
            df_copper = df_copper.join(df_lme, how='left')
            df_copper['LME_Copper_Inventory'] = df_copper['LME_Copper_Inventory'].ffill()
            
        df_copper.dropna(inplace=True)
        df_copper.to_csv(copper_path)
        print(f"Successfully injected new indicators into {copper_path}. New shape: {df_copper.shape}")

if __name__ == "__main__":
    fetch_eia_opec_production()
    fetch_nasdaq_lme_copper()
    merge_new_macro_features()
