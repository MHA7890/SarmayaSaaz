import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
import logging
import torch
import torch.nn as nn

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# --- Re-define LSTM Architecture for Loading ---
class MUFAP_LSTM(nn.Module):
    def __init__(self, input_dim):
        super(MUFAP_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.unsqueeze(1) 
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out.squeeze(1)

class MUFAPPredictor:
    def __init__(self, 
                 raw_file="MUFAP_Historical_NAV.csv",
                 clustered_dir="data/mufap_clustered",
                 model_dir="models/mufap",
                 results_dir="results/mufap"):
        
        self.raw_file = raw_file
        self.clustered_dir = clustered_dir
        self.model_dir = model_dir
        
        # Load Routing Map
        with open(os.path.join(results_dir, "ensemble_weights.json"), "r") as f:
            self.routing_table = json.load(f)
            
        # Build Category Map
        raw_df = pd.read_csv(raw_file)
        self.fund_to_cat = dict(zip(raw_df['Fund'].dropna(), raw_df['Category'].dropna()))
        self.horizons = [7, 14, 28, 42, 60, 90, 120]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _determine_cluster(self, fund_name):
        if fund_name not in self.fund_to_cat:
            raise ValueError(f"Fund '{fund_name}' not found in database.")
            
        category = str(self.fund_to_cat[fund_name]).lower()
        
        if 'equity' in category or 'index' in category or 'etf' in category:
            return 'Equity'
        elif 'money market' in category:
            return 'MoneyMarket'
        elif 'income' in category or 'fixed' in category or 'debt' in category:
            return 'Income'
        elif 'commodit' in category or 'gold' in category:
            return 'Commodity'
        else:
            return 'Balanced'
            
    def _load_model(self, cluster, horizon, input_dim):
        route = self.routing_table[cluster][f"{horizon}d"]
        model_name = route["Winning_Model"]
        model_path = os.path.join(self.model_dir, route["Path"])
        
        if model_name == "LSTM":
            model = MUFAP_LSTM(input_dim).to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            model.eval()
            return model, "LSTM"
        else:
            model = joblib.load(model_path)
            return model, model_name

    def predict(self, fund_name):
        logger.info(f"Initializing Inference for: {fund_name}")
        
        cluster = self._determine_cluster(fund_name)
        logger.info(f"Routed to Super-Cluster: {cluster}")
        
        # In a true live deployment, we would fetch today's NAV from an API and today's macro from yfinance here.
        # For this prototype, we load the absolute final row of the clustered dataset (which represents "Today").
        safe_fund_name = fund_name.replace('/', '_').replace('\\', '_')
        file_path = os.path.join(self.clustered_dir, cluster, f"{safe_fund_name}.csv")
        
        if not os.path.exists(file_path):
            # Try direct name
            file_path = os.path.join(self.clustered_dir, cluster, f"{fund_name}.csv")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Clustered data for {fund_name} not found. Did it pass the 200-day Age Filter?")
                
        df = pd.read_csv(file_path, index_col='Validity_Date', parse_dates=True)
        latest_date = df.index[-1]
        
        # Base Feature Columns (Drop non-features exactly like Stage 4)
        drop_cols = ['Fund'] + [f'Target_{h}d' for h in self.horizons]
        feature_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype != 'object']
        
        # Isolate the final row (Today)
        live_data = df.iloc[-1:][feature_cols]
        current_nav = live_data['NAV_to_SMA_200'].values[0] # Wait, NAV was dropped in Stage 2!
        
        # We need the absolute NAV to convert percentages to Rupee values.
        # Since NAV was dropped in Stage 2 to prevent leakage, we must look it up from the raw CSV!
        raw_df = pd.read_csv(self.raw_file)
        raw_fund_df = raw_df[raw_df['Fund'] == fund_name].sort_values('Validity_Date')
        current_nav = float(raw_fund_df.iloc[-1]['NAV'])
        
        logger.info(f"Latest Data Date: {latest_date.date()} | Current NAV: Rs {current_nav:.4f}")
        
        # Load Frozen Scaler
        scaler_path = os.path.join(self.model_dir, f"{cluster}_scaler.pkl")
        scaler = joblib.load(scaler_path)
        scaled_features = scaler.transform(live_data)
        
        results = {
            "Fund": fund_name,
            "Cluster": cluster,
            "Latest_Date": str(latest_date.date()),
            "Current_NAV": current_nav,
            "Predictions": {}
        }
        
        # Run Gauntlet
        for h in self.horizons:
            model, model_type = self._load_model(cluster, h, scaled_features.shape[1])
            
            if model_type == "LSTM":
                x_t = torch.FloatTensor(scaled_features).to(self.device)
                with torch.no_grad():
                    pred_pct = model(x_t).cpu().numpy()[0]
            else:
                pred_pct = model.predict(scaled_features)[0]
                
            # Math Translation: Percentage to PKR
            projected_nav = current_nav * (1 + pred_pct)
            
            results["Predictions"][f"{h}d"] = {
                "Predicted_Return_Pct": round(float(pred_pct) * 100, 2),
                "Projected_NAV": round(float(projected_nav), 4),
                "Model_Used": model_type
            }
            
        return results

if __name__ == "__main__":
    predictor = MUFAPPredictor()
    
    # Test on a highly stable Money Market fund
    print("\n" + "="*50)
    mm_res = predictor.predict("UBL Money Market Fund")
    print(json.dumps(mm_res, indent=4))
    
    # Test on a highly volatile Equity fund
    print("\n" + "="*50)
    eq_res = predictor.predict("Meezan Islamic Fund")
    print(json.dumps(eq_res, indent=4))
    
    # Test on a Gold fund
    print("\n" + "="*50)
    gold_res = predictor.predict("Meezan Gold Fund")
    print(json.dumps(gold_res, indent=4))
    print("="*50 + "\n")
