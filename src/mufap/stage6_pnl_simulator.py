import os
import pandas as pd
import numpy as np
import json
import joblib
import logging
import torch
import torch.nn as nn

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# --- Re-define LSTM Architecture ---
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

class MUFAPBacktester:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.raw_file = "MUFAP_Historical_NAV.csv"
        self.clustered_dir = "data/mufap_clustered"
        self.model_dir = "models/mufap"
        self.results_dir = "results/mufap"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        with open(os.path.join(self.results_dir, "ensemble_weights.json"), "r") as f:
            self.routing_table = json.load(f)
            
        raw_df = pd.read_csv(self.raw_file)
        self.fund_to_cat = dict(zip(raw_df['Fund'].dropna(), raw_df['Category'].dropna()))
        
    def _determine_cluster(self, fund_name):
        category = str(self.fund_to_cat.get(fund_name, 'Balanced')).lower()
        if 'equity' in category or 'index' in category or 'etf' in category: return 'Equity'
        elif 'money market' in category: return 'MoneyMarket'
        elif 'income' in category or 'fixed' in category or 'debt' in category: return 'Income'
        elif 'commodit' in category or 'gold' in category: return 'Commodity'
        else: return 'Balanced'

    def run_backtest(self, fund_name, days=365):
        logger.info(f"Initiating {days}-Day PnL Simulator for: {fund_name}")
        cluster = self._determine_cluster(fund_name)
        
        # Load Clustered Data
        safe_fund_name = fund_name.replace('/', '_').replace('\\', '_')
        file_path = os.path.join(self.clustered_dir, cluster, f"{safe_fund_name}.csv")
        df = pd.read_csv(file_path, index_col='Validity_Date', parse_dates=True)
        
        # Load Raw Data to get exact absolute NAVs
        raw_df = pd.read_csv(self.raw_file)
        raw_fund = raw_df[raw_df['Fund'] == fund_name].copy()
        raw_fund['Validity_Date'] = pd.to_datetime(raw_fund['Validity_Date'])
        raw_fund.set_index('Validity_Date', inplace=True)
        raw_fund = raw_fund.sort_index()
        
        # Align timelines and slice last N days
        df = df.iloc[-days:].copy()
        start_date = df.index[0]
        end_date = df.index[-1]
        raw_slice = raw_fund.loc[start_date:end_date].copy()
        
        # We need the absolute 'NAV' and 'Spread_Ratio' which was preserved in Stage 1/2
        # If Spread_Ratio is missing in raw, recalculate it
        if 'Spread_Ratio' not in raw_slice.columns:
            raw_slice['Spread_Ratio'] = (raw_slice['Offer'] - raw_slice['NAV']) / raw_slice['NAV']
            
        df['Absolute_NAV'] = raw_slice['NAV']
        df['Daily_Return_Abs'] = df['Absolute_NAV'].pct_change().fillna(0)
        df['Real_Spread'] = raw_slice['Spread_Ratio'].fillna(0)
        
        # Prepare Features
        drop_cols = ['Fund', 'Absolute_NAV', 'Daily_Return_Abs', 'Real_Spread'] + [f'Target_{h}d' for h in [7, 14, 28, 42, 60, 90, 120]]
        feature_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype != 'object']
        X_batch = df[feature_cols].copy()
        
        # Scale Features
        scaler_path = os.path.join(self.model_dir, f"{cluster}_scaler.pkl")
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X_batch)
        
        # Load 7d Model
        route = self.routing_table[cluster]["7d"]
        model_name = route["Winning_Model"]
        model_path = os.path.join(self.model_dir, route["Path"])
        
        if model_name == "LSTM":
            model = MUFAP_LSTM(X_scaled.shape[1]).to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            model.eval()
            with torch.no_grad():
                preds = model(torch.FloatTensor(X_scaled).to(self.device)).cpu().numpy()
        else:
            model = joblib.load(model_path)
            preds = model.predict(X_scaled)
            
        df['AI_7d_Prediction'] = preds
        
        # --- Trading Simulator ---
        baseline_capital = self.initial_capital
        ai_capital = self.initial_capital
        
        # We start in 'Cash' (not invested)
        # Or realistically, 'Buy and Hold' starts fully invested
        baseline_units = self.initial_capital / (df['Absolute_NAV'].iloc[0] * (1 + df['Real_Spread'].iloc[0]))
        
        ai_units = 0.0
        ai_cash = self.initial_capital
        in_market = False
        
        tracker = []
        trades = 0
        
        for i in range(len(df)):
            date = df.index[i]
            current_nav = df['Absolute_NAV'].iloc[i]
            spread = df['Real_Spread'].iloc[i]
            pred_return = df['AI_7d_Prediction'].iloc[i]
            
            # Baseline Value
            base_val = baseline_units * current_nav
            
            # --- AI Logic (The Cash Shield) ---
            # If AI predicts > 0.001 (0.1% gain in 7 days), we want to be IN the market
            if pred_return > 0.001:
                if not in_market:
                    # Buy Signal! Deduct the Spread (Front-End Load Fee)
                    offer_price = current_nav * (1 + spread)
                    ai_units = ai_cash / offer_price
                    ai_cash = 0.0
                    in_market = True
                    trades += 1
            else:
                # Sell Signal! Crash predicted. Move to Cash.
                if in_market:
                    ai_cash = ai_units * current_nav # Assuming selling at NAV
                    ai_units = 0.0
                    in_market = False
                    trades += 1
                    
            ai_val = ai_cash + (ai_units * current_nav)
            
            tracker.append({
                'Date': date,
                'NAV': current_nav,
                'Baseline_Value': base_val,
                'AI_Value': ai_val,
                'In_Market': in_market
            })
            
        res_df = pd.DataFrame(tracker)
        
        # Metrics Calculation
        base_return = ((res_df['Baseline_Value'].iloc[-1] - self.initial_capital) / self.initial_capital) * 100
        ai_return = ((res_df['AI_Value'].iloc[-1] - self.initial_capital) / self.initial_capital) * 100
        alpha = ai_return - base_return
        
        def max_drawdown(series):
            roll_max = series.cummax()
            drawdown = (series - roll_max) / roll_max
            return drawdown.min() * 100
            
        base_dd = max_drawdown(res_df['Baseline_Value'])
        ai_dd = max_drawdown(res_df['AI_Value'])
        
        logger.info(f"--- BACKTEST RESULTS: {fund_name} (365 Days) ---")
        logger.info(f"Buy & Hold Return: {base_return:.2f}% | Max Drawdown: {base_dd:.2f}%")
        logger.info(f"AI Engine Return : {ai_return:.2f}% | Max Drawdown: {ai_dd:.2f}%")
        logger.info(f"AI Alpha Generated: {alpha:+.2f}%")
        logger.info(f"Total AI Trades Executed: {trades}")
        
        # Save Curve for Dashboard
        os.makedirs(os.path.join(self.results_dir, "backtests"), exist_ok=True)
        res_df.to_csv(os.path.join(self.results_dir, "backtests", f"bt_{safe_fund_name}.csv"), index=False)
        return res_df

if __name__ == "__main__":
    tester = MUFAPBacktester()
    tester.run_backtest("Meezan Islamic Fund")
    print("-" * 50)
    tester.run_backtest("UBL Asset Allocation Fund")
