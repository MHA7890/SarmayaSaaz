import os
import json
import logging
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import ta
import torch
import torch.nn as nn
from datetime import datetime

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_DIR = "models/stocks"
HORIZONS = [7, 14, 28, 42, 60, 90, 120]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mapping Dictionary (Same as Stage 3)
SECTOR_MAP = {
    'Financials': ['ABL', 'AHCL', 'AICL', 'AKBL', 'BAFL', 'BAHL', 'BOP', 'FABL', 'HBL', 'HMB', 'MCB', 'MEBL', 'NBP', 'PSX', 'SCBPL', 'UBL'],
    'Energy_Power': ['APL', 'ATRL', 'CNERGY', 'HUBC', 'KAPCO', 'KEL', 'MARI', 'OGDC', 'POL', 'PPL', 'PSO', 'SNGP', 'SSGC'],
    'Cement_Construction': ['BWCL', 'CHCC', 'DGKC', 'FCCL', 'KOHC', 'LUCK', 'MLCF', 'PIOC', 'POWER', 'THALL', 'ISL', 'INIL'],
    'Tech_Telecom': ['AIRLINK', 'NETSOL', 'PTC', 'SYS', 'TRG'],
    'Pharmaceuticals': ['ABOT', 'AGP', 'CPHL', 'GLAXO', 'HALEON', 'HINOON', 'IBFL', 'SEARL', 'SHFA'],
    'Fertilizers_Chemicals': ['COLG', 'EFERT', 'ENGROH', 'FATIMA', 'FFC', 'GHGL', 'LCI', 'LOTCHEM', 'PABC', 'EPCL'],
    'Consumer_Autos': ['ATLH', 'BNWM', 'FFL', 'GADT', 'GAL', 'GHNI', 'HCAR', 'HUMNL', 'ILP', 'INDU', 'JDWS', 'JVDC', 'KTML', 'MEHT', 'MTL', 'MUREB', 'NATF', 'NESTLE', 'NML', 'PAEL', 'PAKT', 'PGLC', 'PIBTL', 'PKGS', 'PSEL', 'RMPL', 'SAZEW', 'SRVI', 'SSOM', 'TGL', 'UNITY', 'UPFL', 'YOUW']
}
TICKER_TO_SECTOR = {}
for sector, tickers in SECTOR_MAP.items():
    for ticker in tickers:
        TICKER_TO_SECTOR[ticker] = sector

# --- Re-define LSTM Architecture ---
class PSX_LSTM(nn.Module):
    def __init__(self, input_dim):
        super(PSX_LSTM, self).__init__()
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

class PSXPredictor:
    def __init__(self):
        with open(os.path.join(MODEL_DIR, "ensemble_weights.json"), "r") as f:
            self.routing_table = json.load(f)
            
    def _fetch_and_clean(self, symbol):
        df = yf.download(symbol, period="300d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if 'Adj Close' not in df.columns:
            df['Adj Close'] = df['Close']
        df.ffill(inplace=True)
        return df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

    def _engineer_macro(self, df, prefix):
        df[f'{prefix}_Log_Return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
        sma_200 = ta.trend.sma_indicator(df['Adj Close'], window=200)
        df[f'{prefix}_SMA_200_Ratio'] = df['Adj Close'] / sma_200
        df[f'{prefix}_Volatility_30d'] = df[f'{prefix}_Log_Return'].rolling(window=30).std()
        return df[[f'{prefix}_Log_Return', f'{prefix}_SMA_200_Ratio', f'{prefix}_Volatility_30d']]

    def predict(self, base_ticker):
        ticker = base_ticker.upper()
        sector = TICKER_TO_SECTOR.get(ticker, "Diversified")
        logger.info(f"Initializing Inference for: {ticker}")
        logger.info(f"Routed to Super-Sector: {sector}")
        
        # 1. Fetch Live Data
        df = self._fetch_and_clean(f"{ticker}.KA")
        if df is None:
            logger.error(f"Failed to fetch live data for {ticker}.KA")
            return None
            
        current_price = df['Adj Close'].iloc[-1]
        latest_date = df.index[-1].strftime('%Y-%m-%d')
        logger.info(f"Latest Data Date: {latest_date} | Current Price: Rs {current_price:.2f}")

        # 2. Re-create Stage 2 Features
        df['SMA_20'] = ta.trend.sma_indicator(df['Adj Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Adj Close'], window=50)
        df['SMA_200'] = ta.trend.sma_indicator(df['Adj Close'], window=200)
        df['Close_to_SMA_20'] = df['Adj Close'] / df['SMA_20']
        df['Close_to_SMA_50'] = df['Adj Close'] / df['SMA_50']
        df['Close_to_SMA_200'] = df['Adj Close'] / df['SMA_200']
        df['RSI_14'] = ta.momentum.rsi(df['Adj Close'], window=14)
        macd = ta.trend.MACD(df['Adj Close'])
        df['MACD_Diff'] = macd.macd_diff()
        for days in [1, 3, 7, 14]:
            df[f'Log_Return_{days}d'] = np.log(df['Adj Close'] / df['Adj Close'].shift(days))
            
        bollinger = ta.volatility.BollingerBands(df['Adj Close'], window=20, window_dev=2)
        df['BB_Width'] = (bollinger.bollinger_hband() - bollinger.bollinger_lband()) / df['SMA_20']
        atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Adj Close'], window=14)
        df['ATR_Normalized'] = atr.average_true_range() / df['Adj Close']
        df['Volatility_30d'] = df['Log_Return_1d'].rolling(window=30).std()
        
        df['SMA_Volume_20'] = df['Volume'].rolling(window=20).mean()
        safe_sma = df['SMA_Volume_20'].replace(0, 1)
        df['Volume_Surge'] = df['Volume'] / safe_sma
        obv = ta.volume.OnBalanceVolumeIndicator(df['Adj Close'], df['Volume']).on_balance_volume()
        obv_sma = obv.rolling(window=20).mean().replace(0, 1)
        df['OBV_Momentum'] = obv / obv_sma

        # 3. Re-create Stage 3 Macro Features
        pkr_df = self._engineer_macro(self._fetch_and_clean("PKR=X"), "Currency")
        df = df.join(pkr_df, how='left')
        
        if sector == "Energy_Power":
            oil_df = self._engineer_macro(self._fetch_and_clean("CL=F"), "Oil")
            df = df.join(oil_df, how='left')
        elif sector == "Tech_Telecom":
            nasdaq_df = self._engineer_macro(self._fetch_and_clean("^IXIC"), "Tech")
            df = df.join(nasdaq_df, how='left')
            
        df.ffill(inplace=True)
        
        # 4. Anti-Leakage Drop
        drop_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SMA_20', 'SMA_50', 'SMA_200', 'SMA_Volume_20']
        df.drop(columns=drop_cols, inplace=True)
        
        # Extract 'Today'
        X_live_raw = df.iloc[-1:].values
        
        # 5. Apply Frozen Scaler
        scaler_path = os.path.join(MODEL_DIR, f"{sector}_scaler.pkl")
        scaler = joblib.load(scaler_path)
        X_live_scaled = scaler.transform(X_live_raw)
        
        # 6. Inference Loop via JSON Router
        results = {
            "Ticker": ticker,
            "Sector": sector,
            "Latest_Date": latest_date,
            "Current_Price": current_price,
            "Predictions": {}
        }
        
        for horizon in HORIZONS:
            route = self.routing_table[sector][f"{horizon}d"]
            model_name = route["Winning_Model"]
            model_path = os.path.join(MODEL_DIR, route["Path"])
            
            if model_name == "LSTM":
                model = PSX_LSTM(X_live_scaled.shape[1]).to(DEVICE)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
                model.eval()
                with torch.no_grad():
                    pred_return = model(torch.FloatTensor(X_live_scaled).to(DEVICE)).cpu().numpy()[0]
            else:
                model = joblib.load(model_path)
                pred_return = model.predict(X_live_scaled)[0]
                
            projected_price = current_price * (1 + pred_return)
            
            results["Predictions"][f"{horizon}d"] = {
                "Predicted_Return_Pct": round(float(pred_return * 100), 2),
                "Projected_Price": round(float(projected_price), 4),
                "Model_Used": model_name
            }
            
        print("\n" + json.dumps(results, indent=4))
        return results

if __name__ == "__main__":
    engine = PSXPredictor()
    print("="*50)
    engine.predict("SYS")
    print("="*50)
    engine.predict("OGDC")
    print("="*50)
    engine.predict("MEBL")
    print("="*50)
