import os
import json
import torch
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import Booster
from catboost import CatBoostRegressor

import shap
from src.commodities.inference_utils import LSTMModel, GRUModel, TransformerModel, NBeatsModel, TFTModel
from src.commodities.live_data_pipeline import get_live_features

# Configuration
MODELS_DIR = './models/commodities/models_production'
RESULTS_DIR = './results/commodities'
DATA_DIR = './data/commodities'

SEQ_LENGTH = 10
HIDDEN_SIZE = 32

def get_top_models(commodity, horizon, top_n=3):
    """Reads the JSON metrics and returns the top N models based on Directional Accuracy."""
    ml_path = os.path.join(RESULTS_DIR, f'stage2_ml_metrics_{horizon}d.json')
    dl_path = os.path.join(RESULTS_DIR, f'stage2_dl_metrics_{horizon}d.json')
    
    if not os.path.exists(ml_path) or not os.path.exists(dl_path):
        raise ValueError(f"Metrics not found for horizon {horizon}d. Did you run the training loop?")
        
    with open(ml_path, 'r') as f:
        ml_data = json.load(f)
    with open(dl_path, 'r') as f:
        dl_data = json.load(f)
        
    all_models = {}
    
    # ML Models
    if commodity in ml_data:
        for model_name, metrics in ml_data[commodity].items():
            all_models[model_name] = {'acc': metrics['Dir_Acc'], 'mae': metrics['MAE']}
            
    # DL Models
    if commodity in dl_data:
        for model_name, metrics in dl_data[commodity].items():
            all_models[model_name] = {'acc': metrics['Dir_Acc'], 'mae': metrics['MAE']}
            
    if not all_models:
        raise ValueError(f"Commodity {commodity} not found in metrics files.")
        
    # Sort models by Directional Accuracy descending
    sorted_models = sorted(all_models.items(), key=lambda item: item[1]['acc'], reverse=True)
    return [(m, data['acc'], data['mae']) for m, data in sorted_models[:top_n]]

def prepare_data(commodity, horizon):
    """Loads CSV data or live data, extracts final features, and prepares inputs for ML and DL."""
    try:
        df_live_seq, current_price = get_live_features(commodity)
        features = list(df_live_seq.columns)
        X_raw = df_live_seq.values
        # Need to ensure X_raw has at least SEQ_LENGTH rows. 
        # The live_data_pipeline is configured to return exactly 10 rows.
    except Exception as e:
        print(f"Warning: Live data pipeline failed ({e}). Falling back to static CSV.")
        file_path = os.path.join(DATA_DIR, f"{commodity}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data for {commodity} not found.")
            
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        df = df.ffill()
        
        drop_cols = ['Target_Close_Next', 'Target_Return_Next', 'Target_Direction', 
                     'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        features = [col for col in df.columns if col not in drop_cols]
        features = [col for col in features if not col.startswith('Target_')]
        
        current_price = df['Close'].iloc[-1]
        X_raw = df[features].values
    
    # ML Input (1D vector of the very last row)
    X_ml = X_raw[-1:]
    
    # DL Input (Sequences of last SEQ_LENGTH rows)
    X_dl_raw = X_raw[-SEQ_LENGTH:]
    
    # If we don't have enough data for a sequence, we cannot run DL models.
    if len(X_dl_raw) < SEQ_LENGTH:
        raise ValueError(f"Not enough data to create a sequence of length {SEQ_LENGTH}")
        
    # Load Scaler
    scaler_path = os.path.join(MODELS_DIR, f'{commodity}_scaler_{horizon}d.pkl')
    # Fallback to general scaler if horizon-specific doesn't exist
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(MODELS_DIR, f'{commodity}_scaler.pkl')
        
    scaler = joblib.load(scaler_path)
    X_dl_scaled = scaler.transform(X_dl_raw)
    
    # Add batch dimension [1, 10, num_features]
    X_dl_tensor = torch.tensor(X_dl_scaled, dtype=torch.float32).unsqueeze(0)
    
    return X_ml, X_dl_tensor, len(features), current_price, features

def load_and_predict(model_name, commodity, horizon, X_ml, X_dl_tensor, input_size):
    """Loads the specific artifact from disk and predicts the return."""
    device = torch.device('cpu')
    pred_return = None
    
    # XGBoost
    if model_name == 'XGBoost':
        model = XGBRegressor()
        model.load_model(os.path.join(MODELS_DIR, f'{commodity}_xgboost_{horizon}d.json'))
        try:
            pred_return = model.predict(X_ml)[0]
        except ValueError:
            # Fallback for reverted champions expecting 27 features
            pred_return = model.predict(X_ml[:, :27])[0]
        
    # LightGBM
    elif model_name == 'LightGBM':
        model = Booster(model_file=os.path.join(MODELS_DIR, f'{commodity}_lightgbm_{horizon}d.txt'))
        if model.num_feature() == 27 and X_ml.shape[1] > 27:
            X_ml = X_ml[:, :27]
        pred_return = model.predict(X_ml)[0]
        
    # CatBoost
    elif model_name == 'CatBoost':
        model = CatBoostRegressor()
        model.load_model(os.path.join(MODELS_DIR, f'{commodity}_catboost_{horizon}d.cbm'))
        if len(model.feature_names_) == 27 and X_ml.shape[1] > 27:
            X_ml = X_ml[:, :27]
        pred_return = model.predict(X_ml)[0]
        
    # RandomForest
    elif model_name == 'RandomForest':
        model = joblib.load(os.path.join(MODELS_DIR, f'{commodity}_randomforest_{horizon}d.pkl'))
        if model.n_features_in_ == 27 and X_ml.shape[1] > 27:
            X_ml = X_ml[:, :27]
        pred_return = model.predict(X_ml)[0]
        
    # Deep Learning PyTorch Models
    else:
        model_path = os.path.join(MODELS_DIR, f'{commodity}_{model_name.lower().replace("-","")}_{horizon}d.pt')
        state_dict = torch.load(model_path, map_location=device)
        
        # Dynamically detect if this is a reverted champion expecting 27 features
        expected_input = input_size
        if 'lstm.weight_ih_l0' in state_dict:
            expected_input = state_dict['lstm.weight_ih_l0'].shape[1]
        elif 'gru.weight_ih_l0' in state_dict:
            expected_input = state_dict['gru.weight_ih_l0'].shape[1]
        elif 'input_proj.weight' in state_dict:
            expected_input = state_dict['input_proj.weight'].shape[1]
        elif 'block1.fc1.weight' in state_dict:
            expected_input = state_dict['block1.fc1.weight'].shape[1] // SEQ_LENGTH
            
        if expected_input == 27 and input_size > 27:
            X_dl_tensor = X_dl_tensor[:, :, :27]
            dynamic_input_size = 27
        else:
            dynamic_input_size = input_size

        if model_name == 'LSTM':
            model = LSTMModel(dynamic_input_size, HIDDEN_SIZE)
        elif model_name == 'GRU':
            model = GRUModel(dynamic_input_size, HIDDEN_SIZE)
        elif model_name == 'Transformer':
            model = TransformerModel(dynamic_input_size, HIDDEN_SIZE)
        elif model_name == 'N-BEATS':
            model = NBeatsModel(SEQ_LENGTH, dynamic_input_size)
        elif model_name == 'TFT':
            model = TFTModel(dynamic_input_size, HIDDEN_SIZE)
        else:
            raise ValueError(f"Unknown model architecture: {model_name}")
            
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            pred_return = model(X_dl_tensor).squeeze().item()
            
    return pred_return, model

def get_latest_news_url(commodity):
    """Retrieves the URL of the most recent live news article."""
    try:
        live_csv = os.path.join(DATA_DIR, f"{commodity}_news_live.csv")
        if os.path.exists(live_csv):
            df_news = pd.read_csv(live_csv)
            if not df_news.empty and 'URL' in df_news.columns:
                return df_news.iloc[-1]['URL'], df_news.iloc[-1]['Headline']
    except:
        pass
    return None, None

def calculate_shap_drivers(model_name, model, X_ml, features, commodity):
    """Uses SHAP to mathematically determine the top 3 drivers of the prediction."""
    drivers = []
    try:
        # SHAP only supports TreeExplainer for these specific models out of the box
        if model_name in ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest']:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_ml)
            
            # For a single prediction, shap_values is a 1D array
            if len(shap_values.shape) > 1:
                shap_vals = shap_values[0]
            else:
                shap_vals = shap_values
                
            # Zip features with their SHAP impact values
            feature_impacts = list(zip(features, shap_vals))
            # Sort by absolute impact (magnitude)
            feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            
            latest_url, latest_headline = get_latest_news_url(commodity)
            
            for feat, impact in feature_impacts[:3]:
                driver = {
                    'feature': feat,
                    'impact': impact,
                    'type': 'Bullish 🟢' if impact > 0 else 'Bearish 🔴'
                }
                
                # If the feature is sentiment-related, attach the news URL
                if 'Sentiment' in feat or 'Daily_NSS' in feat or 'News_Volume' in feat:
                    if latest_url:
                        driver['url'] = latest_url
                        driver['headline'] = latest_headline
                        
                drivers.append(driver)
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        
    return drivers

def ensemble_predict(commodity, horizon, top_n=3):
    """
    Main function: gets top models, processes data, runs inferences, and computes accuracy-weighted average.
    """
    # 1. Get Top Models
    top_models = get_top_models(commodity, horizon, top_n)
    print(f"\\n{'='*50}\\nENSEMBLE INFERENCE: {commodity.upper()} ({horizon}-Day)\\n{'='*50}")
    print(f"Selected Top {top_n} Models Based on Directional Accuracy:")
    for m, acc, mae in top_models:
        print(f" - {m}: {acc}% (MAE: {mae:.4f})")
        
    # 2. Prepare Data
    X_ml, X_dl_tensor, input_size, current_price, features = prepare_data(commodity, horizon)
    print(f"Current Price (Today): ${current_price:.4f}")
    
    # 3. Run Inference
    total_weight = 0
    weighted_return = 0
    predictions = {}
    
    top_drivers = []
    
    total_mae = 0
    
    for model_name, acc, mae in top_models:
        try:
            pred_ret, model_obj = load_and_predict(model_name, commodity, horizon, X_ml, X_dl_tensor, input_size)
            predictions[model_name] = pred_ret
            # Weight is the historical accuracy
            weighted_return += pred_ret * acc
            total_weight += acc
            total_mae += mae * acc
            print(f"[{model_name}] Predicted Return: {pred_ret:.4f} (Implied Price: ${current_price * (1 + pred_ret):.4f})")
            
            # Calculate drivers from the highest-accuracy tree model
            if not top_drivers and model_name in ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest']:
                top_drivers = calculate_shap_drivers(model_name, model_obj, X_ml, features, commodity)
                
        except Exception as e:
            print(f"⚠️ Error loading {model_name}: {str(e)}")
            continue
            
    # 4. Math Consensus
    if total_weight == 0:
        raise RuntimeError("Ensemble failed: No models were successfully loaded.")
        
    final_return = weighted_return / total_weight
    final_price = current_price * (1 + final_return)
    direction = "UP 🟢" if final_return > 0 else "DOWN 🔴"
    
    # Calculate True Confidence Intervals using historical MAE
    final_mae = total_mae / total_weight
    predicted_min = final_price - final_mae
    predicted_max = final_price + final_mae
    
    confidence_pct = total_weight / len(predictions) if len(predictions) > 0 else 0
    
    print(f"\\n✅ FINAL ENSEMBLE PREDICTION ({horizon} Days Out)")
    print(f"Consensus Return : {final_return * 100:.2f}%")
    print(f"Target Price     : ${final_price:.4f}")
    print(f"Direction        : {direction}")
    
    return {
        'commodity': commodity,
        'horizon': horizon,
        'current_price': current_price,
        'predicted_price': final_price,
        'predicted_return': final_return,
        'predicted_min': predicted_min,
        'predicted_max': predicted_max,
        'confidence_pct': confidence_pct,
        'direction': direction,
        'models_used': [m[0] for m in top_models],
        'market_drivers': top_drivers
    }

if __name__ == "__main__":
    # Test Block: Run an ensemble for Gold (60-day horizon) and Natural Gas (14-day horizon)
    print("Running Ensemble Backend Test...")
    ensemble_predict('gold', 60, top_n=3)
    ensemble_predict('natural_gas', 14, top_n=2)
    ensemble_predict('crude_oil', 120, top_n=3)
