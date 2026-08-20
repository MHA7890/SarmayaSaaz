import os
import json
import shutil

MODELS_DIR = './models'
PROD_DIR = './models_production'
RESULTS_DIR = './results'

COMMODITIES = ['gold', 'silver', 'copper', 'natural_gas', 'crude_oil', 'wheat']
HORIZONS = [1, 7, 14, 28, 42, 60, 90, 120]

os.makedirs(PROD_DIR, exist_ok=True)

def get_top_models(commodity, horizon, top_n=3):
    ml_path = os.path.join(RESULTS_DIR, f'stage2_ml_metrics_{horizon}d.json')
    dl_path = os.path.join(RESULTS_DIR, f'stage2_dl_metrics_{horizon}d.json')
    
    with open(ml_path, 'r') as f: ml_data = json.load(f)
    with open(dl_path, 'r') as f: dl_data = json.load(f)
        
    all_models = {}
    if commodity in ml_data:
        for model_name, metrics in ml_data[commodity].items():
            all_models[model_name] = metrics['Dir_Acc']
            
    if commodity in dl_data:
        for model_name, metrics in dl_data[commodity].items():
            all_models[model_name] = metrics['Dir_Acc']
            
    sorted_models = sorted(all_models.items(), key=lambda item: item[1], reverse=True)
    return [m[0] for m in sorted_models[:top_n]]

def copy_file_safe(src, dst):
    if os.path.exists(src):
        shutil.copy2(src, dst)
        return True
    return False

copied_count = 0

for commodity in COMMODITIES:
    for horizon in HORIZONS:
        # Copy scalers
        scaler_specific = os.path.join(MODELS_DIR, f'{commodity}_scaler_{horizon}d.pkl')
        scaler_general = os.path.join(MODELS_DIR, f'{commodity}_scaler.pkl')
        
        if copy_file_safe(scaler_specific, os.path.join(PROD_DIR, f'{commodity}_scaler_{horizon}d.pkl')):
            copied_count += 1
        elif copy_file_safe(scaler_general, os.path.join(PROD_DIR, f'{commodity}_scaler.pkl')):
            copied_count += 1
            
        top_models = get_top_models(commodity, horizon, 3)
        for model_name in top_models:
            filename = ""
            if model_name == 'XGBoost':
                filename = f'{commodity}_xgboost_{horizon}d.json'
            elif model_name == 'LightGBM':
                filename = f'{commodity}_lightgbm_{horizon}d.txt'
            elif model_name == 'CatBoost':
                filename = f'{commodity}_catboost_{horizon}d.cbm'
            elif model_name == 'RandomForest':
                filename = f'{commodity}_randomforest_{horizon}d.pkl'
            else:
                m_clean = model_name.lower().replace("-", "")
                filename = f'{commodity}_{m_clean}_{horizon}d.pt'
                
            src = os.path.join(MODELS_DIR, filename)
            dst = os.path.join(PROD_DIR, filename)
            if copy_file_safe(src, dst):
                copied_count += 1

print(f"Successfully copied {copied_count} essential model/scaler files to {PROD_DIR}.")
