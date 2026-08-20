import json
import os
import subprocess
import shutil

horizons = ['7d', '14d', '28d', '42d', '60d', '90d', '120d']
models = ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 'LSTM', 'GRU', 'Transformer', 'N-BEATS', 'TFT']

def get_filename(comm, model, h):
    m_lower = model.lower()
    if model == 'XGBoost': return f"{comm}_xgboost_{h}.json"
    if model == 'LightGBM': return f"{comm}_lightgbm_{h}.txt"
    if model == 'CatBoost': return f"{comm}_catboost_{h}.cbm"
    if model == 'RandomForest': return f"{comm}_randomforest_{h}.pkl"
    if model in ['LSTM', 'GRU', 'Transformer', 'TFT']: return f"{comm}_{m_lower}_{h}.pt"
    if model == 'N-BEATS': return f"{comm}_nbeats_{h}.pt"
    return None

def get_git_json(file_path):
    try:
        result = subprocess.run(['git', 'show', f'HEAD:{file_path}'], capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except Exception:
        return None

def load_local_json(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except:
        return None

reverted_count = 0
total_compared = 0

for h in horizons:
    ml_file = f'results/stage2_ml_metrics_{h}.json'
    dl_file = f'results/stage2_dl_metrics_{h}.json'
    
    ml_old = get_git_json(ml_file) or {}
    ml_new = load_local_json(ml_file) or {}
    dl_old = get_git_json(dl_file) or {}
    dl_new = load_local_json(dl_file) or {}
    
    old_all = {**ml_old}
    for comm, data in dl_old.items():
        if comm not in old_all: old_all[comm] = {}
        old_all[comm].update(data)
        
    new_all = {**ml_new}
    for comm, data in dl_new.items():
        if comm not in new_all: new_all[comm] = {}
        new_all[comm].update(data)
        
    for comm in old_all.keys():
        for model in models:
            if model in old_all[comm] and model in new_all.get(comm, {}):
                total_compared += 1
                old_diracc = old_all[comm][model].get('Dir_Acc', 0)
                new_diracc = new_all[comm][model].get('Dir_Acc', 0)
                
                # If degraded, REVERT
                if new_diracc < old_diracc:
                    filename = get_filename(comm, model, h)
                    if filename:
                        backup_path = os.path.join('models_champion_backup', filename)
                        active_path = os.path.join('models', filename)
                        
                        if os.path.exists(backup_path):
                            shutil.copy(backup_path, active_path)
                            print(f"[REVERTED] {comm.upper()} | {h} | {model} (Acc dropped from {old_diracc}% to {new_diracc}%)")
                            reverted_count += 1

print(f"\n--- REVERSION COMPLETE ---")
print(f"Total Models Compared: {total_compared}")
print(f"Total Models Reverted to Champions: {reverted_count}")
