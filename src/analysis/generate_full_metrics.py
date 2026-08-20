import json
import os
import subprocess
import pandas as pd

horizons = ['7d', '14d', '28d', '42d', '60d', '90d', '120d']
models = ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 'LSTM', 'GRU', 'Transformer', 'N-BEATS', 'TFT']

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

comparison_data = []

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
                old_metrics = old_all[comm][model]
                new_metrics = new_all[comm][model]
                
                comparison_data.append({
                    'Commodity': comm,
                    'Horizon': h,
                    'Model': model,
                    'Old_DirAcc': old_metrics.get('Dir_Acc', 0),
                    'New_DirAcc': new_metrics.get('Dir_Acc', 0),
                    'Diff_DirAcc': new_metrics.get('Dir_Acc', 0) - old_metrics.get('Dir_Acc', 0),
                    'Old_MAE': old_metrics.get('MAE', 0),
                    'New_MAE': new_metrics.get('MAE', 0),
                    'Diff_MAE': old_metrics.get('MAE', 0) - new_metrics.get('MAE', 0) 
                })

df = pd.DataFrame(comparison_data)
df = df.sort_values(by=['Commodity', 'Horizon', 'Model'])

print("### 5. Full Metrics Comparison (Pre-Sentiment vs Post-Sentiment)")
print("<details>")
print("<summary>Click to view the full performance table for all models</summary>\n")

print("| Commodity | Horizon | Model | Pre-Sentiment DirAcc | Post-Sentiment DirAcc | DirAcc Diff | Pre-Sentiment MAE | Post-Sentiment MAE | MAE Diff (Higher is Better) |")
print("|---|---|---|---|---|---|---|---|---|")

for _, row in df.iterrows():
    dir_acc_diff_str = f"+{row['Diff_DirAcc']:.2f}%" if row['Diff_DirAcc'] > 0 else f"{row['Diff_DirAcc']:.2f}%"
    mae_diff_str = f"+{row['Diff_MAE']:.4f}" if row['Diff_MAE'] > 0 else f"{row['Diff_MAE']:.4f}"
    
    print(f"| {row['Commodity'].upper()} | {row['Horizon']} | {row['Model']} | {row['Old_DirAcc']:.2f}% | {row['New_DirAcc']:.2f}% | {dir_acc_diff_str} | {row['Old_MAE']:.4f} | {row['New_MAE']:.4f} | {mae_diff_str} |")

print("\n</details>")
