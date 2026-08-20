import json
import os
import subprocess
import pandas as pd

horizons = ['7d', '14d', '28d', '42d', '60d', '90d', '120d']
metrics_to_track = ['Dir_Acc', 'MAE', 'R2']
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
    
    # Load old and new
    ml_old = get_git_json(ml_file) or {}
    ml_new = load_local_json(ml_file) or {}
    dl_old = get_git_json(dl_file) or {}
    dl_new = load_local_json(dl_file) or {}
    
    # Combine old and new for the horizon
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
                    'Horizon': h,
                    'Commodity': comm,
                    'Model': model,
                    'Old_DirAcc': old_metrics.get('Dir_Acc', 0),
                    'New_DirAcc': new_metrics.get('Dir_Acc', 0),
                    'Diff_DirAcc': new_metrics.get('Dir_Acc', 0) - old_metrics.get('Dir_Acc', 0),
                    'Old_MAE': old_metrics.get('MAE', 0),
                    'New_MAE': new_metrics.get('MAE', 0),
                    'Diff_MAE': old_metrics.get('MAE', 0) - new_metrics.get('MAE', 0) # Positive means improvement (lower MAE)
                })

df = pd.DataFrame(comparison_data)

# Print Summary Markdown
print("# Sentiment Integration: Champion vs Challenger Benchmark Results\n")
print("## 1. Overall System Impact\n")

# Average Improvement
avg_diracc_imp = df['Diff_DirAcc'].mean()
avg_mae_imp = df['Diff_MAE'].mean()

print(f"**Average Directional Accuracy Shift:** {'+' if avg_diracc_imp > 0 else ''}{avg_diracc_imp:.2f}%\n")
print(f"**Average MAE Improvement (Error Reduction):** {'+' if avg_mae_imp > 0 else ''}{avg_mae_imp:.2f}\n")

print("## 2. Impact by Horizon\n")
horizon_impact = df.groupby('Horizon')[['Diff_DirAcc', 'Diff_MAE']].mean().sort_index()
print("| Horizon | Directional Accuracy Change (%) | MAE Improvement (Lower Error) |")
print("|---------|---------------------------------|-------------------------------|")
for idx, row in horizon_impact.iterrows():
    print(f"| {idx} | {row['Diff_DirAcc']:.2f}% | {row['Diff_MAE']:.4f} |")

print("\n## 3. Top 5 Most Improved Models (The New Champions)\n")
top_5 = df.sort_values('Diff_DirAcc', ascending=False).head(5)
print("| Commodity | Horizon | Model | Old DirAcc | New DirAcc | Improvement |")
print("|-----------|---------|-------|------------|------------|-------------|")
for _, row in top_5.iterrows():
    print(f"| {row['Commodity']} | {row['Horizon']} | {row['Model']} | {row['Old_DirAcc']}% | {row['New_DirAcc']}% | +{row['Diff_DirAcc']:.2f}% |")

print("\n## 4. Top 5 Degraded Models (Revert to Old Champions)\n")
bot_5 = df.sort_values('Diff_DirAcc', ascending=True).head(5)
print("| Commodity | Horizon | Model | Old DirAcc | New DirAcc | Degradation |")
print("|-----------|---------|-------|------------|------------|-------------|")
for _, row in bot_5.iterrows():
    print(f"| {row['Commodity']} | {row['Horizon']} | {row['Model']} | {row['Old_DirAcc']}% | {row['New_DirAcc']}% | {row['Diff_DirAcc']:.2f}% |")

