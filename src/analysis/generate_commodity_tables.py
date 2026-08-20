import json
import pandas as pd
import subprocess

horizons = ['7d', '14d', '28d', '42d', '60d', '90d', '120d']
models = ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 'LSTM', 'GRU', 'Transformer', 'N-BEATS', 'TFT']
commodities = ['gold', 'silver', 'copper', 'natural_gas', 'crude_oil', 'wheat']

def get_git_json(file_path):
    try:
        result = subprocess.run(['git', 'show', f'HEAD:{file_path}'], capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except Exception:
        return {}

def load_local_json(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except:
        return {}

old_data = []
new_data = []

for h in horizons:
    ml_old = get_git_json(f'results/stage2_ml_metrics_{h}.json') or {}
    dl_old = get_git_json(f'results/stage2_dl_metrics_{h}.json') or {}
    ml_new = load_local_json(f'results/stage2_ml_metrics_{h}.json') or {}
    dl_new = load_local_json(f'results/stage2_dl_metrics_{h}.json') or {}
    
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
            if model in old_all.get(comm, {}):
                old_data.append({
                    'Commodity': comm.upper(),
                    'Horizon': h,
                    'Model': model,
                    'DirAcc': old_all[comm][model].get('Dir_Acc', 0),
                    'MAE': old_all[comm][model].get('MAE', 0)
                })
            if model in new_all.get(comm, {}):
                new_data.append({
                    'Commodity': comm.upper(),
                    'Horizon': h,
                    'Model': model,
                    'DirAcc': new_all[comm][model].get('Dir_Acc', 0),
                    'MAE': new_all[comm][model].get('MAE', 0)
                })

df_old = pd.DataFrame(old_data)
df_new = pd.DataFrame(new_data)

md_out = "\n\n### 10. Ultimate Champions per Commodity (Pre vs. Post Sentiment)\n\n"
md_out += "This section breaks down the absolute best performing architecture for every horizon, before and after sentiment integration.\n\n"

for comm in sorted([c.upper() for c in commodities]):
    md_out += f"#### {comm}\n\n"
    md_out += "| Horizon | Best Pre-Sentiment Model | Pre DirAcc | Pre MAE | Best Post-Sentiment Model | Post DirAcc | Post MAE |\n"
    md_out += "|---|---|---|---|---|---|---|\n"
    
    df_old_comm = df_old[df_old['Commodity'] == comm]
    df_new_comm = df_new[df_new['Commodity'] == comm]
    
    for h in horizons:
        df_old_h = df_old_comm[df_old_comm['Horizon'] == h]
        df_new_h = df_new_comm[df_new_comm['Horizon'] == h]
        
        if df_old_h.empty or df_new_h.empty: continue
        
        best_old = df_old_h.loc[df_old_h['DirAcc'].idxmax()]
        best_new = df_new_h.loc[df_new_h['DirAcc'].idxmax()]
        
        md_out += f"| {h} | **{best_old['Model']}** | {best_old['DirAcc']:.2f}% | {best_old['MAE']:.4f} | **{best_new['Model']}** | {best_new['DirAcc']:.2f}% | {best_new['MAE']:.4f} |\n"
    md_out += "\n"

with open('project_overview.md', 'a') as f:
    f.write(md_out)
