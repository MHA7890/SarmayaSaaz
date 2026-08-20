import json
import pandas as pd

horizons = ['7d', '14d', '28d', '42d', '60d', '90d', '120d']
models = ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 'LSTM', 'GRU', 'Transformer', 'N-BEATS', 'TFT']

def load_local_json(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except:
        return {}

comparison_data = []

for h in horizons:
    ml_new = load_local_json(f'results/stage2_ml_metrics_{h}.json')
    dl_new = load_local_json(f'results/stage2_dl_metrics_{h}.json')
    
    new_all = {**ml_new}
    for comm, data in dl_new.items():
        if comm not in new_all: new_all[comm] = {}
        new_all[comm].update(data)
        
    for comm in new_all.keys():
        for model in models:
            if model in new_all.get(comm, {}):
                metrics = new_all[comm][model]
                comparison_data.append({
                    'Commodity': comm,
                    'Horizon': h,
                    'Model': model,
                    'DirAcc': metrics.get('Dir_Acc', 0)
                })

df = pd.DataFrame(comparison_data)

# 1. Best Commodities (Which was easiest to predict?)
comm_avg = df.groupby('Commodity')['DirAcc'].mean().sort_values(ascending=False).reset_index()
comm_md = "### 6. Predictability by Commodity (Easiest to Hardest)\n\n"
comm_md += "| Rank | Commodity | Average Directional Accuracy |\n|---|---|---|\n"
for i, row in comm_avg.iterrows():
    comm_md += f"| {i+1} | **{row['Commodity'].upper()}** | {row['DirAcc']:.2f}% |\n"

# 2. Best Models (Which architectures performed best overall?)
model_avg = df.groupby('Model')['DirAcc'].mean().sort_values(ascending=False).reset_index()
model_md = "\n### 7. Global Model Performance\n\n"
model_md += "| Rank | Architecture | Average Directional Accuracy |\n|---|---|---|\n"
for i, row in model_avg.iterrows():
    model_md += f"| {i+1} | **{row['Model']}** | {row['DirAcc']:.2f}% |\n"

# 3. Best Horizons (When is the market most predictable?)
horizon_avg = df.groupby('Horizon')['DirAcc'].mean().sort_values(ascending=False).reset_index()
horizon_md = "\n### 8. Predictability by Time Horizon\n\n"
horizon_md += "| Rank | Horizon | Average Directional Accuracy |\n|---|---|---|\n"
for i, row in horizon_avg.iterrows():
    horizon_md += f"| {i+1} | **{row['Horizon']}** | {row['DirAcc']:.2f}% |\n"

# 4. The "Holy Grail" Combinations (Best specific setups)
top_combos = df.sort_values(by='DirAcc', ascending=False).head(10).reset_index()
combo_md = "\n### 9. Top 10 Best Individual Models (The Holy Grails)\n\n"
combo_md += "| Rank | Commodity | Horizon | Model | Directional Accuracy |\n|---|---|---|---|---|\n"
for i, row in top_combos.iterrows():
    combo_md += f"| {i+1} | **{row['Commodity'].upper()}** | {row['Horizon']} | {row['Model']} | **{row['DirAcc']:.2f}%** |\n"

analysis_md = "\n\n---\n\n## Advanced Post-Sentiment Analysis\n\n" + comm_md + model_md + horizon_md + combo_md

with open('project_overview.md', 'a') as f:
    f.write(analysis_md)
