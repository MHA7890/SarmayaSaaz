import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Cell 1: Markdown
cells.append(nbf.v4.new_markdown_cell("""# Machine Learning Training (Tree-Based Models) - 90-Day Horizon
This notebook trains 4 tree-based models across 6 commodities for a **90-Day horizon**:
1. **XGBoost**
2. **LightGBM**
3. **CatBoost**
4. **Random Forest**

Target variables are dynamically shifted by 90 days. Anomaly clipping is set to [-80%, +150%]. Overfitting protections are strictly enforced to handle massive overlapping targets."""))

# Cell 2: Imports
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import os
import json
import warnings
import joblib
import plotly.express as px
import plotly.graph_objects as go
from tqdm.notebook import tqdm
from IPython.display import display, HTML, clear_output
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

# CONFIGURATION
HORIZON = 90

data_dir = './data'
results_dir = './results'
models_dir = './models'
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
commodities = ['gold', 'silver', 'copper', 'natural_gas', 'crude_oil', 'wheat']
"""))

# Cell 3: Load Data
cells.append(nbf.v4.new_code_cell("""datasets = {}
print("Loading and validating datasets...")
for name in commodities:
    file_path = os.path.join(data_dir, f"{name}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        datasets[name] = df
        print(f"✅ {name.upper()}: {df.shape} | Dates: {df.index.min().date()} to {df.index.max().date()} | NaNs: {df.isna().sum().sum()}")
    else:
        print(f"❌ {name.upper()}: Not found")
"""))

# Cell 4: Markdown Split strategy
cells.append(nbf.v4.new_markdown_cell("""### Train / Validation / Test Split Strategy
- **Train (70%)**: Used to fit the trees.
- **Validation (10%)**: Used for early stopping (preventing overfitting).
- **Test (20%)**: Completely unseen data used for final metrics.
"""))

# Cell 5: Metrics and Helpers
cells.append(nbf.v4.new_code_cell("""def calculate_metrics(y_true, y_pred, y_true_prev):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE handling zeros
    nonzero = y_true != 0
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    
    # R2 Score
    r2 = r2_score(y_true, y_pred)
    
    # Directional Accuracy
    actual_dir = np.sign(y_true - y_true_prev)
    pred_dir = np.sign(y_pred - y_true_prev)
    correct_dir = (actual_dir == pred_dir)
    dir_acc = np.mean(correct_dir) * 100
    
    # Naive Baseline (predict today's price for horizon)
    naive_mae = mean_absolute_error(y_true, y_true_prev)
    
    # Improvement vs Naive
    imp_pct = ((naive_mae - mae) / naive_mae) * 100 if naive_mae > 0 else 0
    
    return {
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'MAPE': round(mape, 2),
        'Dir_Acc': round(dir_acc, 2),
        'R2': round(r2, 4),
        'Naive_MAE': round(naive_mae, 4),
        'Improvement_Pct': round(imp_pct, 2)
    }

def get_train_val_test(df, horizon):
    drop_cols = ['Target_Close_Next', 'Target_Return_Next', 'Target_Direction', 
                 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    features = [col for col in df.columns if col not in drop_cols]
    
    df_copy = df.copy()
    
    # Dynamic 90-day shifting
    df_copy[f'Target_Close_{horizon}d'] = df_copy['Close'].shift(-horizon)
    df_copy[f'Target_Return_{horizon}d'] = ((df_copy[f'Target_Close_{horizon}d'] - df_copy['Close']) / df_copy['Close']).clip(lower=-0.8, upper=1.5)
    
    # Drop NaNs at the end caused by shifting
    df_copy = df_copy.dropna(subset=[f'Target_Return_{horizon}d'])
    
    X = df_copy[features]
    y_return = df_copy[f'Target_Return_{horizon}d']
    
    total_len = len(df_copy)
    train_end = int(total_len * 0.7)
    val_end = int(total_len * 0.8)
    
    X_train = X.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    X_test = X.iloc[val_end:]
    
    y_train = y_return.iloc[:train_end]
    y_val = y_return.iloc[train_end:val_end]
    y_test = y_return.iloc[val_end:]
    
    y_true_price_test = df_copy[f'Target_Close_{horizon}d'].iloc[val_end:].values
    y_true_prev_test = df_copy['Close'].iloc[val_end:].values
    
    return X_train, X_val, X_test, y_train, y_val, y_test, y_true_price_test, y_true_prev_test, features
"""))

# Cell 6: Training Loop
cells.append(nbf.v4.new_code_cell("""results = {}
feature_importances = {}

for name, df in tqdm(datasets.items(), desc="Commodities"):
    X_train, X_val, X_test, y_train, y_val, y_test, y_true_price, y_true_prev, features = get_train_val_test(df, HORIZON)
    
    eval_set_xgb_lgb = [(X_val, y_val)]
    
    print(f"\\n{'='*50}\\nTraining {name.upper()} ({HORIZON}-Day Horizon)...\\n{'='*50}")
    
    res = {}
    fi = {}
    
    # 1. XGBoost (Lowered LR for overlapping target generalization)
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.03, max_depth=5, subsample=0.8, 
                       colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=3.0, random_state=42,
                       early_stopping_rounds=20)
    xgb.fit(X_train, y_train, eval_set=eval_set_xgb_lgb, verbose=False)
    xgb_ret = xgb.predict(X_test)
    xgb_price = y_true_prev * (1 + xgb_ret)
    res['XGBoost'] = calculate_metrics(y_true_price, xgb_price, y_true_prev)
    xgb.save_model(os.path.join(models_dir, f'{name}_xgboost_{HORIZON}d.json'))
    fi['XGBoost'] = xgb.feature_importances_
    
    # 2. LightGBM (Lowered LR)
    lgb = LGBMRegressor(n_estimators=200, learning_rate=0.03, max_depth=5, subsample=0.8, 
                        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=3.0, random_state=42, verbose=-1)
    from lightgbm import early_stopping
    lgb.fit(X_train, y_train, eval_set=eval_set_xgb_lgb, callbacks=[early_stopping(stopping_rounds=20, verbose=False)])
    lgb_ret = lgb.predict(X_test)
    lgb_price = y_true_prev * (1 + lgb_ret)
    res['LightGBM'] = calculate_metrics(y_true_price, lgb_price, y_true_prev)
    lgb.booster_.save_model(os.path.join(models_dir, f'{name}_lightgbm_{HORIZON}d.txt'))
    fi['LightGBM'] = lgb.feature_importances_
    
    # 3. CatBoost (Lowered LR)
    cat = CatBoostRegressor(iterations=200, learning_rate=0.03, depth=5, l2_leaf_reg=5.0, 
                            random_seed=42, verbose=0, early_stopping_rounds=20)
    cat.fit(X_train, y_train, eval_set=(X_val, y_val))
    cat_ret = cat.predict(X_test)
    cat_price = y_true_prev * (1 + cat_ret)
    res['CatBoost'] = calculate_metrics(y_true_price, cat_price, y_true_prev)
    cat.save_model(os.path.join(models_dir, f'{name}_catboost_{HORIZON}d.cbm'))
    fi['CatBoost'] = cat.feature_importances_
    
    # 4. Random Forest (Increased min_samples_leaf to 10 for overlapping targets)
    rf = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=15, 
                               max_features='sqrt', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_ret = rf.predict(X_test)
    rf_price = y_true_prev * (1 + rf_ret)
    res['RandomForest'] = calculate_metrics(y_true_price, rf_price, y_true_prev)
    joblib.dump(rf, os.path.join(models_dir, f'{name}_randomforest_{HORIZON}d.pkl'))
    fi['RandomForest'] = rf.feature_importances_
    
    results[name] = res
    feature_importances[name] = pd.DataFrame(fi, index=features)
    
    # Display running results
    df_res = pd.DataFrame(res).T
    display(df_res)
"""))

# Cell 7: Save Results
cells.append(nbf.v4.new_code_cell("""out_path = os.path.join(results_dir, f'stage2_ml_metrics_{HORIZON}d.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=4)
print(f"Metrics saved to {out_path}")
"""))

# Cell 8: Feature Importance
cells.append(nbf.v4.new_code_cell("""from plotly.subplots import make_subplots

for name in commodities:
    df_fi = feature_importances[name]
    # Normalize each column to 100%
    df_fi_norm = df_fi.div(df_fi.sum(axis=0), axis=1) * 100
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=("XGBoost", "LightGBM", "CatBoost", "Random Forest"))
    models = ["XGBoost", "LightGBM", "CatBoost", "RandomForest"]
    
    for i, model_name in enumerate(models):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        top_10 = df_fi_norm[model_name].sort_values(ascending=True).tail(10)
        
        fig.add_trace(go.Bar(x=top_10.values, y=top_10.index, orientation='h', name=model_name), row=row, col=col)
        
    fig.update_layout(height=700, title_text=f"{name.upper()} ({HORIZON}d): Top 10 Features Broken Down by Model", showlegend=False)
    fig.show()
"""))

nb.cells = cells
with open('ML_Training_90d.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Created ML_Training_90d.ipynb for 90-Day Horizon")
