import pandas as pd
import numpy as np
import os
import json
import warnings
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')

data_dir = './data'
results_dir = './results'
models_dir = './models'
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
commodities = ['gold', 'silver', 'copper', 'natural_gas', 'crude_oil', 'wheat']

def calculate_metrics(y_true, y_pred, y_true_prev):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    actual_dir = np.sign(y_true - y_true_prev)
    pred_dir = np.sign(y_pred - y_true_prev)
    
    correct_dir = (actual_dir == pred_dir)
    dir_acc = np.mean(correct_dir) * 100
    
    return float(mae), float(rmse), float(dir_acc)

def main():
    results = {}
    
    for name in commodities:
        print(f"--- Processing {name.upper()} ---")
        file_path = os.path.join(data_dir, f"{name}.csv")
        if not os.path.exists(file_path):
            print(f"Data for {name} not found.")
            continue
            
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        
        drop_cols = ['Target_Close_Next', 'Target_Return_Next', 'Target_Direction', 
                     'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        features = [col for col in df.columns if col not in drop_cols]
        
        X = df[features]
        y_price = df['Target_Close_Next']
        y_return = df['Target_Return_Next']
        
        train_size = int(len(df) * 0.8)
        
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        
        y_price_train, y_price_test = y_price.iloc[:train_size], y_price.iloc[train_size:]
        y_return_train, y_return_test = y_return.iloc[:train_size], y_return.iloc[train_size:]
        
        # Actual true price we compare against
        y_true_price = y_price_test.values
        # Yesterday's close (needed to convert predicted returns back to prices, and for directional accuracy)
        y_true_prev = df['Close'].iloc[train_size:].values
        
        # --- 1. XGBoost predicting Raw Price ---
        xgb_p = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
        xgb_p.fit(X_train, y_price_train)
        xgb_p_preds = xgb_p.predict(X_test)
        xp_mae, xp_rmse, xp_dir = calculate_metrics(y_true_price, xgb_p_preds, y_true_prev)
        
        # --- 2. XGBoost predicting Return (then converted to price) ---
        xgb_r = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
        xgb_r.fit(X_train, y_return_train)
        xgb_r_ret_preds = xgb_r.predict(X_test)
        xgb_r_price_preds = y_true_prev * (1 + xgb_r_ret_preds)
        xr_mae, xr_rmse, xr_dir = calculate_metrics(y_true_price, xgb_r_price_preds, y_true_prev)
        
        # --- 3. LightGBM predicting Raw Price ---
        lgb_p = LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
        lgb_p.fit(X_train, y_price_train)
        lgb_p_preds = lgb_p.predict(X_test)
        lp_mae, lp_rmse, lp_dir = calculate_metrics(y_true_price, lgb_p_preds, y_true_prev)
        
        # --- 4. LightGBM predicting Return (then converted to price) ---
        lgb_r = LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
        lgb_r.fit(X_train, y_return_train)
        lgb_r_ret_preds = lgb_r.predict(X_test)
        lgb_r_price_preds = y_true_prev * (1 + lgb_r_ret_preds)
        lr_mae, lr_rmse, lr_dir = calculate_metrics(y_true_price, lgb_r_price_preds, y_true_prev)
        
        # Save the best Return models
        joblib.dump(xgb_r, os.path.join(models_dir, f'xgb_return_{name}.pkl'))
        joblib.dump(lgb_r, os.path.join(models_dir, f'lgb_return_{name}.pkl'))
        
        results[name] = {
            'Test_Size': len(y_true_price),
            'XGB_Price_Target': {'MAE': round(xp_mae,4), 'RMSE': round(xp_rmse,4), 'Dir_Acc': round(xp_dir,2)},
            'XGB_Return_Target': {'MAE': round(xr_mae,4), 'RMSE': round(xr_rmse,4), 'Dir_Acc': round(xr_dir,2)},
            'LGB_Price_Target': {'MAE': round(lp_mae,4), 'RMSE': round(lp_rmse,4), 'Dir_Acc': round(lp_dir,2)},
            'LGB_Return_Target': {'MAE': round(lr_mae,4), 'RMSE': round(lr_rmse,4), 'Dir_Acc': round(lr_dir,2)}
        }
        
        print("  [Results] XGBoost (Price Target)  : MAE {:.4f} | RMSE {:.4f} | Dir Acc {:.2f}%".format(xp_mae, xp_rmse, xp_dir))
        print("  [Results] XGBoost (Return Target) : MAE {:.4f} | RMSE {:.4f} | Dir Acc {:.2f}%".format(xr_mae, xr_rmse, xr_dir))
        print("  [Results] LightGBM(Price Target)  : MAE {:.4f} | RMSE {:.4f} | Dir Acc {:.2f}%".format(lp_mae, lp_rmse, lp_dir))
        print("  [Results] LightGBM(Return Target) : MAE {:.4f} | RMSE {:.4f} | Dir Acc {:.2f}%".format(lr_mae, lr_rmse, lr_dir))
        print()
        
    out_path = os.path.join(results_dir, 'stage2_ml_models_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"All ML models trained and evaluated. Results saved to {out_path}.")

if __name__ == "__main__":
    main()
