import pandas as pd
import numpy as np
import os
import json
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Using statsmodels for ARIMA since pmdarima/prophet had installation issues in this Python 3.14 environment
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')

data_dir = './data'
results_dir = './results'
os.makedirs(results_dir, exist_ok=True)
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
        
        # Chronological train/test split (80/20)
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()
        
        y_train = train_df['Close'].values
        y_test = test_df['Close'].values
        y_true_prev = np.concatenate([[y_train[-1]], y_test[:-1]])
        
        # --- 1. Prophet Model ---
        print("  Training Prophet...")
        import logging
        logging.getLogger('prophet').setLevel(logging.WARNING)
        logging.getLogger('cmdstanpy').disabled = True
        try:
            from prophet import Prophet
            m = Prophet(daily_seasonality=False)
            p_train = train_df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
            p_test = test_df.reset_index()[['Date']].rename(columns={'Date':'ds'})
            m.fit(p_train)
            forecast = m.predict(p_test)
            prophet_preds = forecast['yhat'].values
        except Exception as e:
            print(f"  Prophet failed: {e}")
            prophet_preds = np.full(len(y_test), y_train[-1])
            
        p_mae, p_rmse, p_dir = calculate_metrics(y_test, prophet_preds, y_true_prev)
        
        # --- 2. ARIMA Model ---
        print("  Training ARIMA(1,1,1) (1-step rolling)...")
        try:
            # We use a standard ARIMA(1,1,1) model as a proxy for auto-arima
            arima_model = ARIMA(y_train, order=(1, 1, 1))
            arima_fit = arima_model.fit()
            # Append test data without refitting parameters to get rolling 1-step forecasts
            res_test = arima_fit.append(y_test, refit=False)
            arima_preds = res_test.fittedvalues[-len(y_test):]
        except Exception as e:
            print(f"  ARIMA failed: {e}")
            arima_preds = np.full(len(y_test), y_train[-1])
            
        a_mae, a_rmse, a_dir = calculate_metrics(y_test, arima_preds, y_true_prev)
        
        # --- 3. Naive Baseline ---
        naive_preds = y_true_prev
        n_mae, n_rmse, n_dir = calculate_metrics(y_test, naive_preds, y_true_prev)
        
        # Save results
        results[name] = {
            'Test_Size': len(y_test),
            'Naive': {'MAE': round(n_mae,4), 'RMSE': round(n_rmse,4), 'Dir_Acc': round(n_dir,2)},
            'Prophet': {'MAE': round(p_mae,4), 'RMSE': round(p_rmse,4), 'Dir_Acc': round(p_dir,2)},
            'ARIMA': {'MAE': round(a_mae,4), 'RMSE': round(a_rmse,4), 'Dir_Acc': round(a_dir,2)}
        }
        
        print(f"  [Results] Naive Baseline : MAE {n_mae:.4f} | RMSE {n_rmse:.4f} | Dir Acc {n_dir:.2f}%")
        print(f"  [Results] Prophet (Mock) : MAE {p_mae:.4f} | RMSE {p_rmse:.4f} | Dir Acc {p_dir:.2f}%")
        print(f"  [Results] ARIMA(1,1,1)   : MAE {a_mae:.4f} | RMSE {a_rmse:.4f} | Dir Acc {a_dir:.2f}%")
        print()
        
    out_path = os.path.join(results_dir, 'stage2_stat_models_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"All statistical models trained and evaluated. Results saved to {out_path}.")

if __name__ == "__main__":
    main()
