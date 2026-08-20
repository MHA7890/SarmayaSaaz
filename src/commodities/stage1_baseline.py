import pandas as pd
import numpy as np
import os
import json

data_dir = 'p:/ffcproj/data'
results_dir = 'p:/ffcproj/results'
os.makedirs(results_dir, exist_ok=True)

commodities = ['gold', 'silver', 'copper', 'natural_gas', 'crude_oil', 'wheat']

def calculate_metrics(y_true, y_pred, y_true_prev):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    # Directional Accuracy: Did we predict the right direction compared to the previous day?
    # +1 for up, -1 for down, 0 for flat
    actual_dir = np.sign(y_true - y_true_prev)
    pred_dir = np.sign(y_pred - y_true_prev)
    
    # Calculate match percentage
    correct_dir = (actual_dir == pred_dir)
    dir_acc = np.mean(correct_dir) * 100
    
    return {
        'MAE': round(float(mae), 4),
        'RMSE': round(float(rmse), 4),
        'Directional_Accuracy_%': round(float(dir_acc), 2)
    }

def main():
    baseline_results = {}
    
    for name in commodities:
        file_path = os.path.join(data_dir, f"{name}.csv")
        if not os.path.exists(file_path):
            print(f"Skipping {name}: Data file not found.")
            continue
            
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        
        # For naive forecast: Tomorrow's price = Today's price
        # So prediction for day t is the actual close of day t-1
        df['Pred_Close'] = df['Close'].shift(1)
        
        # We also need the previous day's actual close to compute direction
        # In the naive case, Pred_Close IS Prev_Close
        df['Prev_Close'] = df['Close'].shift(1)
        
        # Drop the first row which has NaNs due to shift
        df_valid = df.dropna(subset=['Pred_Close', 'Prev_Close'])
        
        # Chronological train/test split (80/20)
        # This establishes a standard test set for future stages
        train_size = int(len(df_valid) * 0.8)
        test_df = df_valid.iloc[train_size:]
        
        y_true = test_df['Close'].values
        y_pred = test_df['Pred_Close'].values
        y_true_prev = test_df['Prev_Close'].values
        
        metrics = calculate_metrics(y_true, y_pred, y_true_prev)
        baseline_results[name] = metrics
        
        print(f"--- Baseline Metrics for {name.upper()} ---")
        print(f"Test Set Size: {len(test_df)} days (20% of data)")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print()
        
    # Save results for future comparisons
    out_path = os.path.join(results_dir, 'stage1_baseline_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(baseline_results, f, indent=4)
    print(f"Baseline results saved to {out_path}")

if __name__ == "__main__":
    main()
