import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import json
from captum.attr import IntegratedGradients

# Re-import architectures to load weights
from stage2_dl_models import LSTMModel, TransformerModel

data_dir = './data'
models_dir = './models'
results_dir = './results'
os.makedirs(results_dir, exist_ok=True)

commodities = ['gold', 'silver', 'copper', 'natural_gas', 'crude_oil', 'wheat']
SEQ_LENGTH = 10
HIDDEN_SIZE = 32

def create_sequences_simple(X, y, seq_length):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:(i+seq_length)])
        ys.append(y[i+seq_length - 1])
    return np.array(xs), np.array(ys)

def get_base_mse(model, X_tensor, y_tensor):
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor).squeeze()
        mse = torch.mean((preds - y_tensor) ** 2).item()
    return mse

def permutation_importance(model, X_np, y_tensor, num_features):
    base_mse = get_base_mse(model, torch.tensor(X_np, dtype=torch.float32), y_tensor)
    importances = np.zeros(num_features)
    
    for f in range(num_features):
        X_shuffled = X_np.copy()
        # Shuffle the feature across the batch dimension (swap entire sequences of this feature among samples)
        np.random.shuffle(X_shuffled[:, :, f])
        
        shuffled_mse = get_base_mse(model, torch.tensor(X_shuffled, dtype=torch.float32), y_tensor)
        importances[f] = shuffled_mse - base_mse
        
    # Normalize to sum to 1 for easier comparison (clip negative values to 0)
    importances = np.clip(importances, 0, None)
    total = np.sum(importances)
    if total > 0:
        importances = importances / total
    return importances

def calculate_integrated_gradients(model, X_tensor):
    model.eval()
    ig = IntegratedGradients(model)
    # Output of model is [batch, 1], so target is 0 for all batch elements
    attributions = ig.attribute(X_tensor, target=0, n_steps=20)
    
    # attributions shape: [batch, seq_length, num_features]
    # We take the absolute value (magnitude of attribution) and mean over batch and sequence
    attr_np = attributions.detach().numpy()
    feature_attr = np.mean(np.abs(attr_np), axis=(0, 1))
    
    # Normalize to sum to 1
    total = np.sum(feature_attr)
    if total > 0:
        feature_attr = feature_attr / total
    return feature_attr

def main():
    results = {}
    
    for comm in commodities:
        print(f"--- Extracting Feature Importance for {comm.upper()} ---")
        
        file_path = os.path.join(data_dir, f"{comm}.csv")
        if not os.path.exists(file_path):
            continue
            
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        
        drop_cols = ['Target_Close_Next', 'Target_Return_Next', 'Target_Direction', 
                     'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        features = [col for col in df.columns if col not in drop_cols]
        num_features = len(features)
        
        X = df[features]
        y_return = df['Target_Return_Next']
        
        total_len = len(X)
        val_end = int(total_len * 0.8)
        
        X_test_raw = X.iloc[val_end:].values
        y_test_raw = y_return.iloc[val_end:].values
        
        # Load scaler
        scaler_path = os.path.join(models_dir, f"scaler_{comm}.pkl")
        if not os.path.exists(scaler_path):
            continue
        scaler = joblib.load(scaler_path)
        
        X_test_scaled = scaler.transform(X_test_raw)
        
        X_test_seq, y_test_seq = create_sequences_simple(X_test_scaled, y_test_raw, SEQ_LENGTH)
        X_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
        y_tensor = torch.tensor(y_test_seq, dtype=torch.float32)
        
        comm_results = {}
        
        # --- LSTM ---
        lstm_path = os.path.join(models_dir, f"lstm_{comm}.pt")
        if os.path.exists(lstm_path):
            lstm_model = LSTMModel(num_features, HIDDEN_SIZE)
            lstm_model.load_state_dict(torch.load(lstm_path))
            
            print("  Running LSTM Permutation Importance...")
            lstm_perm = permutation_importance(lstm_model, X_test_seq, y_tensor, num_features)
            print("  Running LSTM Integrated Gradients...")
            lstm_ig = calculate_integrated_gradients(lstm_model, X_tensor)
            
            comm_results['LSTM'] = {
                'Permutation': {f: round(float(w), 4) for f, w in zip(features, lstm_perm)},
                'IntegratedGradients': {f: round(float(w), 4) for f, w in zip(features, lstm_ig)}
            }
            
        # --- Transformer ---
        tf_path = os.path.join(models_dir, f"transformer_{comm}.pt")
        if os.path.exists(tf_path):
            tf_model = TransformerModel(num_features, HIDDEN_SIZE)
            tf_model.load_state_dict(torch.load(tf_path))
            
            print("  Running Transformer Permutation Importance...")
            tf_perm = permutation_importance(tf_model, X_test_seq, y_tensor, num_features)
            print("  Running Transformer Integrated Gradients...")
            tf_ig = calculate_integrated_gradients(tf_model, X_tensor)
            
            comm_results['Transformer'] = {
                'Permutation': {f: round(float(w), 4) for f, w in zip(features, tf_perm)},
                'IntegratedGradients': {f: round(float(w), 4) for f, w in zip(features, tf_ig)}
            }
            
        results[comm] = comm_results
        
    out_path = os.path.join(results_dir, "dl_feature_importance.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nSaved all DL feature importance metrics to {out_path}")

if __name__ == "__main__":
    main()
