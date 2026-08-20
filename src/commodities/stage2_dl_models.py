import pandas as pd
import numpy as np
import os
import json
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import copy
import joblib

warnings.filterwarnings('ignore')

# --- Configuration ---
data_dir = './data'
results_dir = './results'
models_dir = './models'
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
commodities = ['gold', 'silver', 'copper', 'natural_gas', 'crude_oil', 'wheat']

SEQ_LENGTH = 10
BATCH_SIZE = 32
EPOCHS = 150 # Increased max epochs since we use early stopping
PATIENCE = 10 # Stop if validation loss doesn't improve for 10 epochs
LR = 0.001
HIDDEN_SIZE = 32

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def calculate_metrics(y_true, y_pred, y_true_prev):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    actual_dir = np.sign(y_true - y_true_prev)
    pred_dir = np.sign(y_pred - y_true_prev)
    
    correct_dir = (actual_dir == pred_dir)
    dir_acc = np.mean(correct_dir) * 100
    
    return float(mae), float(rmse), float(dir_acc)

def create_sequences(X, y, prev_close, true_price, seq_length):
    xs, ys, prevs, trues = [], [], [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:(i+seq_length)])
        ys.append(y[i+seq_length - 1]) 
        prevs.append(prev_close[i+seq_length - 1])
        trues.append(true_price[i+seq_length - 1])
    return np.array(xs), np.array(ys), np.array(prevs), np.array(trues)

# --- Models ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads=4):
        super(TransformerModel, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = self.input_proj(x)
        out = self.transformer_encoder(x)
        out = self.fc(out[:, -1, :])
        return out

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience):
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    best_epoch = 0
    
    for epoch in range(epochs):
        # Training Phase
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item() * batch_x.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            best_epoch = epoch
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                # print(f"      Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch}")
                break
                
    # Load best weights
    model.load_state_dict(best_model_wts)
    return model

def predict_model(model, X_test_tensor):
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor)
    return preds.squeeze().numpy()

def main():
    results = {}
    
    for name in commodities:
        print(f"--- Processing {name.upper()} ---")
        file_path = os.path.join(data_dir, f"{name}.csv")
        if not os.path.exists(file_path):
            continue
            
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        
        drop_cols = ['Target_Close_Next', 'Target_Return_Next', 'Target_Direction', 
                     'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        features = [col for col in df.columns if col not in drop_cols]
        
        X = df[features].values
        y_return = df['Target_Return_Next'].values
        
        prev_close_arr = df['Close'].values
        true_price_arr = df['Target_Close_Next'].values
        
        # 1. Train/Validation/Test Split (Chronological 70/10/20)
        total_len = len(X)
        train_end = int(total_len * 0.7)
        val_end = int(total_len * 0.8)
        
        X_train_raw = X[:train_end]
        X_val_raw = X[train_end:val_end]
        X_test_raw = X[val_end:]
        
        y_train_raw = y_return[:train_end]
        y_val_raw = y_return[train_end:val_end]
        y_test_raw = y_return[val_end:]
        
        # 2. Standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)
        X_test_scaled = scaler.transform(X_test_raw)
        
        # 3. Create Sequences
        p_c_train, p_c_val, p_c_test = prev_close_arr[:train_end], prev_close_arr[train_end:val_end], prev_close_arr[val_end:]
        t_p_train, t_p_val, t_p_test = true_price_arr[:train_end], true_price_arr[train_end:val_end], true_price_arr[val_end:]
        
        X_train_seq, y_train_seq, _, _ = create_sequences(X_train_scaled, y_train_raw, p_c_train, t_p_train, SEQ_LENGTH)
        X_val_seq, y_val_seq, _, _ = create_sequences(X_val_scaled, y_val_raw, p_c_val, t_p_val, SEQ_LENGTH)
        X_test_seq, y_test_seq, p_c_test_seq, t_p_test_seq = create_sequences(X_test_scaled, y_test_raw, p_c_test, t_p_test, SEQ_LENGTH)
        
        # 4. Data Loaders
        train_loader = DataLoader(TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32), torch.tensor(y_train_seq, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=False)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val_seq, dtype=torch.float32), torch.tensor(y_val_seq, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=False)
        
        X_test_t = torch.tensor(X_test_seq, dtype=torch.float32)
        input_size = X_train_seq.shape[2]
        criterion = nn.MSELoss()
        
        # --- Train LSTM ---
        print("  Training LSTM (with Early Stopping)...")
        lstm = LSTMModel(input_size, HIDDEN_SIZE)
        optimizer_lstm = torch.optim.Adam(lstm.parameters(), lr=LR)
        lstm = train_model(lstm, train_loader, val_loader, criterion, optimizer_lstm, EPOCHS, PATIENCE)
        
        lstm_ret_preds = predict_model(lstm, X_test_t)
        lstm_price_preds = p_c_test_seq * (1 + lstm_ret_preds)
        lstm_mae, lstm_rmse, lstm_dir = calculate_metrics(t_p_test_seq, lstm_price_preds, p_c_test_seq)
        
        # --- Train Transformer ---
        print("  Training Transformer (with Early Stopping)...")
        transformer = TransformerModel(input_size, HIDDEN_SIZE)
        optimizer_trans = torch.optim.Adam(transformer.parameters(), lr=LR)
        transformer = train_model(transformer, train_loader, val_loader, criterion, optimizer_trans, EPOCHS, PATIENCE)
        
        trans_ret_preds = predict_model(transformer, X_test_t)
        trans_price_preds = p_c_test_seq * (1 + trans_ret_preds)
        trans_mae, trans_rmse, trans_dir = calculate_metrics(t_p_test_seq, trans_price_preds, p_c_test_seq)
        
        # Save Models and Scaler
        joblib.dump(scaler, os.path.join(models_dir, f'scaler_{name}.pkl'))
        torch.save(lstm.state_dict(), os.path.join(models_dir, f'lstm_{name}.pt'))
        torch.save(transformer.state_dict(), os.path.join(models_dir, f'transformer_{name}.pt'))
        
        results[name] = {
            'Test_Size': len(t_p_test_seq),
            'LSTM': {'MAE': round(lstm_mae,4), 'RMSE': round(lstm_rmse,4), 'Dir_Acc': round(lstm_dir,2)},
            'Transformer': {'MAE': round(trans_mae,4), 'RMSE': round(trans_rmse,4), 'Dir_Acc': round(trans_dir,2)}
        }
        
        print("  [Results] LSTM        : MAE {:.4f} | RMSE {:.4f} | Dir Acc {:.2f}%".format(lstm_mae, lstm_rmse, lstm_dir))
        print("  [Results] Transformer : MAE {:.4f} | RMSE {:.4f} | Dir Acc {:.2f}%".format(trans_mae, trans_rmse, trans_dir))
        print()
        
    out_path = os.path.join(results_dir, 'stage2_dl_models_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"All DL models trained and evaluated. Results saved to {out_path}.")

if __name__ == "__main__":
    main()
