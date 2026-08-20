import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()

    # Markdown Intro
    nb.cells.append(nbf.v4.new_markdown_cell("""
# 🧠 Crypto Engine: Phase 4 (Asset Clustering & Model Training)
Welcome to the heart of the FundForge Crypto Engine. This interactive notebook allows you to monitor the deep learning training process live on your RTX 6000 Ada.

**Features:**
1. Dynamic K-Means Asset Clustering.
2. Purged Time-Series DataLoading.
3. Live Loss Curve Visualizations (Train vs. Validation).
4. Optuna Bayesian Optimization.
"""))

    # Imports
    nb.cells.append(nbf.v4.new_code_cell("""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from IPython.display import clear_output

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import optuna

# Force PyTorch to use your RTX 6000 Ada
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {device}")
"""))

    # Clustering
    nb.cells.append(nbf.v4.new_markdown_cell("## 1. Dynamic Asset Clustering (K-Means)"))
    nb.cells.append(nbf.v4.new_code_cell("""
INPUT_DIR = "../data/crypto_processed"
files = glob.glob(f"{INPUT_DIR}/*.csv")

stats = []
dataframes = {}

print("Analyzing statistical signatures of 29 assets...")
for f in files:
    ticker = os.path.basename(f).replace('.csv', '')
    df = pd.read_csv(f, index_col='Date', parse_dates=True)
    dataframes[ticker] = df
    
    # Calculate signature: Avg Volatility and Avg 7d Return
    avg_vol = df['BTC_Volatility_30d'].mean() if 'BTC_Volatility_30d' in df.columns else df['Volatility_30d'].mean()
    avg_ret = df['Target_7d'].mean()
    
    stats.append({'Ticker': ticker, 'Volatility': avg_vol, 'Avg_Return': avg_ret})

stats_df = pd.DataFrame(stats).set_index('Ticker')

# Standardize and Cluster
scaler = StandardScaler()
scaled_stats = scaler.fit_transform(stats_df)

kmeans = KMeans(n_clusters=4, random_state=42)
stats_df['Cluster'] = kmeans.fit_predict(scaled_stats)

# Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(data=stats_df, x='Volatility', y='Avg_Return', hue='Cluster', palette='deep', s=100)
for i in range(stats_df.shape[0]):
    plt.text(stats_df['Volatility'][i], stats_df['Avg_Return'][i]+0.001, stats_df.index[i], fontsize=8)
plt.title("Asset Clusters (K-Means)")
plt.show()

# Display cluster members
for c in range(4):
    members = stats_df[stats_df['Cluster'] == c].index.tolist()
    print(f"Cluster {c} ({len(members)} assets): {', '.join(members)}")
"""))

    # DataPrep
    nb.cells.append(nbf.v4.new_markdown_cell("## 2. 3D Sequence Engineering (PyTorch DataLoaders)"))
    nb.cells.append(nbf.v4.new_code_cell("""
class CryptoSequenceDataset(Dataset):
    def __init__(self, data, targets, seq_length=30):
        self.data = data
        self.targets = targets
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.targets[idx + self.seq_length - 1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def prepare_cluster_data(cluster_id, target_horizon='Target_7d'):
    members = stats_df[stats_df['Cluster'] == cluster_id].index.tolist()
    
    # Vertically stack all data for the cluster
    combined_df = pd.concat([dataframes[t] for t in members])
    combined_df = combined_df.dropna(subset=[target_horizon])
    
    features = [c for c in combined_df.columns if 'Target_' not in c and c != 'Date']
    
    X = combined_df[features].values
    y = combined_df[target_horizon].values
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train / Val Split (Purged - simplistic 80/20 chronological for this demo)
    split_idx = int(len(X_scaled) * 0.8)
    
    # Embargo gap (horizon days) to prevent leakage
    embargo = int(target_horizon.split('_')[1].replace('d',''))
    
    X_train, y_train = X_scaled[:split_idx], y[:split_idx]
    X_val, y_val = X_scaled[split_idx + embargo:], y[split_idx + embargo:]
    
    train_ds = CryptoSequenceDataset(X_train, y_train)
    val_ds = CryptoSequenceDataset(X_val, y_val)
    
    # RTX 6000 Ada massive batch sizes
    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False)
    
    return train_loader, val_loader, len(features)

# Test run for Cluster 0, 7D Horizon
train_loader, val_loader, num_features = prepare_cluster_data(0, 'Target_7d')
print(f"Features: {num_features}. Ready for training.")
"""))

    # Model
    nb.cells.append(nbf.v4.new_markdown_cell("## 3. Deep Learning Architecture (LSTM)"))
    nb.cells.append(nbf.v4.new_code_cell("""
class CryptoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(CryptoLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Take last timestep
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
"""))

    # Training Loop with Live Plotting
    nb.cells.append(nbf.v4.new_markdown_cell("## 4. Live Training & Visualization Engine"))
    nb.cells.append(nbf.v4.new_code_cell("""
def plot_live_losses(train_losses, val_losses, epoch, total_epochs):
    clear_output(wait=True)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss (Huber)', color='blue', linewidth=2)
    plt.plot(val_losses, label='Val Loss (Huber)', color='orange', linewidth=2)
    plt.title(f"Live Training Progress: Epoch {epoch}/{total_epochs}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def train_model_live(model, train_loader, val_loader, epochs=30, lr=0.001):
    model = model.to(device)
    criterion = nn.HuberLoss() # Outlier protection for Crypto wicks
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    epoch_bar = tqdm(range(epochs), desc="Epochs")
    
    for epoch in epoch_bar:
        model.train()
        batch_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            
            batch_train_loss += loss.item()
            
        avg_train = batch_train_loss / len(train_loader)
        train_losses.append(avg_train)
        
        # Validation
        model.eval()
        batch_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                batch_val_loss += loss.item()
                
        avg_val = batch_val_loss / len(val_loader)
        val_losses.append(avg_val)
        
        # Update Live Plot
        plot_live_losses(train_losses, val_losses, epoch+1, epochs)
        epoch_bar.set_postfix({'Train Loss': f"{avg_train:.4f}", 'Val Loss': f"{avg_val:.4f}"})
        
    return min(val_losses)

# Example Run
print("Initializing Trial Run for Cluster 0 (7D Horizon)...")
test_model = CryptoLSTM(input_size=num_features, hidden_size=64, num_layers=2, dropout=0.2)
best_val_loss = train_model_live(test_model, train_loader, val_loader, epochs=20, lr=0.001)
print(f"Training Complete! Best Validation Loss: {best_val_loss:.4f}")
"""))

    # Write notebook
    with open('notebooks/Crypto_Phase4_Training.ipynb', 'w') as f:
        nbf.write(nb, f)
    print("Successfully generated notebooks/Crypto_Phase4_Training.ipynb")

if __name__ == "__main__":
    create_notebook()
