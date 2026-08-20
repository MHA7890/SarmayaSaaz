# MUFAP Quant Engine: Master Training Results

## Overview
This document contains the official out-of-sample validation metrics (Mean Absolute Error) for all 5 Mutual Fund Super-Clusters across all 7 forecasting horizons. 

*Note: MAE is represented as a fractional percentage (e.g., `0.031` = 3.1% error margin).*

---

### 1. The Equity Cluster
*(Highly volatile, tracking the PSX KSE-100)*

| Horizon | Winning Model | Validation MAE | Error % |
| :--- | :--- | :--- | :--- |
| **7 Days** | LightGBM | 0.0311 | 3.11% |
| **14 Days** | XGBoost | 0.0431 | 4.31% |
| **28 Days** | LightGBM | 0.0613 | 6.13% |
| **42 Days** | LightGBM | 0.0734 | 7.34% |
| **60 Days** | XGBoost | 0.0801 | 8.01% |
| **90 Days** | LightGBM | 0.0993 | 9.93% |
| **120 Days** | XGBoost | 0.1114 | 11.14% |

---

### 2. The Money Market Cluster
*(Extremely stable, tracking SBP Interest Rates)*

| Horizon | Winning Model | Validation MAE | Error % |
| :--- | :--- | :--- | :--- |
| **7 Days** | LightGBM | 0.0037 | 0.37% |
| **14 Days** | LightGBM | 0.0068 | 0.68% |
| **28 Days** | XGBoost | 0.0126 | 1.26% |
| **42 Days** | XGBoost | 0.0187 | 1.87% |
| **60 Days** | LightGBM | 0.0273 | 2.73% |
| **90 Days** | LightGBM | 0.0344 | 3.44% |
| **120 Days** | XGBoost | 0.0403 | 4.03% |

---

### 3. The Income & Debt Cluster
*(Highly stable, tracking PKRV Bond Yields)*

| Horizon | Winning Model | Validation MAE | Error % |
| :--- | :--- | :--- | :--- |
| **7 Days** | LightGBM | 0.0041 | 0.41% |
| **14 Days** | LightGBM | 0.0078 | 0.78% |
| **28 Days** | XGBoost | 0.0143 | 1.43% |
| **42 Days** | XGBoost | 0.0210 | 2.10% |
| **60 Days** | XGBoost | 0.0312 | 3.12% |
| **90 Days** | XGBoost | 0.0394 | 3.94% |
| **120 Days** | LightGBM | 0.0445 | 4.45% |

---

### 4. The Commodity (Gold) Cluster
*(Noisy, tracking Global Gold Spot Futures)*

| Horizon | Winning Model | Validation MAE | Error % |
| :--- | :--- | :--- | :--- |
| **7 Days** | LSTM (PyTorch) | 0.0324 | 3.24% |
| **14 Days** | LSTM (PyTorch) | 0.0419 | 4.19% |
| **28 Days** | LightGBM | 0.0580 | 5.80% |
| **42 Days** | XGBoost | 0.0727 | 7.27% |
| **60 Days** | LSTM (PyTorch) | 0.0906 | 9.06% |
| **90 Days** | XGBoost | 0.1124 | 11.24% |
| **120 Days** | LSTM (PyTorch) | 0.1382 | 13.82% |

---

### 5. The Balanced / Asset Allocation Cluster
*(Mixed risk, blending Equities and Bonds)*

| Horizon | Winning Model | Validation MAE | Error % |
| :--- | :--- | :--- | :--- |
| **7 Days** | LSTM (PyTorch) | 0.0140 | 1.40% |
| **14 Days** | LightGBM | 0.0209 | 2.09% |
| **28 Days** | XGBoost | 0.0312 | 3.12% |
| **42 Days** | LightGBM | 0.0408 | 4.08% |
| **60 Days** | XGBoost | 0.0508 | 5.08% |
| **90 Days** | LightGBM | 0.0625 | 6.25% |
| **120 Days** | LightGBM | 0.0741 | 7.41% |

---
## Model Architecture Notes
- **Leakage Prevention:** All models were trained using a Strict Chronological Purged Split with a global date anchor (Nov 2024). A rolling purge gap equal to the horizon length was enforced between the Train and Validation sets.
- **Scaler Isolation:** The `StandardScaler` was strictly fitted ONLY on the Training split to prevent statistical target leakage from the validation timeline.
- **Overfitting Shield:** Machine Learning models (XGBoost/LightGBM) utilized 50-round early stopping. Deep Learning models (LSTM) utilized Huber Loss, L2 Weight Decay (1e-4), 20% Dropout, and 15-epoch early stopping.
