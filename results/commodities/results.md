# Comprehensive Forecasting Results

This document compiles the performance metrics (MAE, RMSE, and Directional Accuracy) of all models evaluated across the different stages of the project.

## Gold

| Model Category | Specific Model | MAE | RMSE | Directional Accuracy (%) |
|---|---|---|---|---|
| **Baseline** | Naive (Random Walk) | 34.7000 | 56.4847 | 3.71% |
| **Statistical** | ARIMA(1,1,1) | 34.7186 | 56.4191 | 50.39% |
| **Machine Learning** | XGBoost (Returns) | 36.8494 | 59.5345 | 49.29% |
| **Machine Learning** | LightGBM (Returns) | 36.0378 | 57.8892 | 51.49% |
| **Deep Learning** | LSTM | 44.1992 | 74.2012 | 50.56% |
| **Deep Learning** | Transformer | 44.0380 | 67.1208 | 51.83% |

## Silver

| Model Category | Specific Model | MAE | RMSE | Directional Accuracy (%) |
|---|---|---|---|---|
| **Baseline** | Naive (Random Walk) | 1.0318 | 2.2689 | 3.71% |
| **Statistical** | ARIMA(1,1,1) | 1.0371 | 2.2617 | 51.16% |
| **Machine Learning** | XGBoost (Returns) | 1.0655 | 2.2937 | 50.24% |
| **Machine Learning** | LightGBM (Returns) | 1.0810 | 2.3162 | 51.65% |
| **Deep Learning** | LSTM | 1.2075 | 2.4325 | 52.47% |
| **Deep Learning** | Transformer | 1.1442 | 2.3834 | 50.88% |

## Copper

| Model Category | Specific Model | MAE | RMSE | Directional Accuracy (%) |
|---|---|---|---|---|
| **Baseline** | Naive (Random Walk) | 0.0627 | 0.1003 | 4.02% |
| **Statistical** | ARIMA(1,1,1) | 0.0624 | 0.1001 | 53.32% |
| **Machine Learning** | XGBoost (Returns) | 0.0651 | 0.1023 | 49.92% |
| **Machine Learning** | LightGBM (Returns) | 0.0653 | 0.1027 | 51.81% |
| **Deep Learning** | LSTM | 0.0828 | 0.1269 | 52.31% |
| **Deep Learning** | Transformer | 0.0725 | 0.1091 | 45.45% |

## Natural Gas

| Model Category | Specific Model | MAE | RMSE | Directional Accuracy (%) |
|---|---|---|---|---|
| **Baseline** | Naive (Random Walk) | 0.1084 | 0.2133 | 3.71% |
| **Statistical** | ARIMA(1,1,1) | 0.1090 | 0.2125 | 49.46% |
| **Machine Learning** | XGBoost (Returns) | 0.1186 | 0.2170 | 47.88% |
| **Machine Learning** | LightGBM (Returns) | 0.1171 | 0.2171 | 48.35% |
| **Deep Learning** | LSTM | 0.1242 | 0.2266 | 45.93% |
| **Deep Learning** | Transformer | 0.1181 | 0.2184 | 49.12% |

## Crude Oil

| Model Category | Specific Model | MAE | RMSE | Directional Accuracy (%) |
|---|---|---|---|---|
| **Baseline** | Naive (Random Walk) | 1.3379 | 2.1458 | 3.86% |
| **Statistical** | ARIMA(1,1,1) | 1.3389 | 2.1433 | 50.54% |
| **Machine Learning** | XGBoost (Returns) | 1.4159 | 2.2950 | 44.90% |
| **Machine Learning** | LightGBM (Returns) | 68.8840 | 298.7584 | 46.31% |
| **Deep Learning** | LSTM | 1.6200 | 2.6533 | 52.15% |
| **Deep Learning** | Transformer | 2.6398 | 4.5465 | 50.72% |

## Wheat

| Model Category | Specific Model | MAE | RMSE | Directional Accuracy (%) |
|---|---|---|---|---|
| **Baseline** | Naive (Random Walk) | 7.0537 | 9.4363 | 6.18% |
| **Statistical** | ARIMA(1,1,1) | 7.1049 | 9.4300 | 48.22% |
| **Machine Learning** | XGBoost (Returns) | 7.1885 | 9.6626 | 50.71% |
| **Machine Learning** | LightGBM (Returns) | 7.3560 | 9.9752 | 47.41% |
| **Deep Learning** | LSTM | 8.0482 | 10.5497 | 48.96% |
| **Deep Learning** | Transformer | 8.2420 | 10.5007 | 44.02% |

