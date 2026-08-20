# Sentiment Integration: Champion vs Challenger Benchmark Results

## 1. Overall System Impact

**Average Directional Accuracy Shift:** +1.05%

**Average MAE Improvement (Error Reduction):** +3.68

## 2. Impact by Horizon

| Horizon | Directional Accuracy Change (%) | MAE Improvement (Lower Error) |
|---------|---------------------------------|-------------------------------|
| 120d | 0.67% | 10.1416 |
| 14d | 0.64% | 0.2123 |
| 28d | 1.40% | 1.0785 |
| 42d | 0.62% | 1.1027 |
| 60d | 1.77% | 5.7373 |
| 7d | 0.59% | 0.4248 |
| 90d | 1.63% | 7.0781 |

## 3. Top 5 Most Improved Models (The New Champions)

| Commodity | Horizon | Model | Old DirAcc | New DirAcc | Improvement |
|-----------|---------|-------|------------|------------|-------------|
| gold | 120d | LSTM | 12.15% | 93.01% | +80.86% |
| gold | 90d | LSTM | 9.06% | 86.66% | +77.60% |
| gold | 60d | LSTM | 16.64% | 80.42% | +63.78% |
| silver | 90d | GRU | 19.44% | 77.92% | +58.48% |
| gold | 60d | GRU | 18.11% | 64.11% | +46.00% |

## 4. Top 5 Degraded Models (Revert to Old Champions)

| Commodity | Horizon | Model | Old DirAcc | New DirAcc | Degradation |
|-----------|---------|-------|------------|------------|-------------|
| gold | 60d | XGBoost | 82.5% | 33.87% | -48.63% |
| silver | 90d | TFT | 83.86% | 45.47% | -38.39% |
| gold | 120d | N-BEATS | 76.54% | 41.76% | -34.78% |
| silver | 28d | TFT | 76.94% | 42.74% | -34.20% |
| copper | 120d | Transformer | 80.15% | 47.54% | -32.61% |
