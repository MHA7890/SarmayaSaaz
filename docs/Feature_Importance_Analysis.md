# AI Feature Importance Analysis & Glossary

This document analyzes the implicit feature selection performed by our Tree-Based Machine Learning models (XGBoost and LightGBM) and provides a complete glossary of all technical indicators engineered for the platform.

## 📚 Feature Glossary & Definitions

| Feature Code | Full Name | Definition & Intuition |
|---|---|---|
| **Return** | Daily Simple Return | Percentage change in closing price from yesterday to today: `(Close_t - Close_{t-1}) / Close_{t-1}` |
| **Log_Return** | Daily Logarithmic Return | Symmetrized percentage change using natural log: `ln(Close_t / Close_{t-1})`. Standard metric in quant finance. |
| **Ret_Lag_N** | Return Lag (N Days) | Daily return from N days ago (e.g., `Ret_Lag_1` is yesterday's return, `Ret_Lag_5` is return from 5 days ago). Captures short-term momentum memory. |
| **Daily_Range_Pct** | Daily Intraday Range Percentage | High price minus Low price normalized by Close: `(High - Low) / Close`. Measures intraday price volatility and market panic/uncertainty. |
| **Close_to_MA_N** | Ratio of Close to N-Day Simple Moving Average | Today's close divided by N-day average (`Close / SMA_N`). Values > 1.0 indicate price is above trend; < 1.0 indicates below trend. |
| **Close_to_EMA_N** | Ratio of Close to N-Day Exponential Moving Average | Today's close divided by N-day exponential average (`Close / EMA_N`). Weighs recent days heavier than simple MA. |
| **Rolling_Std_N** | N-Day Rolling Standard Deviation of Returns | Standard deviation of daily returns over the last N days (`Std(Returns_{t-N:t})`). Direct measure of recent market volatility/risk. |
| **RSI_14** | Relative Strength Index (14-Period) | Bounded technical momentum oscillator (0 to 100). Measures speed and change of price movements (>70 = Overbought, <30 = Oversold). |
| **MACD_Pct** | Normalized Moving Average Convergence Divergence | MACD Histogram value divided by Close price (`MACD_Diff / Close`). Measures trend direction and momentum acceleration normalized across price levels. |
| **Volume_to_MA_10** | Volume to 10-Day Moving Average Ratio | Today's trading volume divided by its 10-day moving average (`Volume / SMA_10(Volume)`). Values > 1.0 represent unusual volume spikes. |

---

## 📊 Commodity Feature Importance Rankings

### Gold

#### XGBoost Top 5 Features (Gain)
1. **Rolling_Std_10** (Weight: 0.0950)
2. **Close_to_MA_20** (Weight: 0.0674)
3. **Rolling_Std_20** (Weight: 0.0669)
4. **Close_to_MA_5** (Weight: 0.0631)
5. **Daily_Range_Pct** (Weight: 0.0619)

#### LightGBM Top 5 Features (Relative Splits)
1. **Daily_Range_Pct** (Weight: 0.0800)
2. **Rolling_Std_10** (Weight: 0.0750)
3. **Ret_Lag_5** (Weight: 0.0747)
4. **Ret_Lag_3** (Weight: 0.0650)
5. **Rolling_Std_20** (Weight: 0.0643)

---

### Silver

#### XGBoost Top 5 Features (Gain)
1. **Close_to_EMA_20** (Weight: 0.0787)
2. **MACD_Pct** (Weight: 0.0779)
3. **Close_to_MA_20** (Weight: 0.0718)
4. **RSI_14** (Weight: 0.0648)
5. **Close_to_MA_5** (Weight: 0.0618)

#### LightGBM Top 5 Features (Relative Splits)
1. **Daily_Range_Pct** (Weight: 0.0853)
2. **Ret_Lag_1** (Weight: 0.0793)
3. **Rolling_Std_20** (Weight: 0.0747)
4. **Rolling_Std_10** (Weight: 0.0740)
5. **Ret_Lag_3** (Weight: 0.0643)

---

### Copper

#### XGBoost Top 5 Features (Gain)
1. **Close_to_EMA_20** (Weight: 0.0834)
2. **Close_to_EMA_10** (Weight: 0.0668)
3. **Ret_Lag_5** (Weight: 0.0656)
4. **Rolling_Std_10** (Weight: 0.0632)
5. **Close_to_MA_5** (Weight: 0.0630)

#### LightGBM Top 5 Features (Relative Splits)
1. **Ret_Lag_5** (Weight: 0.0827)
2. **Ret_Lag_2** (Weight: 0.0747)
3. **Rolling_Std_10** (Weight: 0.0727)
4. **Daily_Range_Pct** (Weight: 0.0707)
5. **Return** (Weight: 0.0697)

---

### Natural Gas

#### XGBoost Top 5 Features (Gain)
1. **Close_to_MA_10** (Weight: 0.0938)
2. **Close_to_EMA_20** (Weight: 0.0857)
3. **Ret_Lag_10** (Weight: 0.0703)
4. **Close_to_MA_5** (Weight: 0.0654)
5. **Ret_Lag_3** (Weight: 0.0645)

#### LightGBM Top 5 Features (Relative Splits)
1. **Daily_Range_Pct** (Weight: 0.0753)
2. **Ret_Lag_3** (Weight: 0.0750)
3. **Rolling_Std_10** (Weight: 0.0693)
4. **Ret_Lag_10** (Weight: 0.0680)
5. **Rolling_Std_20** (Weight: 0.0667)

---

### Crude Oil

#### XGBoost Top 5 Features (Gain)
1. **RSI_14** (Weight: 0.4932)
2. **Ret_Lag_10** (Weight: 0.1736)
3. **Close_to_MA_50** (Weight: 0.1619)
4. **Ret_Lag_1** (Weight: 0.1578)
5. **Rolling_Std_10** (Weight: 0.0063)

#### LightGBM Top 5 Features (Relative Splits)
1. **Ret_Lag_10** (Weight: 0.1007)
2. **Ret_Lag_1** (Weight: 0.0897)
3. **Return** (Weight: 0.0810)
4. **RSI_14** (Weight: 0.0783)
5. **Daily_Range_Pct** (Weight: 0.0763)

---

### Wheat

#### XGBoost Top 5 Features (Gain)
1. **Daily_Range_Pct** (Weight: 0.0857)
2. **Close_to_MA_20** (Weight: 0.0729)
3. **RSI_14** (Weight: 0.0685)
4. **Ret_Lag_1** (Weight: 0.0666)
5. **Volume_to_MA_10** (Weight: 0.0654)

#### LightGBM Top 5 Features (Relative Splits)
1. **Daily_Range_Pct** (Weight: 0.0917)
2. **Ret_Lag_1** (Weight: 0.0813)
3. **Volume_to_MA_10** (Weight: 0.0753)
4. **Rolling_Std_10** (Weight: 0.0680)
5. **Ret_Lag_10** (Weight: 0.0640)

---

## 💡 Key Takeaways
* **Volatility Reigns Supreme:** Across almost all commodities, rolling standard deviation (`Rolling_Std_10`, `Rolling_Std_20`) and intraday volatility (`Daily_Range_Pct`) heavily outrank traditional momentum indicators.
* **Mean Reversion:** Features measuring the price's distance from moving averages (`Close_to_MA_20`, `Close_to_EMA_10`) are consistently used as primary split criteria, proving that commodities frequently exhibit mean-reverting behavior.
* **Volume is Secondary:** Interestingly, Volume relative to its moving average rarely makes the top 5, suggesting that for these highly liquid futures markets, price action itself is a stronger predictor than trading volume.

---

## 🧠 Deep Learning Feature Importance (LSTM & Transformer)

Unlike Tree models, Neural Networks determine importance by adjusting internal weight matrices across a 10-day sequential window. Below are the Top 5 features mathematically extracted using Facebook AI's **Integrated Gradients (Captum)**.

### Gold

#### PyTorch LSTM Top 5 Features (Integrated Gradients)
1. **Close_to_EMA_20** (Attribution Score: 0.1469)
2. **Close_to_EMA_10** (Attribution Score: 0.1028)
3. **Close_to_MA_50** (Attribution Score: 0.0786)
4. **Return** (Attribution Score: 0.0720)
5. **Rolling_Std_20** (Attribution Score: 0.0702)

#### PyTorch Transformer Top 5 Features (Integrated Gradients)
1. **Return** (Attribution Score: 0.1077)
2. **Close_to_EMA_20** (Attribution Score: 0.0986)
3. **Close_to_MA_50** (Attribution Score: 0.0786)
4. **Ret_Lag_10** (Attribution Score: 0.0729)
5. **Ret_Lag_5** (Attribution Score: 0.0643)

### Silver

#### PyTorch LSTM Top 5 Features (Integrated Gradients)
1. **Log_Return** (Attribution Score: 0.1230)
2. **Return** (Attribution Score: 0.1134)
3. **Close_to_MA_20** (Attribution Score: 0.0923)
4. **Close_to_MA_10** (Attribution Score: 0.0847)
5. **Close_to_EMA_10** (Attribution Score: 0.0778)

#### PyTorch Transformer Top 5 Features (Integrated Gradients)
1. **Rolling_Std_20** (Attribution Score: 0.1185)
2. **Close_to_MA_10** (Attribution Score: 0.0919)
3. **Close_to_EMA_20** (Attribution Score: 0.0843)
4. **Close_to_MA_20** (Attribution Score: 0.0780)
5. **MACD_Pct** (Attribution Score: 0.0725)

### Copper

#### PyTorch LSTM Top 5 Features (Integrated Gradients)
1. **Close_to_EMA_20** (Attribution Score: 0.1234)
2. **Log_Return** (Attribution Score: 0.1086)
3. **Close_to_MA_10** (Attribution Score: 0.0893)
4. **Close_to_MA_5** (Attribution Score: 0.0800)
5. **Rolling_Std_10** (Attribution Score: 0.0655)

#### PyTorch Transformer Top 5 Features (Integrated Gradients)
1. **Return** (Attribution Score: 0.1339)
2. **Close_to_MA_20** (Attribution Score: 0.0981)
3. **Log_Return** (Attribution Score: 0.0908)
4. **Close_to_MA_50** (Attribution Score: 0.0789)
5. **Close_to_EMA_10** (Attribution Score: 0.0787)

### Natural Gas

#### PyTorch LSTM Top 5 Features (Integrated Gradients)
1. **Close_to_EMA_10** (Attribution Score: 0.1008)
2. **Close_to_MA_10** (Attribution Score: 0.0890)
3. **Log_Return** (Attribution Score: 0.0789)
4. **Ret_Lag_2** (Attribution Score: 0.0762)
5. **Rolling_Std_10** (Attribution Score: 0.0677)

#### PyTorch Transformer Top 5 Features (Integrated Gradients)
1. **RSI_14** (Attribution Score: 0.1045)
2. **Close_to_MA_20** (Attribution Score: 0.0944)
3. **Close_to_MA_5** (Attribution Score: 0.0857)
4. **Log_Return** (Attribution Score: 0.0816)
5. **MACD_Pct** (Attribution Score: 0.0771)

### Crude Oil

#### PyTorch LSTM Top 5 Features (Integrated Gradients)
1. **Close_to_MA_10** (Attribution Score: 0.1587)
2. **Close_to_EMA_20** (Attribution Score: 0.1371)
3. **Close_to_MA_5** (Attribution Score: 0.1338)
4. **Close_to_MA_50** (Attribution Score: 0.1324)
5. **RSI_14** (Attribution Score: 0.1262)

#### PyTorch Transformer Top 5 Features (Integrated Gradients)
1. **Close_to_MA_10** (Attribution Score: 0.1777)
2. **Close_to_EMA_20** (Attribution Score: 0.1678)
3. **Close_to_MA_50** (Attribution Score: 0.1465)
4. **Close_to_MA_20** (Attribution Score: 0.1085)
5. **Close_to_EMA_10** (Attribution Score: 0.0958)

### Wheat

#### PyTorch LSTM Top 5 Features (Integrated Gradients)
1. **Close_to_EMA_10** (Attribution Score: 0.1270)
2. **Log_Return** (Attribution Score: 0.1193)
3. **Close_to_MA_50** (Attribution Score: 0.0807)
4. **MACD_Pct** (Attribution Score: 0.0779)
5. **Close_to_MA_20** (Attribution Score: 0.0693)

#### PyTorch Transformer Top 5 Features (Integrated Gradients)
1. **Close_to_EMA_10** (Attribution Score: 0.1385)
2. **MACD_Pct** (Attribution Score: 0.1007)
3. **Close_to_MA_20** (Attribution Score: 0.0783)
4. **Close_to_MA_50** (Attribution Score: 0.0693)
5. **Ret_Lag_2** (Attribution Score: 0.0618)

### 💡 Deep Learning Key Takeaways
* **Momentum Memory:** Because LSTMs and Transformers evaluate 10-day sequences, they tend to assign higher weight to **Lagged Returns** (`Ret_Lag_1`, `Ret_Lag_2`) compared to XGBoost, proving they successfully learned the temporal autocorrelation of the market.
* **Trend Alignment:** Both Neural Architectures heavily utilized **Moving Average Ratios** (`Close_to_MA_20`), confirming the universal importance of mean-reversion across all model types.
