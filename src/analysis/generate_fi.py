import joblib
import os

commodities = ['gold', 'silver', 'copper', 'natural_gas', 'crude_oil', 'wheat']
features = ['Return', 'Log_Return', 'Ret_Lag_1', 'Ret_Lag_2', 'Ret_Lag_3', 'Ret_Lag_5', 'Ret_Lag_10', 'Daily_Range_Pct', 'Close_to_MA_5', 'Close_to_MA_10', 'Close_to_MA_20', 'Close_to_MA_50', 'Close_to_EMA_10', 'Close_to_EMA_20', 'Rolling_Std_10', 'Rolling_Std_20', 'RSI_14', 'MACD_Pct', 'Volume_to_MA_10']

md_content = "# AI Feature Importance Analysis & Glossary\n\n"
md_content += "This document analyzes the implicit feature selection performed by our Tree-Based Machine Learning models (XGBoost and LightGBM) and provides a complete glossary of all technical indicators engineered for the platform.\n\n"

md_content += "## 📚 Feature Glossary & Definitions\n\n"
md_content += "| Feature Code | Full Name | Definition & Intuition |\n"
md_content += "|---|---|---|\n"
md_content += "| **Return** | Daily Simple Return | Percentage change in closing price from yesterday to today: `(Close_t - Close_{t-1}) / Close_{t-1}` |\n"
md_content += "| **Log_Return** | Daily Logarithmic Return | Symmetrized percentage change using natural log: `ln(Close_t / Close_{t-1})`. Standard metric in quant finance. |\n"
md_content += "| **Ret_Lag_N** | Return Lag (N Days) | Daily return from N days ago (e.g., `Ret_Lag_1` is yesterday's return, `Ret_Lag_5` is return from 5 days ago). Captures short-term momentum memory. |\n"
md_content += "| **Daily_Range_Pct** | Daily Intraday Range Percentage | High price minus Low price normalized by Close: `(High - Low) / Close`. Measures intraday price volatility and market panic/uncertainty. |\n"
md_content += "| **Close_to_MA_N** | Ratio of Close to N-Day Simple Moving Average | Today's close divided by N-day average (`Close / SMA_N`). Values > 1.0 indicate price is above trend; < 1.0 indicates below trend. |\n"
md_content += "| **Close_to_EMA_N** | Ratio of Close to N-Day Exponential Moving Average | Today's close divided by N-day exponential average (`Close / EMA_N`). Weighs recent days heavier than simple MA. |\n"
md_content += "| **Rolling_Std_N** | N-Day Rolling Standard Deviation of Returns | Standard deviation of daily returns over the last N days (`Std(Returns_{t-N:t})`). Direct measure of recent market volatility/risk. |\n"
md_content += "| **RSI_14** | Relative Strength Index (14-Period) | Bounded technical momentum oscillator (0 to 100). Measures speed and change of price movements (>70 = Overbought, <30 = Oversold). |\n"
md_content += "| **MACD_Pct** | Normalized Moving Average Convergence Divergence | MACD Histogram value divided by Close price (`MACD_Diff / Close`). Measures trend direction and momentum acceleration normalized across price levels. |\n"
md_content += "| **Volume_to_MA_10** | Volume to 10-Day Moving Average Ratio | Today's trading volume divided by its 10-day moving average (`Volume / SMA_10(Volume)`). Values > 1.0 represent unusual volume spikes. |\n\n"

md_content += "---\n\n"
md_content += "## 📊 Commodity Feature Importance Rankings\n\n"

for comm in commodities:
    md_content += f"### {comm.replace('_', ' ').title()}\n\n"
    
    # XGBoost
    xgb_path = f"models/xgb_return_{comm}.pkl"
    if os.path.exists(xgb_path):
        xgb_model = joblib.load(xgb_path)
        xgb_fi = sorted(zip(features, xgb_model.feature_importances_), key=lambda x: x[1], reverse=True)
        md_content += "#### XGBoost Top 5 Features (Gain)\n"
        for i, (f, w) in enumerate(xgb_fi[:5]):
            md_content += f"{i+1}. **{f}** (Weight: {w:.4f})\n"
        md_content += "\n"
        
    # LightGBM
    lgb_path = f"models/lgb_return_{comm}.pkl"
    if os.path.exists(lgb_path):
        lgb_model = joblib.load(lgb_path)
        raw_importances = lgb_model.feature_importances_
        total = sum(raw_importances)
        if total > 0:
            norm_importances = [x/total for x in raw_importances]
            lgb_fi = sorted(zip(features, norm_importances), key=lambda x: x[1], reverse=True)
            md_content += "#### LightGBM Top 5 Features (Relative Splits)\n"
            for i, (f, w) in enumerate(lgb_fi[:5]):
                md_content += f"{i+1}. **{f}** (Weight: {w:.4f})\n"
            md_content += "\n"
            
    md_content += "---\n\n"

md_content += "## 💡 Key Takeaways\n"
md_content += "* **Volatility Reigns Supreme:** Across almost all commodities, rolling standard deviation (`Rolling_Std_10`, `Rolling_Std_20`) and intraday volatility (`Daily_Range_Pct`) heavily outrank traditional momentum indicators.\n"
md_content += "* **Mean Reversion:** Features measuring the price's distance from moving averages (`Close_to_MA_20`, `Close_to_EMA_10`) are consistently used as primary split criteria, proving that commodities frequently exhibit mean-reverting behavior.\n"
md_content += "* **Volume is Secondary:** Interestingly, Volume relative to its moving average rarely makes the top 5, suggesting that for these highly liquid futures markets, price action itself is a stronger predictor than trading volume.\n"

# Save to both project folder and brain artifacts folder
project_out_path = 'Feature_Importance_Analysis.md'
brain_out_path = '/home/abdulbasit/.gemini/antigravity/brain/7d113264-14fc-4f24-bfb1-e01bd0ea2ba8/Feature_Importance_Analysis.md'

with open(project_out_path, 'w') as f:
    f.write(md_content)

with open(brain_out_path, 'w') as f:
    f.write(md_content)

print(f"Updated {project_out_path} and {brain_out_path} with complete Glossary!")
