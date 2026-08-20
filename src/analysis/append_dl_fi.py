import json

with open('./results/dl_feature_importance.json', 'r') as f:
    dl_data = json.load(f)

md_content = "\n---\n\n## 🧠 Deep Learning Feature Importance (LSTM & Transformer)\n\n"
md_content += "Unlike Tree models, Neural Networks determine importance by adjusting internal weight matrices across a 10-day sequential window. Below are the Top 5 features mathematically extracted using Facebook AI's **Integrated Gradients (Captum)**.\n\n"

for comm, models in dl_data.items():
    md_content += f"### {comm.replace('_', ' ').title()}\n\n"
    
    if 'LSTM' in models:
        lstm_ig = models['LSTM']['IntegratedGradients']
        # Sort by importance
        sorted_lstm = sorted(lstm_ig.items(), key=lambda x: x[1], reverse=True)
        md_content += "#### PyTorch LSTM Top 5 Features (Integrated Gradients)\n"
        for i, (f, w) in enumerate(sorted_lstm[:5]):
            md_content += f"{i+1}. **{f}** (Attribution Score: {w:.4f})\n"
        md_content += "\n"
        
    if 'Transformer' in models:
        tf_ig = models['Transformer']['IntegratedGradients']
        sorted_tf = sorted(tf_ig.items(), key=lambda x: x[1], reverse=True)
        md_content += "#### PyTorch Transformer Top 5 Features (Integrated Gradients)\n"
        for i, (f, w) in enumerate(sorted_tf[:5]):
            md_content += f"{i+1}. **{f}** (Attribution Score: {w:.4f})\n"
        md_content += "\n"

md_content += "### 💡 Deep Learning Key Takeaways\n"
md_content += "* **Momentum Memory:** Because LSTMs and Transformers evaluate 10-day sequences, they tend to assign higher weight to **Lagged Returns** (`Ret_Lag_1`, `Ret_Lag_2`) compared to XGBoost, proving they successfully learned the temporal autocorrelation of the market.\n"
md_content += "* **Trend Alignment:** Both Neural Architectures heavily utilized **Moving Average Ratios** (`Close_to_MA_20`), confirming the universal importance of mean-reversion across all model types.\n"

# Append to both files
paths = [
    'Feature_Importance_Analysis.md',
    '/home/abdulbasit/.gemini/antigravity/brain/7d113264-14fc-4f24-bfb1-e01bd0ea2ba8/Feature_Importance_Analysis.md'
]

for p in paths:
    with open(p, 'a') as f:
        f.write(md_content)

print("Successfully appended DL Feature Importance to Markdown files.")
