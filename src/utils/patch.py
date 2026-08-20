import re

with open('ensemble_inference.py', 'r') as f:
    code = f.read()

replacement = """def load_and_predict(model_name, commodity, horizon, X_ml, X_dl_tensor, input_size):
    \"\"\"Loads the specific artifact from disk and predicts the return.\"\"\"
    device = torch.device('cpu')
    pred_return = None
    
    # XGBoost
    if model_name == 'XGBoost':
        model = XGBRegressor()
        model.load_model(os.path.join(MODELS_DIR, f'{commodity}_xgboost_{horizon}d.json'))
        try:
            pred_return = model.predict(X_ml)[0]
        except ValueError:
            # Fallback for reverted champions expecting 27 features
            pred_return = model.predict(X_ml[:, :27])[0]
        
    # LightGBM
    elif model_name == 'LightGBM':
        model = Booster(model_file=os.path.join(MODELS_DIR, f'{commodity}_lightgbm_{horizon}d.txt'))
        if model.num_feature() == 27 and X_ml.shape[1] > 27:
            X_ml = X_ml[:, :27]
        pred_return = model.predict(X_ml)[0]
        
    # CatBoost
    elif model_name == 'CatBoost':
        model = CatBoostRegressor()
        model.load_model(os.path.join(MODELS_DIR, f'{commodity}_catboost_{horizon}d.cbm'))
        if len(model.feature_names_) == 27 and X_ml.shape[1] > 27:
            X_ml = X_ml[:, :27]
        pred_return = model.predict(X_ml)[0]
        
    # RandomForest
    elif model_name == 'RandomForest':
        model = joblib.load(os.path.join(MODELS_DIR, f'{commodity}_randomforest_{horizon}d.pkl'))
        if model.n_features_in_ == 27 and X_ml.shape[1] > 27:
            X_ml = X_ml[:, :27]
        pred_return = model.predict(X_ml)[0]
        
    # Deep Learning PyTorch Models
    else:
        model_path = os.path.join(MODELS_DIR, f'{commodity}_{model_name.lower().replace("-","")}_{horizon}d.pt')
        state_dict = torch.load(model_path, map_location=device)
        
        # Dynamically detect if this is a reverted champion expecting 27 features
        expected_input = input_size
        if 'lstm.weight_ih_l0' in state_dict:
            expected_input = state_dict['lstm.weight_ih_l0'].shape[1]
        elif 'gru.weight_ih_l0' in state_dict:
            expected_input = state_dict['gru.weight_ih_l0'].shape[1]
        elif 'input_proj.weight' in state_dict:
            expected_input = state_dict['input_proj.weight'].shape[1]
        elif 'block1.fc1.weight' in state_dict:
            expected_input = state_dict['block1.fc1.weight'].shape[1] // SEQ_LENGTH
            
        if expected_input == 27 and input_size > 27:
            X_dl_tensor = X_dl_tensor[:, :, :27]
            dynamic_input_size = 27
        else:
            dynamic_input_size = input_size

        if model_name == 'LSTM':
            model = LSTMModel(dynamic_input_size, HIDDEN_SIZE)
        elif model_name == 'GRU':
            model = GRUModel(dynamic_input_size, HIDDEN_SIZE)
        elif model_name == 'Transformer':
            model = TransformerModel(dynamic_input_size, HIDDEN_SIZE)
        elif model_name == 'N-BEATS':
            model = NBeatsModel(SEQ_LENGTH, dynamic_input_size)
        elif model_name == 'TFT':
            model = TFTModel(dynamic_input_size, HIDDEN_SIZE)
        else:
            raise ValueError(f"Unknown model architecture: {model_name}")
            
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            pred_return = model(X_dl_tensor).squeeze().item()
            
    return pred_return, model"""

pattern = r'def load_and_predict.*?return pred_return, model'
new_code = re.sub(pattern, replacement, code, flags=re.DOTALL)

with open('ensemble_inference.py', 'w') as f:
    f.write(new_code)
