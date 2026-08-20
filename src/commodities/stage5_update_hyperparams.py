import os
import glob

print("Starting Step 5: Hyperparameter Updates (Overfit Protection)...")

# 1. Update ML Generator Scripts
ml_scripts = glob.glob('create_ml_nb*.py')
for file in ml_scripts:
    with open(file, 'r') as f:
        content = f.read()
    
    # XGB & LGBM Updates
    content = content.replace("reg_lambda=1.0", "reg_lambda=3.0")
    content = content.replace("max_depth=6", "max_depth=5")
    
    # CatBoost Updates
    content = content.replace("l2_leaf_reg=3.0", "l2_leaf_reg=5.0")
    content = content.replace("depth=6", "depth=5")
    
    # Random Forest Updates
    content = content.replace("min_samples_leaf=10", "min_samples_leaf=15")
    
    with open(file, 'w') as f:
        f.write(content)
    
    # Execute to regenerate the notebook
    os.system(f"python3 {file}")

print("ML Notebooks regenerated with increased regularization.")

# 2. Update DL Generator Scripts
dl_scripts = glob.glob('create_dl_nb*.py')
for file in dl_scripts:
    with open(file, 'r') as f:
        content = f.read()
    
    # Increase dropout
    content = content.replace("Dropout(0.2)", "Dropout(0.25)")
    
    with open(file, 'w') as f:
        f.write(content)
        
    # Execute to regenerate the notebook
    os.system(f"python3 {file}")

print("DL Notebooks regenerated with increased dropout.")
print("All Champion vs Challenger Notebooks are ready for training!")
