"""
PHASE 4: MODEL TRAINING (FIXED FOR TIME-SERIES STABILITY)
- Regularized hyperparameters to prevent overfitting
- Target: log_gold_next (log scale)
- Back-transformation: exp(log_pred) for original scale RMSE/MAE
- Proper evaluation metrics (R² on log scale, RMSE/MAE on original scale)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PHASE 4: MODEL TRAINING")
print("=" * 80)

# Define paths
input_dir = Path("outputs")
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

print("\n[1] Loading train/validation/test splits...")

X_train = pd.read_csv(input_dir / "X_train.csv")
y_train_log = pd.read_csv(input_dir / "y_train.csv").iloc[:, 0]
X_val = pd.read_csv(input_dir / "X_val.csv")
y_val_log = pd.read_csv(input_dir / "y_val.csv").iloc[:, 0]
X_test = pd.read_csv(input_dir / "X_test.csv")
y_test_log = pd.read_csv(input_dir / "y_test.csv").iloc[:, 0]

y_train_original = np.exp(y_train_log)
y_val_original = np.exp(y_val_log)
y_test_original = np.exp(y_test_log)

with open(input_dir / "feature_columns.pkl", 'rb') as f:
    feature_cols = pickle.load(f)

print(f"  Train:   {X_train.shape} (y in log scale)")
print(f"  Validation: {X_val.shape} (y in log scale)")
print(f"  Test:    {X_test.shape} (y in log scale)")
print(f"\n  ⚠️  IMPORTANT: Predicting log scale, will back-transform for RMSE/MAE")

def evaluate_model(y_true_log, y_pred_log, y_true_original, model_name, dataset_name):
    """
    Evaluate model on both log scale (R²) and original scale (RMSE, MAE)
    y_true_log, y_pred_log: predictions in log scale
    y_true_original: actual gold prices in original scale (for RMSE/MAE)
    """
    r2_log = r2_score(y_true_log, y_pred_log)
    
    y_pred_original = np.exp(y_pred_log)
    
    rmse_original = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
    mae_original = mean_absolute_error(y_true_original, y_pred_original)
    
    print(f"\n  {model_name} - {dataset_name}:")
    print(f"    RMSE (original scale): {rmse_original:.6f}")
    print(f"    MAE  (original scale): {mae_original:.6f}")
    print(f"    R²   (log scale):      {r2_log:.6f}")
    
    return {'rmse': rmse_original, 'mae': mae_original, 'r2': r2_log}

# ============================================================================
# MODEL 1: XGBOOST REGRESSOR (REGULARIZED)
# ============================================================================
print("\n" + "=" * 80)
print("[2] Training XGBoost Regressor (Regularized)...")
print("=" * 80)

xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=3,              # FIXED: Reduced from 6 to 3 (prevent overfitting)
    learning_rate=0.03,       # FIXED: Reduced from 0.1 to 0.03 (slower learning)
    subsample=0.7,            # FIXED: Added subsample (prevent overfitting)
    colsample_bytree=0.7,     # FIXED: Added column subsampling
    reg_alpha=1.0,            # FIXED: Added L1 regularization
    reg_lambda=1.0,           # FIXED: Added L2 regularization
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train_log)

xgb_train_pred_log = xgb_model.predict(X_train)
xgb_val_pred_log = xgb_model.predict(X_val)
xgb_test_pred_log = xgb_model.predict(X_test)

xgb_train_metrics = evaluate_model(y_train_log, xgb_train_pred_log, y_train_original, "XGBoost", "Train")
xgb_val_metrics = evaluate_model(y_val_log, xgb_val_pred_log, y_val_original, "XGBoost", "Validation")
xgb_test_metrics = evaluate_model(y_test_log, xgb_test_pred_log, y_test_original, "XGBoost", "Test")

# ============================================================================
# MODEL 2: LIGHTGBM REGRESSOR (REGULARIZED)
# ============================================================================
print("\n" + "=" * 80)
print("[3] Training LightGBM Regressor (Regularized)...")
print("=" * 80)

lgb_model = lgb.LGBMRegressor(
    n_estimators=300,
    num_leaves=15,        
    max_depth=3,              
    learning_rate=0.03,      
    subsample=0.7,           
    colsample_bytree=0.7,    
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_model.fit(X_train, y_train_log)

lgb_train_pred_log = lgb_model.predict(X_train)
lgb_val_pred_log = lgb_model.predict(X_val)
lgb_test_pred_log = lgb_model.predict(X_test)

lgb_train_metrics = evaluate_model(y_train_log, lgb_train_pred_log, y_train_original, "LightGBM", "Train")
lgb_val_metrics = evaluate_model(y_val_log, lgb_val_pred_log, y_val_original, "LightGBM", "Validation")
lgb_test_metrics = evaluate_model(y_test_log, lgb_test_pred_log, y_test_original, "LightGBM", "Test")

# ============================================================================
# MODEL 3: RANDOM FOREST REGRESSOR (REGULARIZED)
# ============================================================================
print("\n" + "=" * 80)
print("[4] Training Random Forest Regressor (Regularized)...")
print("=" * 80)

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=5,              # FIXED: Reduced from 10 to 5 (prevent overfitting)
    min_samples_leaf=20,      # FIXED: Added min_samples_leaf (require more samples per leaf)
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train_log)

rf_train_pred_log = rf_model.predict(X_train)
rf_val_pred_log = rf_model.predict(X_val)
rf_test_pred_log = rf_model.predict(X_test)

rf_train_metrics = evaluate_model(y_train_log, rf_train_pred_log, y_train_original, "Random Forest", "Train")
rf_val_metrics = evaluate_model(y_val_log, rf_val_pred_log, y_val_original, "Random Forest", "Validation")
rf_test_metrics = evaluate_model(y_test_log, rf_test_pred_log, y_test_original, "Random Forest", "Test")

print("\n" + "=" * 80)
print("[5] Training Support Vector Regression (Baseline - Regularized)...")
print("=" * 80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

svr_model = SVR(
    kernel='rbf',
    C=10,
    gamma=0.1,
    epsilon=0.01
)
svr_model.fit(X_train_scaled, y_train_log)

svr_train_pred_log = svr_model.predict(X_train_scaled)
svr_val_pred_log = svr_model.predict(X_val_scaled)
svr_test_pred_log = svr_model.predict(X_test_scaled)

svr_train_metrics = evaluate_model(y_train_log, svr_train_pred_log, y_train_original, "SVR", "Train")
svr_val_metrics = evaluate_model(y_val_log, svr_val_pred_log, y_val_original, "SVR", "Validation")
svr_test_metrics = evaluate_model(y_test_log, svr_test_pred_log, y_test_original, "SVR", "Test")

# ============================================================================
# NAÏVE BASELINE MODEL (gold(t+1) = gold(t))
# ============================================================================
print("\n" + "=" * 80)
print("[6] NAÏVE BASELINE EVALUATION (gold(t+1) = gold(t))")
print("=" * 80)

master_df_for_naive = pd.read_csv(input_dir / "master_df.csv")
master_df_for_naive['year_month'] = pd.to_datetime(master_df_for_naive['year_month'])

master_df_for_naive['log_gold'] = np.log(master_df_for_naive['gold_usd_per_gram'])

train_df = pd.read_csv(input_dir / "train_df.csv")
val_df = pd.read_csv(input_dir / "val_df.csv")
test_df = pd.read_csv(input_dir / "test_df.csv")
train_df['year_month'] = pd.to_datetime(train_df['year_month'])
val_df['year_month'] = pd.to_datetime(val_df['year_month'])
test_df['year_month'] = pd.to_datetime(test_df['year_month'])

train_merged = train_df[['year_month']].merge(master_df_for_naive[['year_month', 'log_gold']], on='year_month', how='left')
val_merged = val_df[['year_month']].merge(master_df_for_naive[['year_month', 'log_gold']], on='year_month', how='left')
test_merged = test_df[['year_month']].merge(master_df_for_naive[['year_month', 'log_gold']], on='year_month', how='left')

train_naive_pred_log = train_merged['log_gold'].values
val_naive_pred_log = val_merged['log_gold'].values
test_naive_pred_log = test_merged['log_gold'].values

min_len_train = min(len(train_naive_pred_log), len(y_train_log))
train_naive_pred_log = train_naive_pred_log[:min_len_train]
y_train_log_for_naive = y_train_log[:min_len_train]
y_train_original_for_naive = y_train_original[:min_len_train]

min_len_val = min(len(val_naive_pred_log), len(y_val_log))
val_naive_pred_log = val_naive_pred_log[:min_len_val]
y_val_log_for_naive = y_val_log[:min_len_val]
y_val_original_for_naive = y_val_original[:min_len_val]

min_len_test = min(len(test_naive_pred_log), len(y_test_log))
test_naive_pred_log = test_naive_pred_log[:min_len_test]
y_test_log_for_naive = y_test_log[:min_len_test]
y_test_original_for_naive = y_test_original[:min_len_test]

naive_train_metrics = evaluate_model(y_train_log_for_naive, train_naive_pred_log, y_train_original_for_naive, "Naïve Baseline", "Train")
naive_val_metrics = evaluate_model(y_val_log_for_naive, val_naive_pred_log, y_val_original_for_naive, "Naïve Baseline", "Validation")
naive_test_metrics = evaluate_model(y_test_log_for_naive, test_naive_pred_log, y_test_original_for_naive, "Naïve Baseline", "Test")

# ============================================================================
# MODEL COMPARISON (BEST MODEL BY VALIDATION MAE)
# ============================================================================
print("\n" + "=" * 80)
print("[7] MODEL COMPARISON (Best Model by Validation MAE - Original Scale)")
print("=" * 80)

models_metrics_mae = {
    'XGBoost': xgb_val_metrics['mae'],
    'LightGBM': lgb_val_metrics['mae'],
    'Random Forest': rf_val_metrics['mae'],
    'SVR': svr_val_metrics['mae']
}

sorted_models_mae = sorted(models_metrics_mae.items(), key=lambda x: x[1])
print("\n  Models ranked by Validation MAE (original scale, lowest is best):")
for i, (model_name, mae) in enumerate(sorted_models_mae, 1):
    print(f"    {i}. {model_name}: {mae:.6f} USD/gram")

best_model_name = sorted_models_mae[0][0]
best_val_mae = sorted_models_mae[0][1]
print(f"\n  ✓ Best Model: {best_model_name} (Lowest Validation MAE: {best_val_mae:.6f} USD/gram)")

print("\n  📊 ML Models vs Naïve Baseline Comparison (Validation MAE):")
print(f"    Naïve Baseline: {naive_val_metrics['mae']:.6f} USD/gram")
for model_name, mae in sorted_models_mae:
    improvement = ((naive_val_metrics['mae'] - mae) / naive_val_metrics['mae']) * 100
    status = "✓ IMPROVES" if improvement > 0 else "✗ WORSE"
    print(f"    {model_name}: {mae:.6f} USD/gram ({improvement:+.2f}% vs baseline) {status}")

print("\n  Validation R² (log scale) comparison:")
models_r2 = {
    'XGBoost': xgb_val_metrics['r2'],
    'LightGBM': lgb_val_metrics['r2'],
    'Random Forest': rf_val_metrics['r2'],
    'SVR': svr_val_metrics['r2'],
    'Naïve Baseline': naive_val_metrics['r2']
}
for model_name, r2 in models_r2.items():
    print(f"    {model_name}: {r2:.6f}")

# ============================================================================
# SAVE MODELS (INDIVIDUAL FILES)
# ============================================================================
print("\n" + "=" * 80)
print("[8] Saving models as individual files...")
print("=" * 80)

with open(models_dir / "xgboost_model.pkl", 'wb') as f:
    pickle.dump(xgb_model, f)
print("  ✓ XGBoost model saved to models/xgboost_model.pkl")

with open(models_dir / "lightgbm_model.pkl", 'wb') as f:
    pickle.dump(lgb_model, f)
print("  ✓ LightGBM model saved to models/lightgbm_model.pkl")

with open(models_dir / "random_forest_model.pkl", 'wb') as f:
    pickle.dump(rf_model, f)
print("  ✓ Random Forest model saved to models/random_forest_model.pkl")

with open(models_dir / "svr_model.pkl", 'wb') as f:
    pickle.dump(svr_model, f)
print("  ✓ SVR model saved to models/svr_model.pkl")

with open(models_dir / "scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)
print("  ✓ Scaler saved to models/scaler.pkl")

with open(output_dir / "feature_columns.pkl", 'wb') as f:
    pickle.dump(feature_cols, f)
print("  ✓ Feature columns saved to outputs/feature_columns.pkl")

results_df = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM', 'Random Forest', 'SVR', 'Naïve Baseline'],
    'Train_RMSE': [xgb_train_metrics['rmse'], lgb_train_metrics['rmse'], 
                   rf_train_metrics['rmse'], svr_train_metrics['rmse'], naive_train_metrics['rmse']],
    'Train_MAE': [xgb_train_metrics['mae'], lgb_train_metrics['mae'], 
                  rf_train_metrics['mae'], svr_train_metrics['mae'], naive_train_metrics['mae']],
    'Train_R2': [xgb_train_metrics['r2'], lgb_train_metrics['r2'], 
                 rf_train_metrics['r2'], svr_train_metrics['r2'], naive_train_metrics['r2']],
    'Val_RMSE': [xgb_val_metrics['rmse'], lgb_val_metrics['rmse'], 
                 rf_val_metrics['rmse'], svr_val_metrics['rmse'], naive_val_metrics['rmse']],
    'Val_MAE': [xgb_val_metrics['mae'], lgb_val_metrics['mae'], 
                rf_val_metrics['mae'], svr_val_metrics['mae'], naive_val_metrics['mae']],
    'Val_R2': [xgb_val_metrics['r2'], lgb_val_metrics['r2'], 
               rf_val_metrics['r2'], svr_val_metrics['r2'], naive_val_metrics['r2']],
    'Test_RMSE': [xgb_test_metrics['rmse'], lgb_test_metrics['rmse'], 
                  rf_test_metrics['rmse'], svr_test_metrics['rmse'], naive_test_metrics['rmse']],
    'Test_MAE': [xgb_test_metrics['mae'], lgb_test_metrics['mae'], 
                 rf_test_metrics['mae'], svr_test_metrics['mae'], naive_test_metrics['mae']],
    'Test_R2': [xgb_test_metrics['r2'], lgb_test_metrics['r2'], 
                rf_test_metrics['r2'], svr_test_metrics['r2'], naive_test_metrics['r2']]
})

print("\n  📊 Evaluation Metrics Note:")
print("    - RMSE/MAE: On original scale (USD per gram)")
print("    - R²: On log scale (proper evaluation metric)")

results_df.to_csv(output_dir / "model_evaluation_results.csv", index=False)
print("  ✓ Evaluation results saved to outputs/model_evaluation_results.csv")

print("\n" + "=" * 80)
print("PHASE 4 COMPLETE!")
print("=" * 80)

