"""
PHASE 4: MODEL TRAINING (RETURN-BASED, ANTI-OVERFITTING)
- Target: gold_return_next (monthly log return)
- Back-transformation: current_gold * exp(predicted_return) for RMSE/MAE
- R² computed on returns (stationary scale)

Anti-overfitting measures:
- XGBoost/LightGBM: early stopping on validation set; shallower trees (max_depth=2);
  stronger L1/L2; lower learning_rate; more subsampling.
- Random Forest: max_depth=3, min_samples_leaf=50, min_samples_split=100.
- Ridge baseline added for comparison (no overfitting).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
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
y_train_return = pd.read_csv(input_dir / "y_train.csv").iloc[:, 0]   # log returns
X_val = pd.read_csv(input_dir / "X_val.csv")
y_val_return = pd.read_csv(input_dir / "y_val.csv").iloc[:, 0]
X_test = pd.read_csv(input_dir / "X_test.csv")
y_test_return = pd.read_csv(input_dir / "y_test.csv").iloc[:, 0]

with open(input_dir / "feature_columns.pkl", 'rb') as f:
    feature_cols = pickle.load(f)

# Load current gold prices for back-transformation: price(t+1) = gold(t) * exp(return)
master_raw = pd.read_csv(input_dir / "master_df.csv")[['year_month', 'gold_usd_per_gram']]
train_df_ym = pd.read_csv(input_dir / "train_df.csv")[['year_month']]
val_df_ym   = pd.read_csv(input_dir / "val_df.csv")[['year_month']]
test_df_ym  = pd.read_csv(input_dir / "test_df.csv")[['year_month']]

gold_train = train_df_ym.merge(master_raw, on='year_month', how='left')['gold_usd_per_gram'].values
gold_val   = val_df_ym.merge(master_raw, on='year_month', how='left')['gold_usd_per_gram'].values
gold_test  = test_df_ym.merge(master_raw, on='year_month', how='left')['gold_usd_per_gram'].values

print(f"  Train:      {X_train.shape} | gold range ${gold_train.min():.2f}–${gold_train.max():.2f}/gram")
print(f"  Validation: {X_val.shape} | gold range ${gold_val.min():.2f}–${gold_val.max():.2f}/gram")
print(f"  Test:       {X_test.shape} | gold range ${gold_test.min():.2f}–${gold_test.max():.2f}/gram")
print(f"\n  NOTE: TARGET: gold_return_next (log return). Back-transform: current_gold * exp(return)")

def evaluate_model(y_true_return, y_pred_return, current_gold, model_name, dataset_name):
    """
    Evaluate model on return scale (R²) and original price scale (RMSE, MAE).
    y_true_return, y_pred_return: monthly log returns
    current_gold: gold price at time t — used for back-transform to t+1 price
    """
    r2_return = r2_score(y_true_return, y_pred_return)

    # Back-transform: predicted_price(t+1) = current_gold(t) * exp(predicted_return)
    y_pred_price = current_gold * np.exp(y_pred_return)
    y_true_price = current_gold * np.exp(y_true_return)

    rmse = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
    mae  = mean_absolute_error(y_true_price, y_pred_price)

    print(f"\n  {model_name} - {dataset_name}:")
    print(f"    RMSE (USD/gram, original scale): {rmse:.4f}")
    print(f"    MAE  (USD/gram, original scale): {mae:.4f}")
    print(f"    R²   (returns scale):            {r2_return:.4f}")

    return {'rmse': rmse, 'mae': mae, 'r2': r2_return}

# ============================================================================
# MODEL 1: XGBOOST REGRESSOR (ANTI-OVERFIT: early stop + strong regularization)
# ============================================================================
print("\n" + "=" * 80)
print("[2] Training XGBoost Regressor (early stopping + regularized)...")
print("=" * 80)

xgb_model = xgb.XGBRegressor(
    n_estimators=500,         # Allow more rounds; early stop will cut when val stops improving
    max_depth=2,              # Shallower trees = less overfitting
    learning_rate=0.02,
    subsample=0.6,
    colsample_bytree=0.6,
    reg_alpha=2.0,
    reg_lambda=2.0,
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train, y_train_return,
    eval_set=[(X_val, y_val_return)],
    verbose=False
)

xgb_train_pred = xgb_model.predict(X_train)
xgb_val_pred   = xgb_model.predict(X_val)
xgb_test_pred  = xgb_model.predict(X_test)

xgb_train_metrics = evaluate_model(y_train_return, xgb_train_pred, gold_train, "XGBoost", "Train")
xgb_val_metrics   = evaluate_model(y_val_return,   xgb_val_pred,   gold_val,   "XGBoost", "Validation")
xgb_test_metrics  = evaluate_model(y_test_return,  xgb_test_pred,  gold_test,  "XGBoost", "Test")

# ============================================================================
# MODEL 2: LIGHTGBM REGRESSOR (ANTI-OVERFIT: early stop + strong regularization)
# ============================================================================
print("\n" + "=" * 80)
print("[3] Training LightGBM Regressor (early stopping + regularized)...")
print("=" * 80)

lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    num_leaves=7,              # Fewer leaves = simpler trees
    max_depth=2,
    learning_rate=0.02,
    subsample=0.6,
    colsample_bytree=0.6,
    reg_alpha=2.0,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_model.fit(
    X_train, y_train_return,
    eval_set=[(X_val, y_val_return)],
    callbacks=[lgb.early_stopping(50, verbose=False)]
)

lgb_train_pred = lgb_model.predict(X_train)
lgb_val_pred   = lgb_model.predict(X_val)
lgb_test_pred  = lgb_model.predict(X_test)

lgb_train_metrics = evaluate_model(y_train_return, lgb_train_pred, gold_train, "LightGBM", "Train")
lgb_val_metrics   = evaluate_model(y_val_return,   lgb_val_pred,   gold_val,   "LightGBM", "Validation")
lgb_test_metrics  = evaluate_model(y_test_return,  lgb_test_pred,  gold_test,  "LightGBM", "Test")

# ============================================================================
# MODEL 3: RANDOM FOREST REGRESSOR (ANTI-OVERFIT: shallower, larger leaves)
# ============================================================================
print("\n" + "=" * 80)
print("[4] Training Random Forest Regressor (strong regularization)...")
print("=" * 80)

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=3,
    min_samples_leaf=50,      # Require many samples per leaf = less overfitting
    min_samples_split=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train_return)

rf_train_pred = rf_model.predict(X_train)
rf_val_pred   = rf_model.predict(X_val)
rf_test_pred  = rf_model.predict(X_test)

rf_train_metrics = evaluate_model(y_train_return, rf_train_pred, gold_train, "Random Forest", "Train")
rf_val_metrics   = evaluate_model(y_val_return,   rf_val_pred,   gold_val,   "Random Forest", "Validation")
rf_test_metrics  = evaluate_model(y_test_return,  rf_test_pred,  gold_test,  "Random Forest", "Test")


# ============================================================================
# RIDGE BASELINE (linear model - no overfitting, reference for signal strength)
# ============================================================================
print("\n" + "=" * 80)
print("[4b] Ridge Regression (linear baseline)...")
print("=" * 80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

ridge_model = Ridge(alpha=10.0, random_state=42)
ridge_model.fit(X_train_scaled, y_train_return)

ridge_train_pred = ridge_model.predict(X_train_scaled)
ridge_val_pred   = ridge_model.predict(X_val_scaled)
ridge_test_pred  = ridge_model.predict(X_test_scaled)

ridge_train_metrics = evaluate_model(y_train_return, ridge_train_pred, gold_train, "Ridge", "Train")
ridge_val_metrics   = evaluate_model(y_val_return,   ridge_val_pred,   gold_val,   "Ridge", "Validation")
ridge_test_metrics  = evaluate_model(y_test_return,  ridge_test_pred,  gold_test,  "Ridge", "Test")

# ============================================================================
# ENSEMBLE (AVERAGE OF XGBOOST, LIGHTGBM, RF)
# ============================================================================
print("\n" + "=" * 80)
print("[5] Ensemble Model (Average of XGBoost, LightGBM, RF)...")
print("=" * 80)

ensemble_train_pred = (xgb_train_pred + lgb_train_pred + rf_train_pred) / 3
ensemble_val_pred   = (xgb_val_pred   + lgb_val_pred   + rf_val_pred)   / 3
ensemble_test_pred  = (xgb_test_pred  + lgb_test_pred  + rf_test_pred)  / 3

ensemble_train_metrics = evaluate_model(y_train_return, ensemble_train_pred, gold_train, "Ensemble", "Train")
ensemble_val_metrics   = evaluate_model(y_val_return,   ensemble_val_pred,   gold_val,   "Ensemble", "Validation")
ensemble_test_metrics  = evaluate_model(y_test_return,  ensemble_test_pred,  gold_test,  "Ensemble", "Test")

# ============================================================================
# NAÏVE BASELINE MODEL (return = 0, i.e. gold(t+1) = gold(t))
# ============================================================================
print("\n" + "=" * 80)
print("[6] NAIVE BASELINE EVALUATION (predicted return = 0, i.e. price unchanged)")
print("=" * 80)

# Naive baseline: predict return = 0 (no price change)
naive_train_pred = np.zeros(len(y_train_return))
naive_val_pred   = np.zeros(len(y_val_return))
naive_test_pred  = np.zeros(len(y_test_return))

naive_train_metrics = evaluate_model(y_train_return, naive_train_pred, gold_train, "Naive Baseline", "Train")
naive_val_metrics   = evaluate_model(y_val_return,   naive_val_pred,   gold_val,   "Naive Baseline", "Validation")
naive_test_metrics  = evaluate_model(y_test_return,  naive_test_pred,  gold_test,  "Naive Baseline", "Test")

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
    'Ensemble': ensemble_val_metrics['mae']
}

sorted_models_mae = sorted(models_metrics_mae.items(), key=lambda x: x[1])
print("\n  Models ranked by Validation MAE (original scale, lowest is best):")
for i, (model_name, mae) in enumerate(sorted_models_mae, 1):
    print(f"    {i}. {model_name}: {mae:.4f} USD/gram")

best_model_name = sorted_models_mae[0][0]
best_val_mae = sorted_models_mae[0][1]
print(f"\n  Best Model: {best_model_name} (Lowest Validation MAE: {best_val_mae:.4f} USD/gram)")

print("\n  ML Models vs Naive Baseline Comparison (Validation MAE):")
print(f"    Naive Baseline: {naive_val_metrics['mae']:.4f} USD/gram")
for model_name, mae in sorted_models_mae:
    improvement = ((naive_val_metrics['mae'] - mae) / naive_val_metrics['mae']) * 100
    status = "IMPROVES" if improvement > 0 else "WORSE"
    print(f"    {model_name}: {mae:.4f} ({improvement:+.1f}% vs baseline) {status}")

print("\n  Validation R2 (returns scale) comparison:")
models_r2 = {
    'XGBoost': xgb_val_metrics['r2'],
    'LightGBM': lgb_val_metrics['r2'],
    'Random Forest': rf_val_metrics['r2'],
    'Ensemble': ensemble_val_metrics['r2'],
    'Ridge': ridge_val_metrics['r2'],
    'Naive Baseline': naive_val_metrics['r2']
}
for model_name, r2 in models_r2.items():
    print(f"    {model_name}: {r2:.4f}")

print("\n  Train vs Val MAE (overfitting check; smaller gap = better generalization):")
for name, train_m, val_m in [
    ('XGBoost', xgb_train_metrics['mae'], xgb_val_metrics['mae']),
    ('LightGBM', lgb_train_metrics['mae'], lgb_val_metrics['mae']),
    ('Random Forest', rf_train_metrics['mae'], rf_val_metrics['mae']),
    ('Ridge', ridge_train_metrics['mae'], ridge_val_metrics['mae']),
]:
    gap = val_m - train_m
    print(f"    {name}: Train MAE={train_m:.4f}, Val MAE={val_m:.4f}, gap={gap:.4f}")

# ============================================================================
# SAVE MODELS (INDIVIDUAL FILES)
# ============================================================================
print("\n" + "=" * 80)
print("[8] Saving models as individual files...")
print("=" * 80)

with open(models_dir / "xgboost_model.pkl", 'wb') as f:
    pickle.dump(xgb_model, f)
print("  XGBoost model saved to models/xgboost_model.pkl")

with open(models_dir / "lightgbm_model.pkl", 'wb') as f:
    pickle.dump(lgb_model, f)
print("  LightGBM model saved to models/lightgbm_model.pkl")

with open(models_dir / "random_forest_model.pkl", 'wb') as f:
    pickle.dump(rf_model, f)
print("  Random Forest model saved to models/random_forest_model.pkl")

with open(output_dir / "feature_columns.pkl", 'wb') as f:
    pickle.dump(feature_cols, f)
print("  Feature columns saved to outputs/feature_columns.pkl")

results_df = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM', 'Random Forest', 'Ensemble', 'Ridge', 'Naive Baseline'],
    'Train_RMSE': [xgb_train_metrics['rmse'], lgb_train_metrics['rmse'],
                   rf_train_metrics['rmse'],  ensemble_train_metrics['rmse'],
                   ridge_train_metrics['rmse'], naive_train_metrics['rmse']],
    'Train_MAE':  [xgb_train_metrics['mae'],  lgb_train_metrics['mae'],
                   rf_train_metrics['mae'],   ensemble_train_metrics['mae'],
                   ridge_train_metrics['mae'],  naive_train_metrics['mae']],
    'Train_R2':   [xgb_train_metrics['r2'],   lgb_train_metrics['r2'],
                   rf_train_metrics['r2'],    ensemble_train_metrics['r2'],
                   ridge_train_metrics['r2'],   naive_train_metrics['r2']],
    'Val_RMSE':   [xgb_val_metrics['rmse'],   lgb_val_metrics['rmse'],
                   rf_val_metrics['rmse'],    ensemble_val_metrics['rmse'],
                   ridge_val_metrics['rmse'],   naive_val_metrics['rmse']],
    'Val_MAE':    [xgb_val_metrics['mae'],    lgb_val_metrics['mae'],
                   rf_val_metrics['mae'],     ensemble_val_metrics['mae'],
                   ridge_val_metrics['mae'],    naive_val_metrics['mae']],
    'Val_R2':     [xgb_val_metrics['r2'],     lgb_val_metrics['r2'],
                   rf_val_metrics['r2'],      ensemble_val_metrics['r2'],
                   ridge_val_metrics['r2'],     naive_val_metrics['r2']],
    'Test_RMSE':  [xgb_test_metrics['rmse'],  lgb_test_metrics['rmse'],
                   rf_test_metrics['rmse'],   ensemble_test_metrics['rmse'],
                   ridge_test_metrics['rmse'],  naive_test_metrics['rmse']],
    'Test_MAE':   [xgb_test_metrics['mae'],   lgb_test_metrics['mae'],
                   rf_test_metrics['mae'],    ensemble_test_metrics['mae'],
                   ridge_test_metrics['mae'],   naive_test_metrics['mae']],
    'Test_R2':    [xgb_test_metrics['r2'],    lgb_test_metrics['r2'],
                   rf_test_metrics['r2'],     ensemble_test_metrics['r2'],
                   ridge_test_metrics['r2'],    naive_test_metrics['r2']]
})

print("\n  Evaluation Metrics Note:")
print("    RMSE/MAE: Original scale (USD/gram) via back-transform: current_gold * exp(return)")
print("    R2:       On returns scale (stationary)")

results_df.to_csv(output_dir / "model_evaluation_results.csv", index=False)
print("  OK: Evaluation results saved to outputs/model_evaluation_results.csv")

print("\n" + "=" * 80)
print("PHASE 4 COMPLETE!")
print("=" * 80)

