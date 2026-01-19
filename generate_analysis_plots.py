"""
Generate analysis plots and data for the Flask about page:
- Correlation Analysis
- Feature Importance for all models
- Model predictions comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from sklearn.inspection import permutation_importance

print("=" * 80)
print("GENERATING ANALYSIS PLOTS AND DATA")
print("=" * 80)

input_dir = Path("outputs")
models_dir = Path("models")
plots_dir = Path("static")
plots_dir.mkdir(exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("\n[1] Loading data and models...")

features_df = pd.read_csv(input_dir / "features_df.csv")
test_df = pd.read_csv(input_dir / "test_df.csv")
X_test = pd.read_csv(input_dir / "X_test.csv")
y_test = pd.read_csv(input_dir / "y_test.csv").iloc[:, 0]

print("  Loading models from individual files...")
with open(models_dir / "xgboost_model.pkl", 'rb') as f:
    xgb_model = pickle.load(f)
with open(models_dir / "lightgbm_model.pkl", 'rb') as f:
    lgb_model = pickle.load(f)
with open(models_dir / "random_forest_model.pkl", 'rb') as f:
    rf_model = pickle.load(f)
with open(models_dir / "svr_model.pkl", 'rb') as f:
    svr_model = pickle.load(f)
with open(models_dir / "scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

with open(input_dir / "feature_columns.pkl", 'rb') as f:
    feature_cols = pickle.load(f)

models_dict = {
    'xgboost': xgb_model,
    'lightgbm': lgb_model,
    'random_forest': rf_model,
    'svr': svr_model,
    'scaler': scaler,
    'feature_columns': feature_cols
}

print("  ✓ Data and models loaded")

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================
print("\n[2] Generating correlation analysis...")

master_df = pd.read_csv(input_dir / "master_df.csv")
master_df['year_month'] = pd.to_datetime(master_df['year_month'])

if 'log_gold' not in master_df.columns:
    master_df['log_gold'] = np.log(master_df['gold_usd_per_gram'])

if 'cpi_yoy' not in master_df.columns:
    master_df['cpi_yoy'] = (master_df['cpi'] / master_df['cpi'].shift(12) - 1) * 100

if 'real_rate' not in master_df.columns:
    master_df['real_rate'] = master_df['fedfunds'] - master_df['cpi_yoy']

corr_cols = ['log_gold', 'cpi_yoy', 'real_rate', 'fedfunds']
gold_label = 'Log Gold Price'

corr_data = master_df[corr_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title(f'Correlation Analysis: {gold_label} vs Macro Indicators', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(plots_dir / "correlation_analysis.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Correlation heatmap saved")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print("\n[3] Generating feature importance plots...")

models_names = {
    'xgboost': 'XGBoost',
    'lightgbm': 'LightGBM',
    'random_forest': 'Random Forest'
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes = axes.flatten()

for idx, (model_key, model_name) in enumerate(models_names.items()):
    model = models_dict[model_key]
    
    if model_key == 'xgboost':
        importances = model.feature_importances_
    elif model_key == 'lightgbm':
        importances = model.feature_importances_
    elif model_key == 'random_forest':
        importances = model.feature_importances_
    
    indices = np.argsort(importances)[::-1][:15]
    top_features = [feature_cols[i] for i in indices]
    top_importances = importances[indices]
    
    axes[idx].barh(range(len(top_features)), top_importances)
    axes[idx].set_yticks(range(len(top_features)))
    axes[idx].set_yticklabels(top_features)
    axes[idx].set_xlabel('Importance')
    axes[idx].set_title(f'{model_name} - Top 15 Features')
    axes[idx].invert_yaxis()

plt.tight_layout()
plt.savefig(plots_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Feature importance plot saved")

# SVR Permutation Importance
print("\n[4] Calculating SVR permutation importance (this may take a while)...")
svr_model = models_dict['svr']
X_test_scaled = scaler.transform(X_test)

perm_importance = permutation_importance(svr_model, X_test_scaled, y_test, 
                                        n_repeats=10, random_state=42, n_jobs=-1)

indices = np.argsort(perm_importance.importances_mean)[::-1][:15]
top_features_svr = [feature_cols[i] for i in indices]
top_importances_svr = perm_importance.importances_mean[indices]

plt.figure(figsize=(10, 8))
plt.barh(range(len(top_features_svr)), top_importances_svr)
plt.yticks(range(len(top_features_svr)), top_features_svr)
plt.xlabel('Permutation Importance')
plt.title('SVR - Top 15 Features (Permutation Importance)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(plots_dir / "svr_permutation_importance.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ SVR permutation importance plot saved")

# ============================================================================
# MODEL PREDICTIONS COMPARISON
# ============================================================================
print("\n[5] Generating model predictions comparison...")

predictions_log = {}

try:
    predictions_log['XGBoost'] = models_dict['xgboost'].predict(X_test)
    print("  ✓ XGBoost predictions generated")
except Exception as e:
    print(f"  ⚠️  Warning: XGBoost prediction failed: {e}")
    predictions_log['XGBoost'] = np.zeros(len(X_test)) * np.nan

try:
    predictions_log['LightGBM'] = models_dict['lightgbm'].predict(X_test)
    print("  ✓ LightGBM predictions generated")
except Exception as e:
    print(f"  ⚠️  Warning: LightGBM prediction failed: {e}")
    predictions_log['LightGBM'] = np.zeros(len(X_test)) * np.nan

try:
    predictions_log['Random Forest'] = models_dict['random_forest'].predict(X_test)
    print("  ✓ Random Forest predictions generated")
except (AttributeError, Exception) as e:
    print(f"  ⚠️  Warning: Random Forest prediction failed (sklearn version compatibility issue): {e}")
    print("     Note: Models trained with sklearn 1.3.2 may not work with sklearn 1.7.2+")
    print("     Solution: Retrain models with current sklearn version by running Phase 4")
    predictions_log['Random Forest'] = np.zeros(len(X_test)) * np.nan

try:
    predictions_log['SVR'] = models_dict['svr'].predict(scaler.transform(X_test))
    print("  ✓ SVR predictions generated")
except Exception as e:
    print(f"  ⚠️  Warning: SVR prediction failed: {e}")
    predictions_log['SVR'] = np.zeros(len(X_test)) * np.nan

y_test_original = np.exp(y_test.values)

predictions_original = {}
for k, v in predictions_log.items():
    if isinstance(v, np.ndarray) and not np.isnan(v).all():
        predictions_original[k] = np.exp(v)
    else:
        predictions_original[k] = v

test_dates = pd.to_datetime(test_df['year_month'])
comparison_df = pd.DataFrame({
    'Date': test_dates,
    'Actual': y_test_original,
    **{k: v for k, v in predictions_original.items()}
})

plt.figure(figsize=(14, 8))
plt.plot(comparison_df['Date'], comparison_df['Actual'], 'ko-', label='Actual', linewidth=2, markersize=8)
plt.plot(comparison_df['Date'], comparison_df['XGBoost'], 'b--', label='XGBoost', linewidth=2)
plt.plot(comparison_df['Date'], comparison_df['LightGBM'], 'r--', label='LightGBM', linewidth=2)
plt.plot(comparison_df['Date'], comparison_df['Random Forest'], 'g--', label='Random Forest', linewidth=2)
plt.plot(comparison_df['Date'], comparison_df['SVR'], 'm--', label='SVR', linewidth=2)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Gold Price (USD per gram)', fontsize=12)
plt.title('Model Predictions Comparison - Test Period (2024-2025)', fontsize=16, pad=20)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(plots_dir / "model_predictions_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Model predictions comparison plot saved")

# ============================================================================
# DATASET SUMMARY
# ============================================================================
print("\n[6] Generating dataset summary...")

master_df = pd.read_csv(input_dir / "master_df.csv")
dataset_summary = {
    'Dataset': ['CPI', 'Fed Funds', 'USD-INR', 'USD-AUD', 'USD-CAD', 'USD-JPY', 
                'USD-CHF', 'USD-GBP', 'Gold', 'Treasury 10Y'],
    'Frequency': ['Monthly', 'Monthly', 'Monthly', 'Daily→Monthly', 'Daily→Monthly',
                  'Daily→Monthly', 'Daily→Monthly', 'Daily→Monthly', 'Daily→Monthly',
                  'Daily→Monthly'],
    'Date Range': [
        f"{master_df['cpi'].first_valid_index()} to {master_df['cpi'].last_valid_index()}",
        f"{master_df['fedfunds'].first_valid_index()} to {master_df['fedfunds'].last_valid_index()}",
        f"{master_df['usd_inr'].first_valid_index()} to {master_df['usd_inr'].last_valid_index()}",
        f"{master_df['usd_aud'].first_valid_index()} to {master_df['usd_aud'].last_valid_index()}",
        f"{master_df['usd_cad'].first_valid_index()} to {master_df['usd_cad'].last_valid_index()}",
        f"{master_df['usd_jpy'].first_valid_index()} to {master_df['usd_jpy'].last_valid_index()}",
        f"{master_df['usd_chf'].first_valid_index()} to {master_df['usd_chf'].last_valid_index()}",
        f"{master_df['usd_gbp'].first_valid_index()} to {master_df['usd_gbp'].last_valid_index()}",
        f"{master_df['gold_usd_per_gram'].first_valid_index()} to {master_df['gold_usd_per_gram'].last_valid_index()}",
        f"{master_df['treasury_10y'].first_valid_index()} to {master_df['treasury_10y'].last_valid_index()}"
    ]
}

dataset_summary_df = pd.DataFrame(dataset_summary)
dataset_summary_df.to_csv(input_dir / "dataset_summary.csv", index=False)
print("  ✓ Dataset summary saved")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)

