"""
PHASE 3: TRAIN / VALIDATION / TEST SPLIT (REGIME-SAFE)
FIXED: Better regime separation to prevent overfitting
Train: 1977-01 → 2015-12 (stable period)
Validation: 2016-01 → 2019-12 (pre-COVID, different regime)
Test: 2020-01 → 2025-12 (COVID and post-COVID period - true out-of-sample)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle

print("=" * 80)
print("PHASE 3: TRAIN / VALIDATION / TEST SPLIT")
print("=" * 80)

input_dir = Path("outputs")
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

print("\n[1] Loading features dataset...")
features_df = pd.read_csv(input_dir / "features_df.csv")
print(f"  Shape: {features_df.shape}")
print(f"  Date range: {features_df['year_month'].min()} to {features_df['year_month'].max()}")

# ============================================================================
# SPLIT BY TIME PERIOD
# ============================================================================
print("\n[2] Creating time series splits...")

train_start = '1977-01'
train_end = '2015-12'
val_start = '2016-01'
val_end = '2019-12'
test_start = '2020-01'
test_end = '2025-12'

train_df = features_df[(features_df['year_month'] >= train_start) & 
                       (features_df['year_month'] <= train_end)].copy()
val_df = features_df[(features_df['year_month'] >= val_start) & 
                     (features_df['year_month'] <= val_end)].copy()
test_df = features_df[(features_df['year_month'] >= test_start) & 
                      (features_df['year_month'] <= test_end)].copy()

print(f"\n  Train:   {len(train_df)} rows ({train_df['year_month'].min()} to {train_df['year_month'].max()})")
print(f"  Validation: {len(val_df)} rows ({val_df['year_month'].min()} to {val_df['year_month'].max()})")
print(f"  Test:    {len(test_df)} rows ({test_df['year_month'].min()} to {test_df['year_month'].max()})")

# ============================================================================
# SEPARATE FEATURES AND TARGET
# ============================================================================
print("\n[3] Separating features and target...")

exclude_cols = ['year_month', 'date', 'log_gold_next', 'gold_usd_per_gram_next_month']
exclude_cols = [col for col in exclude_cols if col in features_df.columns]

feature_cols = [col for col in features_df.columns if col not in exclude_cols]

X_train = train_df[feature_cols].copy()
y_train = train_df['log_gold_next'].copy()

X_val = val_df[feature_cols].copy()
y_val = val_df['log_gold_next'].copy()

X_test = test_df[feature_cols].copy()
y_test = test_df['log_gold_next'].copy()

print(f"  Features: {len(feature_cols)}")
print(f"  Feature names: {feature_cols}")
print(f"\n  ⚠️  TARGET: log_gold_next (log scale)")
print(f"     Predictions will be back-transformed: exp(log_pred)")

print("\n[4] Saving splits...")

train_df.to_csv(output_dir / "train_df.csv", index=False)
val_df.to_csv(output_dir / "val_df.csv", index=False)
test_df.to_csv(output_dir / "test_df.csv", index=False)

X_train.to_csv(output_dir / "X_train.csv", index=False)
y_train.to_csv(output_dir / "y_train.csv", index=False)
X_val.to_csv(output_dir / "X_val.csv", index=False)
y_val.to_csv(output_dir / "y_val.csv", index=False)
X_test.to_csv(output_dir / "X_test.csv", index=False)
y_test.to_csv(output_dir / "y_test.csv", index=False)

with open(output_dir / "feature_columns.pkl", 'wb') as f:
    pickle.dump(feature_cols, f)

print("  ✓ All splits saved to outputs/")

print("\n" + "=" * 80)
print("PHASE 3 COMPLETE!")
print("=" * 80)

