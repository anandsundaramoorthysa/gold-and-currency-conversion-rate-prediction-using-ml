"""
PHASE 2: FEATURE ENGINEERING (FIXED FOR TIME-SERIES STABILITY)
- Target: Log scale (log_gold_next) instead of raw prices
- Removed noisy features to prevent overfitting
- Added log returns for better stationarity
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("PHASE 2: FEATURE ENGINEERING")
print("=" * 80)

input_dir = Path("outputs")
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

print("\n[1] Loading master dataset...")
master_df = pd.read_csv(input_dir / "master_df.csv")
print(f"  Shape: {master_df.shape}")
print(f"  Date range: {master_df['year_month'].min()} to {master_df['year_month'].max()}")

# ============================================================================
# BASE FEATURES (all columns from master_df)
# ============================================================================
features_df = master_df.copy()

# ============================================================================
# MACRO DERIVED FEATURES
# ============================================================================
print("\n[2] Creating macro derived features...")

features_df['cpi_yoy'] = (features_df['cpi'] / features_df['cpi'].shift(12) - 1) * 100
print("  ✓ CPI YoY Inflation calculated")

features_df['real_rate'] = features_df['fedfunds'] - features_df['cpi_yoy']
print("  ✓ Real Interest Rate calculated")

features_df['yield_spread'] = features_df['treasury_10y'] - features_df['fedfunds']
print("  ✓ Yield Spread calculated")

# ============================================================================
# LOG TRANSFORMATION (CRITICAL FOR TIME-SERIES STABILITY)
# ============================================================================
print("\n[3] Creating log-scale features...")

features_df['log_gold'] = np.log(features_df['gold_usd_per_gram'])
print("  ✓ Log gold price calculated (log_gold)")

features_df['gold_return_1m'] = features_df['log_gold'] - features_df['log_gold'].shift(1)
print("  ✓ Gold log returns (1-month) calculated")

# ============================================================================
# LAG FEATURES (REDUCED SET - PREVENT OVERFITTING)
# ============================================================================
print("\n[4] Creating lag features (reduced set)...")

features_df['gold_lag1'] = features_df['gold_usd_per_gram'].shift(1)
features_df['gold_lag3'] = features_df['gold_usd_per_gram'].shift(3)
print("  ✓ Gold lags (1, 3 months) - lag2 removed")

features_df['fedfunds_lag1'] = features_df['fedfunds'].shift(1)
features_df['cpi_yoy_lag1'] = features_df['cpi_yoy'].shift(1)
features_df['real_rate_lag1'] = features_df['real_rate'].shift(1)
print("  ✓ Macro lags (1 month each) - removed lag2 variants")

# ============================================================================
# TECHNICAL FEATURES (MINIMAL - PREVENT OVERFITTING)
# ============================================================================
print("\n[5] Creating technical features (minimal set)...")

features_df['gold_ma3'] = features_df['gold_usd_per_gram'].rolling(window=3, min_periods=1).mean()
print("  ✓ Gold MA3 (removed MA12)")

print("\n[6] Creating target variable (LOG SCALE)...")

features_df['log_gold_next'] = features_df['log_gold'].shift(-1)
print("  ✓ Target variable created: log_gold_next (log scale)")
print("  ✓ Removed: gold_usd_per_gram_next_month (raw price target - causes overfitting)")

print("\n[7] Selecting essential features (reducing overfitting risk)...")

essential_features = [
    'year_month',
    'gold_lag1',
    'gold_lag3',
    'gold_return_1m',
    'gold_ma3',
    'real_rate',
    'real_rate_lag1',
    'fedfunds_lag1',
    'cpi_yoy_lag1',
    'yield_spread',
    'log_gold_next'
]

features_df = features_df[essential_features].copy()
print(f"  ✓ Reduced to {len(essential_features)} essential features")
print(f"  Removed: gold_lag2, fedfunds_lag2, cpi_yoy_lag2, usd_inr_lags, gold_ma12, volatility features")

print("\n[8] Final cleanup...")

initial_rows = len(features_df)
features_df = features_df.dropna()
final_rows = len(features_df)
print(f"  Dropped {initial_rows - final_rows} rows with NaN values")
print(f"  Final shape: {features_df.shape}")

features_df = features_df.sort_values('year_month', ascending=True).reset_index(drop=True)

# Display summary
print("\n" + "=" * 80)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 80)
print(f"Total features: {len(features_df.columns)}")
print(f"Total rows: {len(features_df)}")
print(f"\nFeature list:")
for i, col in enumerate(features_df.columns, 1):
    is_target = " (TARGET)" if col == 'log_gold_next' else ""
    print(f"  {i:2d}. {col}{is_target}")

print(f"\nDate range: {features_df['year_month'].min()} to {features_df['year_month'].max()}")
print(f"Missing values: {features_df.isnull().sum().sum()}")
print(f"\n⚠️  IMPORTANT: Target is 'log_gold_next' (log scale), not raw price!")

# Save features dataset
output_path = output_dir / "features_df.csv"
features_df.to_csv(output_path, index=False)
print(f"\n[9] Features dataset saved to: {output_path}")

print("\n" + "=" * 80)
print("PHASE 2 COMPLETE!")
print("=" * 80)

