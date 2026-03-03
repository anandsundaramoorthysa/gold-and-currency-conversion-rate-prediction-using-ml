"""
PHASE 2: FEATURE ENGINEERING (RETURN-BASED - FIXES DISTRIBUTION SHIFT)
- Target: gold_return_next (monthly log return) instead of absolute log price
- All gold features are stationary (returns/ratios) instead of raw price levels
- Tree models can now generalise beyond training price range
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
print("\n[4] Creating stationary gold features (FIX: returns/ratios replace raw lags)...")

# gold_return_1m already created above
features_df['gold_return_3m'] = features_df['log_gold'] - features_df['log_gold'].shift(3)
print("  ✓ gold_return_3m: 3-month cumulative log return (replaces gold_lag3)")

features_df['gold_return_6m'] = features_df['log_gold'] - features_df['log_gold'].shift(6)
print("  ✓ gold_return_6m: 6-month cumulative log return (medium-term momentum)")

features_df['gold_return_12m'] = features_df['log_gold'] - features_df['log_gold'].shift(12)
print("  ✓ gold_return_12m: 12-month cumulative log return (annual momentum)")

features_df['gold_volatility_3m'] = features_df['gold_return_1m'].rolling(window=3, min_periods=2).std()
print("  ✓ gold_volatility_3m: 3-month rolling std of log returns")

features_df['fedfunds_lag1'] = features_df['fedfunds'].shift(1)
features_df['cpi_yoy_lag1'] = features_df['cpi_yoy'].shift(1)
features_df['real_rate_lag1'] = features_df['real_rate'].shift(1)
print("  ✓ Macro lags (1 month each)")

# ============================================================================
# TECHNICAL FEATURES (STATIONARY RATIO - NOT RAW PRICE)
# ============================================================================
print("\n[5] Creating stationary momentum feature...")

gold_ma3_raw = features_df['gold_usd_per_gram'].rolling(window=3, min_periods=1).mean()
features_df['gold_ma3_ratio'] = gold_ma3_raw / features_df['gold_usd_per_gram']
print("  ✓ gold_ma3_ratio = MA3 / current_price (stationary, replaces raw gold_ma3)")

print("\n[6] Creating target variable (LOG RETURN - FIX 1)...")

features_df['gold_return_next'] = features_df['log_gold'].shift(-1) - features_df['log_gold']
print("  ✓ Target: gold_return_next = log(gold_t+1) - log(gold_t)  [monthly log return]")
print("  ✓ Removed: log_gold_next (absolute price - caused distribution-shift failure)")
print("  ✓ Back-transform at inference: predicted_price = current_gold * exp(predicted_return)")

print("\n[7] Selecting stationary feature set (FIX 2)...")

essential_features = [
    'year_month',
    # Stationary gold features (returns/ratios - NOT raw price levels)
    'gold_return_1m',       # 1-month log return
    'gold_return_3m',       # 3-month cumulative log return
    'gold_return_6m',       # 6-month cumulative log return (medium-term momentum)
    'gold_return_12m',      # 12-month cumulative log return (annual momentum)
    'gold_ma3_ratio',       # MA3/price ratio (momentum, always near 1.0)
    'gold_volatility_3m',   # 3-month return volatility
    # Macro features (already stationary)
    'real_rate',
    'real_rate_lag1',
    'fedfunds_lag1',
    'cpi_yoy_lag1',
    'yield_spread',
    # Target: log return (not absolute price)
    'gold_return_next'
]

features_df = features_df[essential_features].copy()
print(f"  ✓ {len(essential_features)} features selected (11 predictors + target + year_month)")
print(f"  Removed: gold_lag1, gold_lag3, gold_ma3 (non-stationary raw price levels)")
print(f"  Added:   gold_return_3m/6m/12m, gold_ma3_ratio, gold_volatility_3m (stationary)")

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
print(f"\n⚠️  IMPORTANT: Target is 'gold_return_next' (monthly log return, NOT absolute price)")
print(f"   Back-transform: predicted_price = current_gold_price * exp(predicted_return)")

# Save features dataset
output_path = output_dir / "features_df.csv"
features_df.to_csv(output_path, index=False)
print(f"\n[9] Features dataset saved to: {output_path}")

print("\n" + "=" * 80)
print("PHASE 2 COMPLETE!")
print("=" * 80)

