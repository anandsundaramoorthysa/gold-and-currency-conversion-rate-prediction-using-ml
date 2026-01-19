"""
PHASE 1: DATA PREPROCESSING
Execute exactly as described in requirements

Merged fixes from rebuild_gold_monthly.py, fix_master_dataset_issues.py:
- Gold price: Monthly MEAN aggregation (not first trading day) from daily data
- Date parsing: MM-DD-YYYY format for gold CSV (month-day-year)
- Gold price validation: Fallback to Open/High/Low when Price is suspicious
- Remove date column: Keep only year_month as temporal index
- Enhanced validation:
  * No date column in final dataset
  * Gold price varies month-to-month (no flat yearly blocks)
  * No extreme month-to-month drops (>50%)
  * Monthly volatility visible in key periods (2017-2019, 2020-2022, 2023-2025)
  * All gold prices >= $1/gram
  * Complete timeline 1975-01 to 2025-12 (612 months)
  * Exactly one row per year_month
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("PHASE 1: DATA PREPROCESSING")
print("=" * 80)

data_dir = Path("using_data")
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: LOAD & INSPECT ALL CSV FILES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: LOADING & INSPECTING DATASETS")
print("=" * 80)

datasets = {}

print("\n[1.1] Loading Monthly Files...")
datasets['cpi'] = pd.read_csv(data_dir / "Consumer Price Index for All Urban Consumers (1947 - 2025).csv")
datasets['fedfunds'] = pd.read_csv(data_dir / "FEDFUNDS (1955 - 2025).csv")
datasets['usd_inr'] = pd.read_csv(data_dir / "Foreign Exchange - India (1973 - 2025) Monthly.csv")

print("[1.2] Loading Daily Files...")
datasets['usd_aud'] = pd.read_csv(data_dir / "Foreign Exchange - Australian Dollars (1971 - 2025) Day.csv")
datasets['usd_cad'] = pd.read_csv(data_dir / "Foreign Exchange - Canadian Dollars (1971 - 2025) Day.csv")
datasets['usd_jpy'] = pd.read_csv(data_dir / "Foreign Exchange - Japanese Yen (1971 - 2025) Day.csv")
datasets['usd_chf'] = pd.read_csv(data_dir / "Foreign Exchange - Swiss Francs (1971 - 2025) Day.csv")
datasets['usd_gbp'] = pd.read_csv(data_dir / "Foreign Exchange - UK Pound (1971 - 2025) Day.csv")
datasets['treasury_10y'] = pd.read_csv(data_dir / "Market Yield on US 10 Yr Bonds (1962 - 2025) Day.csv")

print("\n[1.3] Dataset Inspection:")
print("-" * 80)
for name, df in datasets.items():
    print(f"\n{name.upper()}:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Missing values: {df.isnull().sum().sum()}")

datasets['treasury_10y'] = pd.read_csv(data_dir / "Market Yield on US 10 Yr Bonds (1962 - 2025) Day.csv")

# ============================================================================
# STEP 2: COLUMN SELECTION & RENAMING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: COLUMN SELECTION & RENAMING")
print("=" * 80)

datasets['cpi'] = datasets['cpi'][['observation_date', 'CPIAUCSL']].copy()
datasets['cpi'].rename(columns={'observation_date': 'date', 'CPIAUCSL': 'cpi'}, inplace=True)

datasets['fedfunds'] = datasets['fedfunds'][['observation_date', 'FEDFUNDS']].copy()
datasets['fedfunds'].rename(columns={'observation_date': 'date', 'FEDFUNDS': 'fedfunds'}, inplace=True)

if 'EXINUS' in datasets['usd_inr'].columns:
    usd_inr_col = 'EXINUS'
elif 'EXINIJS' in datasets['usd_inr'].columns:
    usd_inr_col = 'EXINIJS'
else:
    raise ValueError("Neither EXINUS nor EXINIJS found in USD-INR CSV!")

datasets['usd_inr'] = datasets['usd_inr'][['observation_date', usd_inr_col]].copy()
datasets['usd_inr'].rename(columns={'observation_date': 'date', usd_inr_col: 'usd_inr'}, inplace=True)

datasets['usd_aud'] = datasets['usd_aud'][['observation_date', 'DEXUSAL']].copy()
datasets['usd_aud'].rename(columns={'observation_date': 'date', 'DEXUSAL': 'usd_aud'}, inplace=True)

datasets['usd_cad'] = datasets['usd_cad'][['observation_date', 'DEXCAUS']].copy()
datasets['usd_cad'].rename(columns={'observation_date': 'date', 'DEXCAUS': 'usd_cad'}, inplace=True)

datasets['usd_jpy'] = datasets['usd_jpy'][['observation_date', 'DEXJPUS']].copy()
datasets['usd_jpy'].rename(columns={'observation_date': 'date', 'DEXJPUS': 'usd_jpy'}, inplace=True)

datasets['usd_chf'] = datasets['usd_chf'][['observation_date', 'DEXSZUS']].copy()
datasets['usd_chf'].rename(columns={'observation_date': 'date', 'DEXSZUS': 'usd_chf'}, inplace=True)

datasets['usd_gbp'] = datasets['usd_gbp'][['observation_date', 'DEXUSUK']].copy()
datasets['usd_gbp'].rename(columns={'observation_date': 'date', 'DEXUSUK': 'usd_gbp'}, inplace=True)

datasets['treasury_10y'] = datasets['treasury_10y'][['observation_date', 'DGS10']].copy()
datasets['treasury_10y'].rename(columns={'observation_date': 'date', 'DGS10': 'treasury_10y'}, inplace=True)

# ============================================================================
# STEP 2.1: GOLD PRICE PROCESSING (SPECIAL HANDLING)
# ============================================================================
print("\n[2.1] Processing Gold Price Data...")

gold_raw = pd.read_csv(data_dir / "Gold Futures Historical Data (1975 - 2025).csv")
print(f"  Gold CSV loaded: {len(gold_raw)} rows")

def clean_price(price_val):
    """Clean price value - remove commas, quotes, convert to float"""
    if pd.isna(price_val):
        return np.nan
    price_str = str(price_val).replace(',', '').replace('"', '').strip()
    try:
        return float(price_str)
    except (ValueError, TypeError):
        return np.nan

gold_raw['Price_clean'] = gold_raw['Price'].apply(clean_price)
gold_raw['Open_clean'] = gold_raw['Open'].apply(clean_price)
gold_raw['High_clean'] = gold_raw['High'].apply(clean_price)
gold_raw['Low_clean'] = gold_raw['Low'].apply(clean_price)

print("  Parsing dates as MM-DD-YYYY format...")
gold_raw['date'] = pd.to_datetime(gold_raw['Date'], format='%m-%d-%Y', errors='coerce')

if gold_raw['date'].isna().sum() > len(gold_raw) * 0.1:
    print("  WARNING: MM-DD-YYYY parsing failed, trying alternative...")
    date_parts = gold_raw['Date'].str.split('-', expand=True)
    gold_raw['extracted_month'] = pd.to_numeric(date_parts[0], errors='coerce')
    gold_raw['extracted_day'] = pd.to_numeric(date_parts[1], errors='coerce')
    gold_raw['extracted_year'] = pd.to_numeric(date_parts[2], errors='coerce')
    gold_raw['date'] = pd.to_datetime(
        gold_raw[['extracted_year', 'extracted_month', 'extracted_day']],
        errors='coerce'
    )

def get_valid_gold_price(row):
    """Get valid gold price in USD per ounce"""
    price = row['Price_clean']
    date_val = row['date']
    
    if pd.isna(date_val):
        return np.nan
    
    year = date_val.year
    
    fallback_values = []
    for val in [row['Open_clean'], row['High_clean'], row['Low_clean']]:
        if pd.notna(val) and 50 < val < 5000:
            fallback_values.append(val)
    
    fallback_avg = np.mean(fallback_values) if len(fallback_values) > 0 else np.nan
    
    if pd.isna(price):
        return fallback_avg
    
    if year >= 2000:
        if price < 200 or price > 5000:
            if pd.notna(fallback_avg) and 200 <= fallback_avg <= 5000:
                return fallback_avg
            return np.nan
    elif year < 2000:
        if price < 50 or price > 2000:
            if pd.notna(fallback_avg) and 50 <= fallback_avg <= 2000:
                return fallback_avg
            return np.nan
    
    return price

gold_raw['gold_usd_per_ounce'] = gold_raw.apply(get_valid_gold_price, axis=1)

gold_raw = gold_raw.dropna(subset=['date'])
gold_raw = gold_raw.sort_values('date', ascending=True).reset_index(drop=True)

gold_raw['year_month'] = gold_raw['date'].dt.to_period('M').astype(str)

print("  Computing monthly averages (MEAN of daily prices)...")
gold_monthly = gold_raw.groupby('year_month', as_index=False).agg({
    'gold_usd_per_ounce': 'mean'
}).copy()

if gold_monthly['gold_usd_per_ounce'].isna().sum() > 0:
    print(f"  WARNING: {gold_monthly['gold_usd_per_ounce'].isna().sum()} months with NaN, interpolating...")
    gold_monthly = gold_monthly.sort_values('year_month', ascending=True).reset_index(drop=True)
    gold_monthly['gold_usd_per_ounce'] = gold_monthly['gold_usd_per_ounce'].interpolate(method='linear', limit_direction='both')
    gold_monthly['gold_usd_per_ounce'] = gold_monthly['gold_usd_per_ounce'].bfill().ffill()

gold_monthly['gold_usd_per_gram'] = gold_monthly['gold_usd_per_ounce'] / 31.1035

invalid = gold_monthly[gold_monthly['gold_usd_per_gram'] < 1]
if len(invalid) > 0:
    print(f"  WARNING: {len(invalid)} invalid gold prices found, interpolating...")
    gold_monthly = gold_monthly.sort_values('year_month', ascending=True).reset_index(drop=True)
    gold_monthly['gold_usd_per_gram'] = gold_monthly['gold_usd_per_gram'].interpolate(method='linear', limit_direction='both')
    gold_monthly['gold_usd_per_gram'] = gold_monthly['gold_usd_per_gram'].bfill().ffill()
    
    invalid_after = gold_monthly[gold_monthly['gold_usd_per_gram'] < 1]
    if len(invalid_after) > 0:
        print(f"  ERROR: {len(invalid_after)} invalid prices remain after interpolation!")
        raise ValueError("Gold price validation failed after interpolation!")

print(f"  ✓ Gold monthly averages computed: {len(gold_monthly)} months")
print(f"  Price range: ${gold_monthly['gold_usd_per_gram'].min():.2f} to ${gold_monthly['gold_usd_per_gram'].max():.2f}/gram")

print(f"  Sample variation (first 6 months):")
for i, row in gold_monthly.head(6).iterrows():
    print(f"    {row['year_month']}: ${row['gold_usd_per_gram']:.4f}/gram")

datasets['gold'] = gold_monthly[['year_month', 'gold_usd_per_gram']].copy()

print("\n[2.2] Columns renamed and selected:")
for name, df in datasets.items():
    print(f"  {name}: {list(df.columns)}")

# ============================================================================
# STEP 3: DATE STANDARDIZATION (FOR NON-GOLD DATASETS)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: DATE STANDARDIZATION")
print("=" * 80)

def robust_parse_date(date_str):
    """Robust date parsing - handles multiple formats"""
    if pd.isna(date_str):
        return pd.NaT
    date_str = str(date_str).strip()
    # Try YYYY-MM-DD first (for most Fed datasets)
    try:
        parsed = pd.to_datetime(date_str, format='%Y-%m-%d', errors='coerce')
        if not pd.isna(parsed):
            return parsed
    except:
        pass
    # Try DD-MM-YYYY
    try:
        parsed = pd.to_datetime(date_str, format='%d-%m-%Y', errors='coerce')
        if not pd.isna(parsed):
            return parsed
    except:
        pass
    # Try auto-detection as fallback
    try:
        return pd.to_datetime(date_str, errors='coerce')
    except:
        return pd.NaT

for name, df in datasets.items():
    if name != 'gold':
        datasets[name]['date'] = datasets[name]['date'].apply(robust_parse_date)
        initial_count = len(datasets[name])
        datasets[name] = datasets[name].dropna(subset=['date'])
        dropped_count = initial_count - len(datasets[name])
        if dropped_count > 0:
            print(f"  {name}: Dropped {dropped_count} rows with invalid dates")

for name, df in datasets.items():
    if name != 'gold':
        datasets[name] = datasets[name].drop_duplicates(subset=['date'], keep='first')
        datasets[name] = datasets[name].drop_duplicates()

for name, df in datasets.items():
    if name != 'gold':
        datasets[name] = datasets[name].sort_values('date', ascending=True).reset_index(drop=True)
        datasets[name]['year_month'] = datasets[name]['date'].dt.to_period('M').astype(str)

print("\n[3.1] Date standardization complete:")
for name, df in datasets.items():
    if name != 'gold':
        print(f"  {name}: {df['date'].min()} to {df['date'].max()}, {len(df)} rows")
    else:
        print(f"  {name}: {len(df)} months (already monthly)")

# ============================================================================
# STEP 4: HANDLE DIFFERENT FREQUENCIES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: HANDLE DIFFERENT FREQUENCIES")
print("=" * 80)

monthly_files = ['cpi', 'fedfunds', 'usd_inr']
for name in monthly_files:
    datasets[name] = datasets[name].groupby('year_month', as_index=False).agg({
        'date': 'first',
        name: 'mean'
    })

column_mapping = {
    'usd_aud': 'usd_aud',
    'usd_cad': 'usd_cad',
    'usd_jpy': 'usd_jpy',
    'usd_chf': 'usd_chf',
    'usd_gbp': 'usd_gbp',
    'treasury_10y': 'treasury_10y'
}
daily_files = ['usd_aud', 'usd_cad', 'usd_jpy', 'usd_chf', 'usd_gbp', 'treasury_10y']

print("\n[4.1] Converting daily files to monthly (first available trading day):")
for name in daily_files:
    df = datasets[name].copy()
    col_name = column_mapping[name]
    
    monthly_data = []
    for year_month, group in df.groupby('year_month'):
        group_sorted = group.sort_values('date', ascending=True)
        if col_name in group_sorted.columns:
            first_valid = group_sorted[group_sorted[col_name].notna()].iloc[0] if group_sorted[col_name].notna().any() else None
            if first_valid is not None:
                monthly_data.append({
                    'year_month': year_month,
                    col_name: first_valid[col_name]
                })
    
    datasets[name] = pd.DataFrame(monthly_data)
    if len(datasets[name]) > 0:
        datasets[name] = datasets[name].sort_values('year_month', ascending=True).reset_index(drop=True)
    
    print(f"  {name}: {len(datasets[name])} monthly records")

# Gold is already monthly, so no conversion needed
print("  gold: Already monthly (using MEAN aggregation)")

# ============================================================================
# STEP 5: MISSING VALUE HANDLING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: MISSING VALUE HANDLING")
print("=" * 80)

# Forward fill and backward fill for each dataset (except gold - already processed)
for name, df in datasets.items():
    if name != 'gold':
        for col in df.columns:
            if col not in ['year_month', 'date']:
                datasets[name][col] = datasets[name][col].ffill().bfill()

print("[5.1] Forward and backward fill completed for all datasets")

# ============================================================================
# STEP 6: MERGE & FINAL MASTER DATASET
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: MERGE & FINAL MASTER DATASET")
print("=" * 80)

master_df = datasets['gold'][['year_month', 'gold_usd_per_gram']].copy()

merge_order = ['cpi', 'fedfunds', 'usd_inr', 'usd_aud', 'usd_cad', 'usd_jpy', 'usd_chf', 'usd_gbp', 'treasury_10y']

for name in merge_order:
    if name in datasets and len(datasets[name]) > 0:
        df_to_merge = datasets[name][['year_month', name]].copy()
        master_df = master_df.merge(df_to_merge, on='year_month', how='left')
        print(f"  Merged {name}")

master_df = master_df.sort_values('year_month', ascending=True).reset_index(drop=True)
master_df = master_df.drop_duplicates(subset=['year_month'], keep='first')

master_df = master_df[
    (master_df['year_month'] >= '1975-01') & 
    (master_df['year_month'] <= '2025-12')
].copy()

all_months = pd.period_range(start='1975-01', end='2025-12', freq='M').astype(str)
missing_months = sorted(set(all_months) - set(master_df['year_month'].unique()))
if missing_months:
    print(f"  WARNING: {len(missing_months)} missing months detected: {missing_months[:5]}...")
    missing_rows = []
    for ym in missing_months:
        row_dict = {'year_month': ym}
        for col in master_df.columns:
            if col != 'year_month':
                row_dict[col] = np.nan
        missing_rows.append(row_dict)
    missing_df = pd.DataFrame(missing_rows)
    master_df = pd.concat([master_df, missing_df], ignore_index=True)
    master_df = master_df.sort_values('year_month', ascending=True).reset_index(drop=True)

missing_before = master_df['gold_usd_per_gram'].isna().sum()
if missing_before > 0:
    print(f"  WARNING: {missing_before} rows with missing gold prices, filling...")
    master_df['gold_usd_per_gram'] = master_df['gold_usd_per_gram'].bfill().ffill()

for col in master_df.columns:
    if col != 'year_month':
        master_df[col] = master_df[col].ffill().bfill()

threshold = len(master_df.columns) * 0.5
rows_before = len(master_df)
master_df = master_df.dropna(thresh=threshold)
rows_dropped = rows_before - len(master_df)
if rows_dropped > 0:
    print(f"  Dropped {rows_dropped} rows with >50% missing values")

for col in master_df.columns:
    if col != 'year_month':
        if master_df[col].isna().sum() > 0:
            master_df[col] = master_df[col].ffill().bfill()

if 'date' in master_df.columns:
    master_df = master_df.drop(columns=['date'])
    print("  ✓ Date column removed (keeping only year_month as temporal index)")
else:
    print("  ✓ No date column found (already removed or never existed)")

# ============================================================================
# STEP 7: FINAL VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: FINAL VALIDATION")
print("=" * 80)

print(f"\n[7.1] Master dataset created:")
print(f"  Shape: {master_df.shape}")
print(f"  Columns: {list(master_df.columns)}")
print(f"  Date range: {master_df['year_month'].min()} to {master_df['year_month'].max()}")

if 'date' in master_df.columns:
    raise ValueError("FAILED: date column still exists!")
else:
    print("  PASSED: date column removed")

duplicates = master_df.duplicated(subset=['year_month'], keep=False).sum()
if duplicates > 0:
    raise ValueError(f"FAILED: {duplicates} duplicate year_month entries found!")
else:
    print("  PASSED: One row per year_month")

monthly_changes = master_df['gold_usd_per_gram'].diff().abs()
flat_months = (monthly_changes < 0.001).sum()
print(f"  Flat months (< $0.001 change): {flat_months} out of {len(master_df)}")

yearly_variation = master_df.groupby(master_df['year_month'].str[:4])['gold_usd_per_gram'].agg(['nunique'])
yearly_blocks = yearly_variation[yearly_variation['nunique'] == 1]
if len(yearly_blocks) > 5:
    raise ValueError(f"FAILED: {len(yearly_blocks)} years with flat gold prices!")
else:
    print(f"  PASSED: Gold price varies month-to-month (only {len(yearly_blocks)} flat years)")

monthly_pct_change = master_df['gold_usd_per_gram'].pct_change() * 100
extreme_drops = (monthly_pct_change < -50).sum()
if extreme_drops > 0:
    raise ValueError(f"FAILED: {extreme_drops} months with >50% drop!")
else:
    max_drop = monthly_pct_change.min()
    print(f"  PASSED: No extreme drops (max drop: {max_drop:.2f}%)")

periods_to_check = {
    '2017-2019': ('2017-01', '2019-12'),
    '2020-2022': ('2020-01', '2022-12'),
    '2023-2025': ('2023-01', '2025-12')
}
all_periods_pass = True
for period_name, (start, end) in periods_to_check.items():
    period_data = master_df[(master_df['year_month'] >= start) & (master_df['year_month'] <= end)].copy()
    if len(period_data) > 0:
        unique_values = period_data['gold_usd_per_gram'].nunique()
        if unique_values < len(period_data) * 0.8:
            print(f"  WARNING: Low variation in {period_name}")
            all_periods_pass = False
        else:
            print(f"  PASSED: Good variation in {period_name} ({unique_values} unique values)")

invalid_gold = master_df[master_df['gold_usd_per_gram'] < 1]
if len(invalid_gold) > 0:
    raise ValueError(f"FAILED: {len(invalid_gold)} rows with gold < $1/gram!")
else:
    print("  PASSED: All gold prices >= $1/gram")

expected_months = (2025 - 1975 + 1) * 12
actual_months = len(master_df)
if actual_months != expected_months:
    all_months = pd.period_range(start='1975-01', end='2025-12', freq='M').astype(str)
    missing = sorted(set(all_months) - set(master_df['year_month'].unique()))
    if missing:
        print(f"  WARNING: {len(missing)} missing months: {missing[:10]}...")
    print(f"  Expected: {expected_months}, Actual: {actual_months}")
else:
    print(f"  PASSED: Complete timeline ({actual_months} months)")

if master_df['year_month'].min() <= '1975-01' and master_df['year_month'].max() >= '2025-12':
    all_months_check = pd.period_range(start='1975-01', end='2025-12', freq='M').astype(str)
    if len(set(all_months_check) - set(master_df['year_month'].unique())) == 0:
        print("  PASSED: Timeline 1975-01 -> 2025-12 (complete)")
    else:
        missing_check = sorted(set(all_months_check) - set(master_df['year_month'].unique()))
        print(f"  WARNING: Missing months: {missing_check}")
else:
    raise ValueError(f"FAILED: Timeline is {master_df['year_month'].min()} -> {master_df['year_month'].max()}")

is_sorted = master_df['year_month'].is_monotonic_increasing
if not is_sorted:
    master_df = master_df.sort_values('year_month', ascending=True).reset_index(drop=True)
    print("  OK: Fixed: Re-sorted by year_month")

column_order = ['year_month'] + [col for col in master_df.columns if col != 'year_month']
master_df = master_df[column_order]

# Save master dataset
master_df.to_csv(output_dir / "master_df.csv", index=False)
print(f"\n[7.2] Master dataset saved to: {output_dir / 'master_df.csv'}")

print("\n" + "=" * 80)
print("PHASE 1 COMPLETE!")
print("=" * 80)
