# PHASE 1: DATA PREPROCESSING - DETAILED EXPLANATION

## Overview
Phase 1 is responsible for loading, cleaning, standardizing, and merging all raw CSV datasets into a single master dataset ready for feature engineering. This phase handles 10 different data sources with varying formats, frequencies (daily vs monthly), and date formats.

---

## STEP-BY-STEP BREAKDOWN

### STEP 1: LOAD & INSPECT ALL CSV FILES

#### Lines 28-30: Directory Setup
```python
data_dir = Path("using_data")
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)
```
**Explanation:**
- Creates Path objects for input and output directories
- `mkdir(exist_ok=True)` creates the outputs directory if it doesn't exist (won't error if it already exists)
- All processed data will be saved to the `outputs/` folder

#### Lines 39-41: Initialize Dataset Dictionary
```python
datasets = {}
```
**Explanation:**
- Creates an empty dictionary to store all loaded datasets
- Each dataset will be stored with a key name (e.g., 'cpi', 'gold', 'fedfunds')

#### Lines 41-42: Load Monthly Files
```python
datasets['cpi'] = pd.read_csv(data_dir / "Consumer Price Index for All Urban Consumers (1947 - 2025).csv")
datasets['fedfunds'] = pd.read_csv(data_dir / "FEDFUNDS (1955 - 2025).csv")
datasets['usd_inr'] = pd.read_csv(data_dir / "Foreign Exchange - India (1973 - 2025) Monthly.csv")
```
**Explanation:**
- Loads three datasets that are already in monthly frequency
- CPI (Consumer Price Index): Measures inflation
- FEDFUNDS: Federal Funds Rate (interest rate set by the Fed)
- USD-INR: Exchange rate for US Dollar to Indian Rupee
- These files don't need frequency conversion (already monthly)

#### Lines 46-52: Load Daily Files
```python
datasets['usd_aud'] = pd.read_csv(data_dir / "Foreign Exchange - Australian Dollars (1971 - 2025) Day.csv")
datasets['usd_cad'] = pd.read_csv(data_dir / "Foreign Exchange - Canadian Dollars (1971 - 2025) Day.csv")
datasets['usd_jpy'] = pd.read_csv(data_dir / "Foreign Exchange - Japanese Yen (1971 - 2025) Day.csv")
datasets['usd_chf'] = pd.read_csv(data_dir / "Foreign Exchange - Swiss Francs (1971 - 2025) Day.csv")
datasets['usd_gbp'] = pd.read_csv(data_dir / "Foreign Exchange - UK Pound (1971 - 2025) Day.csv")
datasets['treasury_10y'] = pd.read_csv(data_dir / "Market Yield on US 10 Yr Bonds (1962 - 2025) Day.csv")
```
**Explanation:**
- Loads 6 datasets that are in daily frequency
- These will need to be converted to monthly later
- All are exchange rates except treasury_10y (bond yield)
- Daily data has many more rows and needs aggregation

#### Lines 54-61: Dataset Inspection Loop
```python
for name, df in datasets.items():
    print(f"\n{name.upper()}:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
```
**Explanation:**
- Iterates through all loaded datasets
- Prints shape (rows, columns), column names, and total missing values
- This helps verify data loaded correctly and identify data quality issues

---

### STEP 2: COLUMN SELECTION & RENAMING

#### Lines 71-72: CPI Processing
```python
datasets['cpi'] = datasets['cpi'][['observation_date', 'CPIAUCSL']].copy()
datasets['cpi'].rename(columns={'observation_date': 'date', 'CPIAUCSL': 'cpi'}, inplace=True)
```
**Explanation:**
- Selects only the date and value columns (drops unnecessary columns)
- Renames columns to standardized names: 'date' and 'cpi'
- `copy()` creates a new DataFrame to avoid warnings
- `inplace=True` modifies the DataFrame directly

#### Lines 74-75: Fed Funds Processing
```python
datasets['fedfunds'] = datasets['fedfunds'][['observation_date', 'FEDFUNDS']].copy()
datasets['fedfunds'].rename(columns={'observation_date': 'date', 'FEDFUNDS': 'fedfunds'}, inplace=True)
```
**Explanation:**
- Similar process for Federal Funds Rate
- Standardizes column names to 'date' and 'fedfunds'

#### Lines 77-85: USD-INR Processing (Special Handling)
```python
if 'EXINUS' in datasets['usd_inr'].columns:
    usd_inr_col = 'EXINUS'
elif 'EXINIJS' in datasets['usd_inr'].columns:
    usd_inr_col = 'EXINIJS'
else:
    raise ValueError("Neither EXINUS nor EXINIJS found in USD-INR CSV!")
```
**Explanation:**
- Checks which column name exists (different CSV versions may use different names)
- Handles potential variations in column naming
- Raises error if expected columns not found (data quality check)

#### Lines 87-103: Daily File Column Standardization
```python
datasets['usd_aud'] = datasets['usd_aud'][['observation_date', 'DEXUSAL']].copy()
datasets['usd_aud'].rename(columns={'observation_date': 'date', 'DEXUSAL': 'usd_aud'}, inplace=True)
# ... (similar for other daily files)
```
**Explanation:**
- Each exchange rate file uses different column codes (DEXUSAL, DEXCAUS, etc.)
- Standardizes all to 'date' and descriptive name (usd_aud, usd_cad, etc.)
- This makes merging easier later

---

### STEP 2.1: GOLD PRICE PROCESSING (SPECIAL HANDLING)

Gold requires special processing because:
1. Date format is MM-DD-YYYY (unusual)
2. Price data may have formatting issues (commas, quotes)
3. Needs conversion from USD/ounce to USD/gram
4. Uses MEAN aggregation (not first value) for monthly data

#### Lines 110-111: Load Gold Data
```python
gold_raw = pd.read_csv(data_dir / "Gold Futures Historical Data (1975 - 2025).csv")
print(f"  Gold CSV loaded: {len(gold_raw)} rows")
```
**Explanation:**
- Loads gold futures data (daily frequency)
- This is the target variable we're trying to predict

#### Lines 113-121: Price Cleaning Function
```python
def clean_price(price_val):
    """Clean price value - remove commas, quotes, convert to float"""
    if pd.isna(price_val):
        return np.nan
    price_str = str(price_val).replace(',', '').replace('"', '').strip()
    try:
        return float(price_str)
    except (ValueError, TypeError):
        return np.nan
```
**Explanation:**
- Removes formatting characters (commas, quotes) that prevent conversion to float
- Handles missing values gracefully
- Uses try-except to handle invalid values
- Returns NaN for unparseable values (will be handled later)

#### Lines 123-126: Clean Price Columns
```python
gold_raw['Price_clean'] = gold_raw['Price'].apply(clean_price)
gold_raw['Open_clean'] = gold_raw['Open'].apply(clean_price)
gold_raw['High_clean'] = gold_raw['High'].apply(clean_price)
gold_raw['Low_clean'] = gold_raw['Low'].apply(clean_price)
```
**Explanation:**
- Creates cleaned versions of Price, Open, High, Low columns
- These will be used as fallbacks if Price is invalid
- Open/High/Low provide alternative price sources

#### Lines 128-140: Date Parsing (MM-DD-YYYY Format)
```python
gold_raw['date'] = pd.to_datetime(gold_raw['Date'], format='%m-%d-%Y', errors='coerce')

if gold_raw['date'].isna().sum() > len(gold_raw) * 0.1:
    # Fallback parsing logic
```
**Explanation:**
- Parses dates in MM-DD-YYYY format (month-day-year)
- `errors='coerce'` converts invalid dates to NaT (Not a Time)
- Checks if >10% failed to parse, then tries alternative method
- Fallback splits date string and reconstructs

#### Lines 142-173: Get Valid Gold Price Function
```python
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
```
**Explanation:**
- Validates gold price using year-specific ranges:
  - Pre-2000: $50-$2000 per ounce (historical lower prices)
  - Post-2000: $200-$5000 per ounce (modern price range)
- Uses Open/High/Low as fallbacks if Price is invalid
- Returns NaN only if all sources are invalid

#### Lines 175-178: Apply Price Validation
```python
gold_raw['gold_usd_per_ounce'] = gold_raw.apply(get_valid_gold_price, axis=1)
gold_raw = gold_raw.dropna(subset=['date'])
gold_raw = gold_raw.sort_values('date', ascending=True).reset_index(drop=True)
```
**Explanation:**
- Applies validation function to each row
- Drops rows with invalid dates
- Sorts by date (critical for time-series analysis)
- `reset_index(drop=True)` creates clean index 0,1,2,...

#### Lines 180-185: Create Year-Month Key and Monthly Aggregation
```python
gold_raw['year_month'] = gold_raw['date'].dt.to_period('M').astype(str)
gold_monthly = gold_raw.groupby('year_month', as_index=False).agg({
    'gold_usd_per_ounce': 'mean'
}).copy()
```
**Explanation:**
- `dt.to_period('M')` converts dates to monthly periods (e.g., '2025-01')
- Groups by year_month and calculates MEAN (not first value)
- MEAN is used because it's more stable than first trading day
- Results in one row per month

#### Lines 187-191: Handle Missing Months
```python
if gold_monthly['gold_usd_per_ounce'].isna().sum() > 0:
    gold_monthly = gold_monthly.sort_values('year_month', ascending=True).reset_index(drop=True)
    gold_monthly['gold_usd_per_ounce'] = gold_monthly['gold_usd_per_ounce'].interpolate(method='linear', limit_direction='both')
    gold_monthly['gold_usd_per_ounce'] = gold_monthly['gold_usd_per_ounce'].bfill().ffill()
```
**Explanation:**
- Checks for missing values after aggregation
- Interpolates missing values linearly (fills gaps between known values)
- `limit_direction='both'` fills from both sides
- `bfill().ffill()` handles edge cases (fills beginning/end if interpolation fails)

#### Line 193: Convert to USD per Gram
```python
gold_monthly['gold_usd_per_gram'] = gold_monthly['gold_usd_per_ounce'] / 31.1035
```
**Explanation:**
- Converts from USD/ounce to USD/gram
- Conversion factor: 1 ounce = 31.1035 grams (troy ounce)
- This is the target variable format

#### Lines 195-205: Validate Gold Prices
```python
invalid = gold_monthly[gold_monthly['gold_usd_per_gram'] < 1]
if len(invalid) > 0:
    # Interpolation and validation
```
**Explanation:**
- Ensures all prices are >= $1/gram (reasonable minimum)
- If invalid prices found, interpolates again
- Raises error if invalid prices remain (data quality check)

#### Line 214: Store Gold Dataset
```python
datasets['gold'] = gold_monthly[['year_month', 'gold_usd_per_gram']].copy()
```
**Explanation:**
- Stores only year_month and price columns
- Removes intermediate columns (date, ounces, etc.)
- Gold dataset is now ready for merging

---

### STEP 3: DATE STANDARDIZATION (FOR NON-GOLD DATASETS)

#### Lines 227-250: Robust Date Parsing Function
```python
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
```
**Explanation:**
- Handles multiple date formats (YYYY-MM-DD, DD-MM-YYYY)
- Tries formats in order, falls back to auto-detection
- Returns NaT (Not a Time) for unparseable dates
- Prevents crashes from format variations

#### Lines 252-259: Apply Date Parsing
```python
for name, df in datasets.items():
    if name != 'gold':  # Gold already processed
        datasets[name]['date'] = datasets[name]['date'].apply(robust_parse_date)
        initial_count = len(datasets[name])
        datasets[name] = datasets[name].dropna(subset=['date'])
        dropped_count = initial_count - len(datasets[name])
        if dropped_count > 0:
            print(f"  {name}: Dropped {dropped_count} rows with invalid dates")
```
**Explanation:**
- Applies date parsing to all datasets except gold (already processed)
- Drops rows with invalid dates (NaT values)
- Reports how many rows were dropped (transparency)

#### Lines 261-264: Remove Duplicates
```python
for name, df in datasets.items():
    if name != 'gold':
        datasets[name] = datasets[name].drop_duplicates(subset=['date'], keep='first')
        datasets[name] = datasets[name].drop_duplicates()
```
**Explanation:**
- First `drop_duplicates` removes duplicate dates (keeps first occurrence)
- Second `drop_duplicates` removes completely identical rows
- Prevents merge issues from duplicate keys

#### Lines 266-269: Create Year-Month Key
```python
for name, df in datasets.items():
    if name != 'gold':
        datasets[name] = datasets[name].sort_values('date', ascending=True).reset_index(drop=True)
        datasets[name]['year_month'] = datasets[name]['date'].dt.to_period('M').astype(str)
```
**Explanation:**
- Sorts by date (critical for time-series)
- Creates year_month column (e.g., '2025-01')
- This will be the merge key for combining datasets
- Gold already has year_month, so skipped

---

### STEP 4: HANDLE DIFFERENT FREQUENCIES

Some datasets are monthly, some are daily. Daily data needs to be converted to monthly.

#### Lines 285-290: Monthly Files Aggregation
```python
monthly_files = ['cpi', 'fedfunds', 'usd_inr']
for name in monthly_files:
    datasets[name] = datasets[name].groupby('year_month', as_index=False).agg({
        'date': 'first',
        name: 'mean'
    })
```
**Explanation:**
- Even monthly files may have multiple rows per month
- Groups by year_month and takes:
  - 'first' date (representative date)
  - 'mean' value (average if multiple values per month)
- Ensures one row per month

#### Lines 302-322: Daily Files to Monthly Conversion
```python
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
```
**Explanation:**
- For each daily file, groups by year_month
- Takes FIRST VALID VALUE of the month (first trading day with valid data)
- This is different from gold (which uses MEAN)
- Creates new DataFrame with one row per month

**Why first value?**
- For exchange rates and bond yields, first trading day represents opening conditions
- More stable than using last value (avoids end-of-month volatility)
- Standard practice in financial time-series

---

### STEP 5: MISSING VALUE HANDLING

#### Lines 335-339: Forward/Backward Fill
```python
for name, df in datasets.items():
    if name != 'gold':  # Gold already processed
        for col in df.columns:
            if col not in ['year_month', 'date']:
                datasets[name][col] = datasets[name][col].ffill().bfill()
```
**Explanation:**
- `ffill()` (forward fill): Fills missing values with last known value
- `bfill()` (backward fill): Fills missing values with next known value
- Applied to value columns only (not date/year_month)
- Gold already handled missing values, so skipped

**Why ffill().bfill()?**
- Forward fill handles gaps in middle of time-series
- Backward fill handles missing values at the beginning
- This is appropriate for time-series data (values persist over time)

---

### STEP 6: MERGE & FINAL MASTER DATASET

#### Line 350: Start with Gold as Base
```python
master_df = datasets['gold'][['year_month', 'gold_usd_per_gram']].copy()
```
**Explanation:**
- Starts with gold dataset (our target variable)
- Gold has the most complete date range (1975-2025)
- All other datasets will be merged onto this base

#### Lines 352-358: Merge All Datasets
```python
merge_order = ['cpi', 'fedfunds', 'usd_inr', 'usd_aud', 'usd_cad', 'usd_jpy', 'usd_chf', 'usd_gbp', 'treasury_10y']

for name in merge_order:
    if name in datasets and len(datasets[name]) > 0:
        df_to_merge = datasets[name][['year_month', name]].copy()
        master_df = master_df.merge(df_to_merge, on='year_month', how='left')
```
**Explanation:**
- Merges each dataset one by one
- `how='left'` keeps all gold dates, adds other data where available
- Only merges year_month and value column (drops date columns)
- Order doesn't matter for left joins

**Why left join?**
- Preserves all gold dates (complete timeline)
- Missing values in other datasets become NaN (handled later)
- Gold is the primary dataset (target variable)

#### Lines 360-361: Sort and Remove Duplicates
```python
master_df = master_df.sort_values('year_month', ascending=True).reset_index(drop=True)
master_df = master_df.drop_duplicates(subset=['year_month'], keep='first')
```
**Explanation:**
- Ensures chronological order
- Removes duplicate months (safety check)

#### Lines 363-366: Filter Date Range
```python
master_df = master_df[
    (master_df['year_month'] >= '1975-01') & 
    (master_df['year_month'] <= '2025-12')
].copy()
```
**Explanation:**
- Filters to desired date range (1975-2025)
- Gold data starts in 1975
- Ensures consistent timeline

#### Lines 368-381: Fill Missing Months
```python
all_months = pd.period_range(start='1975-01', end='2025-12', freq='M').astype(str)
missing_months = sorted(set(all_months) - set(master_df['year_month'].unique()))
if missing_months:
    # Create rows for missing months
    missing_rows = []
    for ym in missing_months:
        row_dict = {'year_month': ym}
        for col in master_df.columns:
            if col != 'year_month':
                row_dict[col] = np.nan
        missing_rows.append(row_dict)
    missing_df = pd.DataFrame(missing_rows)
    master_df = pd.concat([master_df, missing_df], ignore_index=True)
```
**Explanation:**
- Creates complete monthly timeline (all months from 1975-01 to 2025-12)
- Identifies missing months
- Creates rows with NaN values for missing months
- Concatenates to fill gaps

**Why fill missing months?**
- Ensures continuous time-series (no gaps)
- Makes feature engineering easier (lags, moving averages)
- Standard practice for time-series analysis

#### Lines 383-390: Fill Missing Gold Prices
```python
missing_before = master_df['gold_usd_per_gram'].isna().sum()
if missing_before > 0:
    master_df['gold_usd_per_gram'] = master_df['gold_usd_per_gram'].bfill().ffill()

for col in master_df.columns:
    if col != 'year_month':
        master_df[col] = master_df[col].ffill().bfill()
```
**Explanation:**
- Gold prices should never be missing (our target variable)
- Fills any gaps with forward/backward fill
- Then fills all other columns the same way

#### Lines 392-397: Drop Rows with Too Many Missing Values
```python
threshold = len(master_df.columns) * 0.5
rows_before = len(master_df)
master_df = master_df.dropna(thresh=threshold)
```
**Explanation:**
- `thresh=threshold` keeps rows with at least 50% non-null values
- Drops rows where >50% of columns are missing (likely data quality issues)
- Prevents rows with too little data

#### Lines 399-402: Final Missing Value Fill
```python
for col in master_df.columns:
    if col != 'year_month':
        if master_df[col].isna().sum() > 0:
            master_df[col] = master_df[col].ffill().bfill()
```
**Explanation:**
- Final pass to fill any remaining NaN values
- Ensures complete dataset (no missing values)

#### Lines 404-408: Remove Date Column
```python
if 'date' in master_df.columns:
    master_df = master_df.drop(columns=['date'])
    print("  ✓ Date column removed (keeping only year_month as temporal index)")
```
**Explanation:**
- Removes date column (we only need year_month)
- Simplifies dataset (year_month is the temporal index)
- Requirement: Only year_month should remain

---

### STEP 7: FINAL VALIDATION

This step performs comprehensive validation to ensure data quality.

#### Lines 417-420: Basic Statistics
```python
print(f"\n[7.1] Master dataset created:")
print(f"  Shape: {master_df.shape}")
print(f"  Columns: {list(master_df.columns)}")
print(f"  Date range: {master_df['year_month'].min()} to {master_df['year_month'].max()}")
```
**Explanation:**
- Prints dataset dimensions and columns
- Verifies date range

#### Lines 422-425: Check Date Column Removed
```python
if 'date' in master_df.columns:
    raise ValueError("FAILED: date column still exists!")
else:
    print("  PASSED: date column removed")
```
**Explanation:**
- Validates that date column was removed (requirement)
- Raises error if still present

#### Lines 427-431: Check Duplicates
```python
duplicates = master_df.duplicated(subset=['year_month'], keep=False).sum()
if duplicates > 0:
    raise ValueError(f"FAILED: {duplicates} duplicate year_month entries found!")
else:
    print("  PASSED: One row per year_month")
```
**Explanation:**
- Ensures exactly one row per month
- Duplicates would cause merge issues later

#### Lines 433-442: Check Gold Price Variation
```python
monthly_changes = master_df['gold_usd_per_gram'].diff().abs()
flat_months = (monthly_changes < 0.001).sum()

yearly_variation = master_df.groupby(master_df['year_month'].str[:4])['gold_usd_per_gram'].agg(['nunique'])
yearly_blocks = yearly_variation[yearly_variation['nunique'] == 1]
if len(yearly_blocks) > 5:
    raise ValueError(f"FAILED: {len(yearly_blocks)} years with flat gold prices!")
```
**Explanation:**
- Checks that gold prices vary month-to-month (not flat)
- Flat prices indicate data quality issues
- Allows up to 5 flat years (some years may be stable)

#### Lines 444-450: Check for Extreme Drops
```python
monthly_pct_change = master_df['gold_usd_per_gram'].pct_change() * 100
extreme_drops = (monthly_pct_change < -50).sum()
if extreme_drops > 0:
    raise ValueError(f"FAILED: {extreme_drops} months with >50% drop!")
```
**Explanation:**
- Validates no months with >50% price drops (unrealistic)
- Extreme drops indicate data errors
- Gold prices should change gradually

#### Lines 452-466: Check Key Periods
```python
periods_to_check = {
    '2017-2019': ('2017-01', '2019-12'),
    '2020-2022': ('2020-01', '2022-12'),
    '2023-2025': ('2023-01', '2025-12')
}
for period_name, (start, end) in periods_to_check.items():
    period_data = master_df[(master_df['year_month'] >= start) & (master_df['year_month'] <= end)].copy()
    if len(period_data) > 0:
        unique_values = period_data['gold_usd_per_gram'].nunique()
        if unique_values < len(period_data) * 0.8:
            print(f"  WARNING: Low variation in {period_name}")
```
**Explanation:**
- Checks specific time periods for price variation
- Ensures >80% unique values (prices change over time)
- These periods are important (pre-COVID, COVID, post-COVID)

#### Lines 468-472: Validate Price Range
```python
invalid_gold = master_df[master_df['gold_usd_per_gram'] < 1]
if len(invalid_gold) > 0:
    raise ValueError(f"FAILED: {len(invalid_gold)} rows with gold < $1/gram!")
```
**Explanation:**
- Ensures all prices are >= $1/gram (reasonable minimum)
- Prices below $1/gram are unrealistic

#### Lines 474-493: Validate Timeline Completeness
```python
expected_months = (2025 - 1975 + 1) * 12
actual_months = len(master_df)
if actual_months != expected_months:
    # Check for missing months
```
**Explanation:**
- Calculates expected number of months (612 months from 1975-2025)
- Verifies we have complete timeline
- Reports any missing months

#### Lines 495-501: Final Sorting and Column Ordering
```python
is_sorted = master_df['year_month'].is_monotonic_increasing
if not is_sorted:
    master_df = master_df.sort_values('year_month', ascending=True).reset_index(drop=True)

column_order = ['year_month'] + [col for col in master_df.columns if col != 'year_month']
master_df = master_df[column_order]
```
**Explanation:**
- Ensures chronological order
- Places year_month as first column (standard format)

#### Lines 503-505: Save Master Dataset
```python
master_df.to_csv(output_dir / "master_df.csv", index=False)
print(f"\n[7.2] Master dataset saved to: {output_dir / 'master_df.csv'}")
```
**Explanation:**
- Saves final master dataset to CSV
- `index=False` prevents saving row numbers as column
- This file will be used in Phase 2 (Feature Engineering)

---

## SUMMARY

Phase 1 accomplishes:

1. **Loads 10 datasets** from CSV files (3 monthly, 6 daily, 1 gold with special handling)
2. **Standardizes columns** (renames to consistent format)
3. **Handles date formats** (multiple formats, robust parsing)
4. **Converts frequencies** (daily → monthly using first valid value)
5. **Processes gold specially** (MM-DD-YYYY dates, MEAN aggregation, USD/ounce → USD/gram)
6. **Fills missing values** (forward/backward fill)
7. **Merges all datasets** (left join on year_month)
8. **Validates data quality** (no duplicates, price variation, complete timeline)
9. **Saves master dataset** (ready for feature engineering)

**Output:** `outputs/master_df.csv` with columns:
- year_month (temporal index)
- gold_usd_per_gram (target variable)
- cpi, fedfunds, usd_inr, usd_aud, usd_cad, usd_jpy, usd_chf, usd_gbp, treasury_10y (features)

**Key Principles:**
- Gold uses MEAN aggregation (more stable)
- Exchange rates use FIRST value (opening conditions)
- Complete timeline (no gaps)
- No missing values (filled with forward/backward fill)
- Chronological order (critical for time-series)

