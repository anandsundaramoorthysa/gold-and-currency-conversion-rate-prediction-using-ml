# PHASE 2: FEATURE ENGINEERING - DETAILED EXPLANATION

## Overview
Phase 2 transforms the cleaned master dataset into a feature-rich dataset ready for machine learning. It creates derived features (macroeconomic indicators, lags, technical indicators) and prepares the target variable. This phase uses log-scale transformations for time-series stability and reduces overfitting by selecting essential features only.

---

## STEP-BY-STEP BREAKDOWN

### STEP 1: LOAD MASTER DATASET

#### Lines 16-18: Directory Setup
```python
input_dir = Path("outputs")
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)
```
**Explanation:**
- Sets input and output directories (both use 'outputs' folder)
- Creates output directory if it doesn't exist
- Input comes from Phase 1's master_df.csv

#### Lines 20-23: Load Master Dataset
```python
master_df = pd.read_csv(input_dir / "master_df.csv")
print(f"  Shape: {master_df.shape}")
print(f"  Date range: {master_df['year_month'].min()} to {master_df['year_month'].max()}")
```
**Explanation:**
- Loads the cleaned master dataset from Phase 1
- Displays shape (rows, columns) and date range
- Verifies data loaded correctly

#### Line 28: Copy Master Dataset
```python
features_df = master_df.copy()
```
**Explanation:**
- Creates a copy to avoid modifying the original
- All feature engineering will be done on features_df
- Original master_df remains unchanged

---

### STEP 2: MACRO DERIVED FEATURES

These features capture macroeconomic relationships and trends.

#### Lines 35-36: CPI Year-over-Year (YoY) Inflation
```python
features_df['cpi_yoy'] = (features_df['cpi'] / features_df['cpi'].shift(12) - 1) * 100
print("  ✓ CPI YoY Inflation calculated")
```
**Explanation:**
- **Formula**: (CPI_current / CPI_12_months_ago - 1) × 100
- **Purpose**: Measures annual inflation rate (percentage change)
- **Why 12 months?**: Compares same month last year (removes seasonality)
- **shift(12)**: Gets CPI value from 12 months earlier
- **Result**: Percentage (e.g., 2.5 means 2.5% inflation)
- **Why important**: Inflation affects gold prices (gold is inflation hedge)

**Example:**
- If CPI in Jan 2025 = 300 and CPI in Jan 2024 = 292
- CPI YoY = (300/292 - 1) × 100 = 2.74% inflation

#### Lines 38-39: Real Interest Rate
```python
features_df['real_rate'] = features_df['fedfunds'] - features_df['cpi_yoy']
print("  ✓ Real Interest Rate calculated")
```
**Explanation:**
- **Formula**: Nominal Interest Rate (fedfunds) - Inflation Rate (cpi_yoy)
- **Purpose**: Real interest rate = nominal rate minus inflation
- **Why important**: 
  - Positive real rate: Money in bank earns more than inflation → Less attractive to hold gold
  - Negative real rate: Inflation erodes savings → Gold becomes attractive
- **Key insight**: Gold prices often rise when real rates are negative or low

**Example:**
- If fedfunds = 5% and cpi_yoy = 3%
- Real rate = 5% - 3% = 2% (positive, gold less attractive)

#### Lines 41-42: Yield Spread
```python
features_df['yield_spread'] = features_df['treasury_10y'] - features_df['fedfunds']
print("  ✓ Yield Spread calculated")
```
**Explanation:**
- **Formula**: 10-Year Treasury Yield - Federal Funds Rate
- **Purpose**: Measures the difference between long-term and short-term rates
- **Why important**:
  - Positive spread (normal): Long-term rates > short-term rates
  - Negative spread (inverted yield curve): Signals economic uncertainty → Gold often rises
  - Large spread: Indicates economic growth expectations
- **Gold connection**: Yield curve inversions often precede gold price increases

**Example:**
- If treasury_10y = 4.5% and fedfunds = 5%
- Yield spread = 4.5% - 5% = -0.5% (inverted, bearish signal)

---

### STEP 3: LOG TRANSFORMATION (CRITICAL FOR TIME-SERIES STABILITY)

Log transformations are essential for time-series data because:
1. **Stabilizes variance**: Gold prices increase over time, variance increases → Log stabilizes variance
2. **Makes relationships linear**: Multiplicative relationships become additive
3. **Normalizes distribution**: Makes data more normally distributed
4. **Reduces impact of outliers**: Log compresses large values

#### Lines 49-50: Log Gold Price
```python
features_df['log_gold'] = np.log(features_df['gold_usd_per_gram'])
print("  ✓ Log gold price calculated (log_gold)")
```
**Explanation:**
- **Formula**: log(gold_price)
- **Purpose**: Converts gold price to log scale
- **Why**: 
  - Prices grow exponentially → Log makes growth linear
  - Easier for models to learn (linear relationships)
  - Reduces impact of extreme values
- **Note**: This is the foundation for target variable

**Example:**
- If gold = $70/gram, log_gold = ln(70) ≈ 4.248
- If gold = $80/gram, log_gold = ln(80) ≈ 4.382
- Difference in log = 0.134 (represents ~14% increase)

#### Lines 52-53: Gold Log Returns (1-Month)
```python
features_df['gold_return_1m'] = features_df['log_gold'] - features_df['log_gold'].shift(1)
print("  ✓ Gold log returns (1-month) calculated")
```
**Explanation:**
- **Formula**: log_gold(t) - log_gold(t-1)
- **Purpose**: Measures month-over-month return (log scale)
- **Why log returns?**:
  - Log difference = percentage change (approximately)
  - More stable than raw percentage changes
  - Additive (can sum over time)
- **Why important**: Captures momentum/trend in gold prices
- **shift(1)**: Gets previous month's log_gold value

**Mathematical Property**: 
- log(A) - log(B) = log(A/B) ≈ (A-B)/B (percentage change)
- This is approximately the percentage change from previous month

**Example:**
- If log_gold(t) = 4.382 and log_gold(t-1) = 4.248
- gold_return_1m = 4.382 - 4.248 = 0.134
- This represents ~14% increase (e^0.134 - 1 ≈ 0.143)

---

### STEP 4: LAG FEATURES (REDUCED SET - PREVENT OVERFITTING)

Lag features use past values to predict future values. This is critical for time-series forecasting.

**Why lag features?**
- Gold prices have momentum (autocorrelation)
- Past prices influence future prices
- Captures short-term trends and patterns

**Why reduced set?**
- Too many lags → Overfitting (model memorizes training data)
- Reduced set (lags 1, 3) prevents overfitting while capturing key patterns

#### Lines 60-62: Gold Lag Features
```python
features_df['gold_lag1'] = features_df['gold_usd_per_gram'].shift(1)
features_df['gold_lag3'] = features_df['gold_usd_per_gram'].shift(3)
print("  ✓ Gold lags (1, 3 months) - lag2 removed")
```
**Explanation:**
- **gold_lag1**: Gold price from 1 month ago (immediate past)
- **gold_lag3**: Gold price from 3 months ago (short-term trend)
- **Why lag 1?**: Most recent price is strongest predictor
- **Why lag 3?**: Captures 3-month trends (quarterly patterns)
- **Why removed lag 2?**: Reduces overfitting (lag 1 and 3 provide sufficient information)
- **shift(1)**: Moves values down by 1 row (previous month)
- **shift(3)**: Moves values down by 3 rows (3 months ago)

**Example:**
- If gold in Jan = $70, Feb = $72, Mar = $71, Apr = $75
- For April row: gold_lag1 = $71 (March), gold_lag3 = $70 (January)

#### Lines 64-67: Macro Lag Features
```python
features_df['fedfunds_lag1'] = features_df['fedfunds'].shift(1)
features_df['cpi_yoy_lag1'] = features_df['cpi_yoy'].shift(1)
features_df['real_rate_lag1'] = features_df['real_rate'].shift(1)
print("  ✓ Macro lags (1 month each) - removed lag2 variants")
```
**Explanation:**
- **fedfunds_lag1**: Federal Funds Rate from 1 month ago
- **cpi_yoy_lag1**: CPI YoY from 1 month ago
- **real_rate_lag1**: Real interest rate from 1 month ago
- **Why lag 1 only?**: 
  - Macro variables change slowly (month-to-month persistence)
  - Current values may not be known at prediction time (use lag 1)
  - Prevents data leakage (using future information)
- **Why removed lag 2?**: Reduces overfitting (lag 1 is sufficient)

**Data Leakage Prevention:**
- When predicting gold for month T, we only use information available before month T
- Lag 1 ensures we use data from month T-1 (known at prediction time)

---

### STEP 5: TECHNICAL FEATURES (MINIMAL - PREVENT OVERFITTING)

Technical features capture price trends and patterns.

#### Lines 74-75: Gold Moving Average (3-Month)
```python
features_df['gold_ma3'] = features_df['gold_usd_per_gram'].rolling(window=3, min_periods=1).mean()
print("  ✓ Gold MA3 (removed MA12)")
```
**Explanation:**
- **Formula**: Average of gold prices over last 3 months
- **Purpose**: Captures short-term trend (smooths out noise)
- **rolling(window=3)**: Creates a rolling window of 3 months
- **min_periods=1**: Allows calculation even with fewer than 3 months (uses available data)
- **.mean()**: Calculates average within window
- **Why MA3?**: Short-term trend indicator (not too sensitive, not too slow)
- **Why removed MA12?**: Reduces overfitting (MA3 is sufficient)

**How it works:**
- For month T: MA3 = (gold[T] + gold[T-1] + gold[T-2]) / 3
- Smooths out month-to-month fluctuations
- Rising MA3 = uptrend, Falling MA3 = downtrend

**Example:**
- If gold in Jan=$70, Feb=$72, Mar=$71
- MA3 for March = (70+72+71)/3 = $71

---

### STEP 6: CREATE TARGET VARIABLE (LOG SCALE)

#### Lines 79-81: Target Variable Creation
```python
features_df['log_gold_next'] = features_df['log_gold'].shift(-1)
print("  ✓ Target variable created: log_gold_next (log scale)")
print("  ✓ Removed: gold_usd_per_gram_next_month (raw price target - causes overfitting)")
```
**Explanation:**
- **Formula**: log_gold_next = log_gold shifted backward by 1 month
- **Purpose**: Creates target variable (what we're predicting)
- **shift(-1)**: Moves values UP by 1 row (next month's value)
- **Why log scale?**: 
  - More stable for time-series (reduces variance)
  - Models perform better on log scale
  - Can back-transform: exp(log_prediction) → original scale
- **Why not raw price?**: Raw price target causes overfitting (model memorizes instead of learning patterns)

**How it works:**
- For month T: log_gold_next = log_gold from month T+1
- We predict next month's log gold price
- After prediction, convert back: prediction = exp(log_prediction)

**Example:**
- If log_gold in Jan = 4.248, Feb = 4.382, Mar = 4.356
- For January row: log_gold_next = 4.382 (February's log gold)
- We're predicting February's price using January's features

---

### STEP 7: SELECT ESSENTIAL FEATURES

#### Lines 85-97: Feature Selection
```python
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
```
**Explanation:**
- **Purpose**: Selects only essential features (reduces overfitting)
- **Removed features**:
  - gold_lag2 (redundant with lag1 and lag3)
  - fedfunds_lag2, cpi_yoy_lag2 (redundant with lag1)
  - usd_inr_lags (weak predictive power)
  - gold_ma12 (redundant with ma3)
  - volatility features (noisy, cause overfitting)
  - raw macro variables (using lagged versions instead)
- **Why reduce features?**: 
  - Prevents overfitting (fewer parameters to learn)
  - Improves generalization (works better on new data)
  - Faster training and prediction
- **Feature categories**:
  1. **Temporal**: year_month
  2. **Price lags**: gold_lag1, gold_lag3
  3. **Returns**: gold_return_1m
  4. **Technical**: gold_ma3
  5. **Macro (current)**: real_rate, yield_spread
  6. **Macro (lagged)**: real_rate_lag1, fedfunds_lag1, cpi_yoy_lag1
  7. **Target**: log_gold_next

**Total features**: 10 features + 1 target = 11 columns

---

### STEP 8: FINAL CLEANUP

#### Lines 105-109: Remove Missing Values
```python
initial_rows = len(features_df)
features_df = features_df.dropna()
final_rows = len(features_df)
print(f"  Dropped {initial_rows - final_rows} rows with NaN values")
```
**Explanation:**
- **dropna()**: Removes rows with any missing values
- **Why necessary?**: 
  - Lag features and rolling windows create NaN values at the beginning
  - CPI YoY requires 12 months of history → Creates NaN for first 12 months
  - Models cannot handle NaN values
- **Trade-off**: Loses some data but ensures clean dataset

**Example:**
- If starting with 612 rows and first 15 rows have NaN
- After dropna(): 597 rows (15 rows removed)

#### Lines 111: Sort by Date
```python
features_df = features_df.sort_values('year_month', ascending=True).reset_index(drop=True)
```
**Explanation:**
- Ensures chronological order (critical for time-series)
- `reset_index(drop=True)`: Creates clean index (0, 1, 2, ...)
- Prevents index issues in later phases

---

### STEP 9: SUMMARY AND SAVE

#### Lines 114-126: Display Summary
```python
print("\n" + "=" * 80)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 80)
print(f"Total features: {len(features_df.columns)}")
print(f"Total rows: {len(features_df)}")
print(f"\nFeature list:")
for i, col in enumerate(features_df.columns, 1):
    is_target = " (TARGET)" if col == 'log_gold_next' else ""
    print(f"  {i:2d}. {col}{is_target}")
```
**Explanation:**
- Displays comprehensive summary
- Shows total features and rows
- Lists all features with target marked
- Helps verify feature engineering completed correctly

#### Lines 129-131: Save Features Dataset
```python
output_path = output_dir / "features_df.csv"
features_df.to_csv(output_path, index=False)
print(f"\n[9] Features dataset saved to: {output_path}")
```
**Explanation:**
- Saves final features dataset to CSV
- `index=False`: Doesn't save row numbers
- This file will be used in Phase 3 (Train/Test Split)

---

## FEATURE SUMMARY

### Final Feature Set (10 Features + 1 Target):

1. **year_month**: Temporal index (YYYY-MM format)
2. **gold_lag1**: Gold price from 1 month ago (USD/gram)
3. **gold_lag3**: Gold price from 3 months ago (USD/gram)
4. **gold_return_1m**: 1-month log return (percentage change)
5. **gold_ma3**: 3-month moving average (trend indicator)
6. **real_rate**: Real interest rate (fedfunds - cpi_yoy)
7. **real_rate_lag1**: Real rate from 1 month ago
8. **fedfunds_lag1**: Federal Funds Rate from 1 month ago
9. **cpi_yoy_lag1**: CPI YoY inflation from 1 month ago
10. **yield_spread**: Treasury 10Y - Fed Funds (yield curve)
11. **log_gold_next**: Target variable (next month's log gold price)

### Feature Categories:

- **Price Features** (3): gold_lag1, gold_lag3, gold_ma3
- **Return Features** (1): gold_return_1m
- **Macro Features** (6): real_rate, real_rate_lag1, fedfunds_lag1, cpi_yoy_lag1, yield_spread

### Key Design Decisions:

1. **Log Scale Target**: Prevents overfitting, improves model stability
2. **Reduced Lag Set**: Only lags 1 and 3 (removed lag 2 to prevent overfitting)
3. **Lagged Macros**: Use lag 1 to prevent data leakage
4. **Minimal Technical Features**: Only MA3 (removed MA12 and volatility)
5. **Essential Features Only**: Removed weak/redundant features

---

## MATHEMATICAL CONCEPTS

### Log Transformations:
- **Natural log**: ln(x) = log_e(x)
- **Property**: ln(a) - ln(b) = ln(a/b) ≈ percentage change
- **Back-transform**: exp(ln(x)) = x

### Lag Operations:
- **shift(n)**: Moves values down by n rows (past values)
- **shift(-n)**: Moves values up by n rows (future values)
- **shift(1)**: Previous month
- **shift(12)**: Same month last year

### Rolling Windows:
- **rolling(window=k)**: Creates window of k consecutive values
- **.mean()**: Average within window
- **min_periods=m**: Minimum m values needed for calculation

---

## OUTPUT

**File**: `outputs/features_df.csv`

**Columns**: 11 columns (10 features + 1 target)
- year_month, gold_lag1, gold_lag3, gold_return_1m, gold_ma3, real_rate, real_rate_lag1, fedfunds_lag1, cpi_yoy_lag1, yield_spread, log_gold_next

**Rows**: ~597 rows (after removing NaN rows)

**Ready for**: Phase 3 (Train/Test Split)

