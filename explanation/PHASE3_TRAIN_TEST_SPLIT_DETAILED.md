# PHASE 3: TRAIN / VALIDATION / TEST SPLIT - DETAILED EXPLANATION

## Overview
Phase 3 splits the feature-engineered dataset into three subsets: training, validation, and test sets. This is a **time-series split**, meaning the data is split chronologically (not randomly) to preserve the temporal order. This prevents data leakage and ensures realistic model evaluation.

---

## STEP-BY-STEP BREAKDOWN

### STEP 1: SETUP AND LOAD DATA

#### Lines 18-20: Directory Setup
```python
input_dir = Path("outputs")
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)
```
**Explanation:**
- Sets input and output directories (both use 'outputs' folder)
- Creates output directory if it doesn't exist
- Input comes from Phase 2's features_df.csv

#### Lines 22-25: Load Features Dataset
```python
features_df = pd.read_csv(input_dir / "features_df.csv")
print(f"  Shape: {features_df.shape}")
print(f"  Date range: {features_df['year_month'].min()} to {features_df['year_month'].max()}")
```
**Explanation:**
- Loads the feature-engineered dataset from Phase 2
- Displays shape (rows, columns) and date range
- Verifies data loaded correctly

---

### STEP 2: TIME SERIES SPLIT

#### Lines 32-37: Define Split Boundaries
```python
train_start = '1977-01'
train_end = '2015-12'
val_start = '2016-01'
val_end = '2019-12'
test_start = '2020-01'
test_end = '2025-12'
```
**Explanation:**
- **Train Period**: 1977-01 to 2015-12 (39 years, ~468 months)
  - **Why 1977?**: After feature engineering (lags, CPI YoY), first ~15 rows are removed (NaN values)
  - **Purpose**: Model learns historical patterns
  - **Period**: Stable economic period (pre-financial crisis, pre-COVID)
  
- **Validation Period**: 2016-01 to 2019-12 (4 years, ~48 months)
  - **Purpose**: Tune hyperparameters and select best model
  - **Period**: Pre-COVID period (different regime from training)
  - **Why important**: Tests model on unseen but similar economic conditions
  
- **Test Period**: 2020-01 to 2025-12 (6 years, ~72 months)
  - **Purpose**: Final evaluation (true out-of-sample)
  - **Period**: COVID and post-COVID period (different regime)
  - **Why important**: Tests model's ability to generalize to completely new conditions

**Split Rationale:**
- **Time-series ordering**: Training → Validation → Test (chronological)
- **Regime separation**: Each period represents different economic conditions
- **No data leakage**: Test set is completely future data (never seen during training)

#### Lines 39-44: Create Splits
```python
train_df = features_df[(features_df['year_month'] >= train_start) & 
                       (features_df['year_month'] <= train_end)].copy()
val_df = features_df[(features_df['year_month'] >= val_start) & 
                     (features_df['year_month'] <= val_end)].copy()
test_df = features_df[(features_df['year_month'] >= test_start) & 
                      (features_df['year_month'] <= test_end)].copy()
```
**Explanation:**
- **Boolean indexing**: Creates masks based on date conditions
- **& operator**: Combines conditions (both must be true)
- **.copy()**: Creates independent copies (prevents reference issues)
- **Result**: Three separate DataFrames with non-overlapping date ranges

**Why .copy()?**
- Without copy(), changes to one DataFrame might affect others
- Copy ensures independence (safe for later processing)

#### Lines 46-48: Print Split Statistics
```python
print(f"\n  Train:   {len(train_df)} rows ({train_df['year_month'].min()} to {train_df['year_month'].max()})")
print(f"  Validation: {len(val_df)} rows ({val_df['year_month'].min()} to {val_df['year_month'].max()})")
print(f"  Test:    {len(test_df)} rows ({test_df['year_month'].min()} to {test_df['year_month'].max()})")
```
**Explanation:**
- Displays number of rows and date ranges for each split
- Verifies splits created correctly
- Helps identify any issues (e.g., empty splits, incorrect dates)

**Expected Output:**
- Train: ~468 rows (1977-01 to 2015-12)
- Validation: ~48 rows (2016-01 to 2019-12)
- Test: ~72 rows (2020-01 to 2025-12)

---

### STEP 3: SEPARATE FEATURES AND TARGET

#### Lines 55-57: Define Exclusion Columns
```python
exclude_cols = ['year_month', 'date', 'log_gold_next', 'gold_usd_per_gram_next_month']
exclude_cols = [col for col in exclude_cols if col in features_df.columns]
```
**Explanation:**
- **exclude_cols**: Columns to exclude from features (X)
- **year_month**: Temporal index (not a feature)
- **date**: Date column (if exists, not a feature)
- **log_gold_next**: Target variable (what we're predicting)
- **gold_usd_per_gram_next_month**: Alternative target (if exists)
- **List comprehension**: Only includes columns that actually exist (prevents errors)

**Why exclude these?**
- **year_month/date**: Not predictive features (just identifiers)
- **log_gold_next**: This is the target (y), not a feature (X)
- We need to separate X (features) and y (target)

#### Line 58: Define Feature Columns
```python
feature_cols = [col for col in features_df.columns if col not in exclude_cols]
```
**Explanation:**
- **List comprehension**: Creates list of all columns except excluded ones
- **Result**: List of feature column names (e.g., ['gold_lag1', 'gold_lag3', ...])
- These will be the input features (X) for the model

**Feature columns should include:**
- gold_lag1, gold_lag3, gold_return_1m, gold_ma3
- real_rate, real_rate_lag1, fedfunds_lag1, cpi_yoy_lag1, yield_spread

#### Lines 60-67: Create X and y for Each Split
```python
X_train = train_df[feature_cols].copy()
y_train = train_df['log_gold_next'].copy()

X_val = val_df[feature_cols].copy()
y_val = val_df['log_gold_next'].copy()

X_test = test_df[feature_cols].copy()
y_test = test_df['log_gold_next'].copy()
```
**Explanation:**
- **X_train/X_val/X_test**: Feature matrices (input variables)
- **y_train/y_val/y_test**: Target vectors (what we're predicting)
- **.copy()**: Creates independent copies
- **Result**: 6 separate objects (3 X's and 3 y's)

**Data Structure:**
- **X**: DataFrame with multiple columns (features)
- **y**: Series with single column (target variable)
- Each split has its own X and y

#### Lines 69-72: Print Feature Information
```python
print(f"  Features: {len(feature_cols)}")
print(f"  Feature names: {feature_cols}")
print(f"\n  ⚠️  TARGET: log_gold_next (log scale)")
print(f"     Predictions will be back-transformed: exp(log_pred)")
```
**Explanation:**
- Displays number of features and feature names
- Reminds that target is in log scale
- Notes that predictions will be converted back to original scale
- Helps verify correct feature/target separation

---

### STEP 4: SAVE SPLITS

#### Lines 76-78: Save Full DataFrames (with year_month)
```python
train_df.to_csv(output_dir / "train_df.csv", index=False)
val_df.to_csv(output_dir / "val_df.csv", index=False)
test_df.to_csv(output_dir / "test_df.csv", index=False)
```
**Explanation:**
- Saves complete DataFrames (includes year_month and all features)
- Useful for analysis and debugging
- **index=False**: Doesn't save row numbers

**Files created:**
- train_df.csv: Full training data (with year_month)
- val_df.csv: Full validation data (with year_month)
- test_df.csv: Full test data (with year_month)

#### Lines 80-85: Save X and y Separately
```python
X_train.to_csv(output_dir / "X_train.csv", index=False)
y_train.to_csv(output_dir / "y_train.csv", index=False)
X_val.to_csv(output_dir / "X_val.csv", index=False)
y_val.to_csv(output_dir / "y_val.csv", index=False)
X_test.to_csv(output_dir / "X_test.csv", index=False)
y_test.to_csv(output_dir / "y_test.csv", index=False)
```
**Explanation:**
- Saves features (X) and target (y) separately
- This format is ready for model training
- **X files**: Feature matrices (multiple columns)
- **y files**: Target vectors (single column)

**Files created:**
- X_train.csv, X_val.csv, X_test.csv: Feature matrices
- y_train.csv, y_val.csv, y_test.csv: Target vectors

#### Lines 87-88: Save Feature Column Names
```python
with open(output_dir / "feature_columns.pkl", 'wb') as f:
    pickle.dump(feature_cols, f)
```
**Explanation:**
- Saves feature column names as a pickle file
- **Pickle format**: Preserves Python objects (lists, dictionaries, etc.)
- **Why save?**: 
  - Needed in Phase 4 (model training) to ensure correct feature order
  - Needed in app.py (prediction) to build feature vectors
- **'wb' mode**: Write binary (required for pickle)

**Why pickle?**
- Preserves exact column order and names
- Easy to load: `pickle.load()` retrieves the list
- Ensures consistency across phases

#### Line 90: Confirmation Message
```python
print("  ✓ All splits saved to outputs/")
```
**Explanation:**
- Confirms all files saved successfully
- Helps track progress

---

## WHY TIME-SERIES SPLIT (NOT RANDOM SPLIT)?

### Random Split (WRONG for Time-Series):
```
❌ Random split would mix past and future data
   - Training set might contain 2020 data
   - Test set might contain 1980 data
   - Model "sees the future" during training → Data leakage!
```

### Time-Series Split (CORRECT):
```
✅ Chronological split preserves temporal order
   - Training: 1977-2015 (past)
   - Validation: 2016-2019 (recent past)
   - Test: 2020-2025 (future)
   - Model never sees future data during training
```

### Key Principles:
1. **No data leakage**: Test set is completely future data
2. **Realistic evaluation**: Tests model on unseen future periods
3. **Regime separation**: Each split represents different economic conditions
4. **Temporal order**: Training → Validation → Test (chronological)

---

## SPLIT RATIONALE

### Training Set (1977-2015):
- **Size**: ~468 months (largest split)
- **Purpose**: Learn historical patterns
- **Characteristics**: 
  - Pre-financial crisis period
  - Pre-COVID period
  - Stable economic conditions
  - Long history (39 years)

### Validation Set (2016-2019):
- **Size**: ~48 months (medium split)
- **Purpose**: 
  - Tune hyperparameters
  - Select best model
  - Early stopping
- **Characteristics**:
  - Pre-COVID period
  - Different regime from training (recent data)
  - Tests generalization to unseen but similar conditions

### Test Set (2020-2025):
- **Size**: ~72 months (smallest split)
- **Purpose**: 
  - Final evaluation
  - True out-of-sample performance
  - Real-world performance estimate
- **Characteristics**:
  - COVID and post-COVID period
  - Completely different regime
  - Tests model's robustness to extreme conditions

---

## DATA LEAKAGE PREVENTION

### What is Data Leakage?
Using information from the future to predict the past (or using test data during training).

### How This Split Prevents Leakage:
1. **Temporal separation**: Test set is chronologically after training set
2. **No overlap**: No date appears in multiple splits
3. **Future data excluded**: Model never sees 2020-2025 data during training
4. **Feature construction**: All features use only past/current information (lags, not leads)

### Example of Leakage (Prevented):
```
❌ WRONG: Using future data in features
   - Feature: gold_price_next_month (leakage!)
   - Feature: future_cpi (leakage!)

✅ CORRECT: Using only past/current data
   - Feature: gold_lag1 (past data)
   - Feature: cpi_yoy_lag1 (past data)
```

---

## OUTPUT FILES

### Full DataFrames (with year_month):
1. **train_df.csv**: Training data (468 rows × 11 columns)
2. **val_df.csv**: Validation data (48 rows × 11 columns)
3. **test_df.csv**: Test data (72 rows × 11 columns)

### Feature Matrices (X):
4. **X_train.csv**: Training features (468 rows × 10 columns)
5. **X_val.csv**: Validation features (48 rows × 10 columns)
6. **X_test.csv**: Test features (72 rows × 10 columns)

### Target Vectors (y):
7. **y_train.csv**: Training target (468 rows × 1 column)
8. **y_val.csv**: Validation target (48 rows × 1 column)
9. **y_test.csv**: Test target (72 rows × 1 column)

### Metadata:
10. **feature_columns.pkl**: List of feature column names

**Total**: 10 files saved to `outputs/` directory

---

## SUMMARY

Phase 3 accomplishes:

1. **Loads feature-engineered data** from Phase 2
2. **Defines split boundaries** (train: 1977-2015, val: 2016-2019, test: 2020-2025)
3. **Creates time-series splits** (chronological, no overlap)
4. **Separates features and target** (X and y for each split)
5. **Saves all splits** (full DataFrames, X matrices, y vectors, feature names)

**Key Principles:**
- Time-series split (not random)
- No data leakage (future data excluded from training)
- Regime separation (different economic periods)
- Ready for model training (Phase 4)

**Output**: 10 files in `outputs/` directory, ready for Phase 4 (Model Training)

