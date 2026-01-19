# Data Preprocessing Explanation: How `master_df.csv` is Created

## Overview

The preprocessing pipeline transforms raw CSV files from different sources into a unified monthly time-series dataset (`master_df.csv`) suitable for machine learning. This process handles different data formats, frequencies, and quality issues.

---

## Step-by-Step Process

### **STEP 1: LOAD & INSPECT ALL CSV FILES**

**Purpose**: Load all 10 datasets and understand their structure.

**Files Loaded**:
- **Monthly Files (3)**:
  - CPI (Consumer Price Index): `observation_date`, `CPIAUCSL`
  - Fed Funds Rate: `observation_date`, `FEDFUNDS`
  - USD-INR: `observation_date`, `EXINUS`

- **Daily Files (7)**:
  - USD-AUD: `observation_date`, `DEXUSAL`
  - USD-CAD: `observation_date`, `DEXCAUS`
  - USD-JPY: `observation_date`, `DEXJPUS`
  - USD-CHF: `observation_date`, `DEXSZUS`
  - USD-GBP: `observation_date`, `DEXUSUK`
  - Gold Futures: `Date`, `Price` (with commas in numbers)
  - Treasury 10Y: `observation_date`, `DGS10`

**Inspection Checks**:
- Shape (rows × columns)
- Column names
- Missing values count
- Date range
- Sample data display

---

### **STEP 2: COLUMN SELECTION & RENAMING**

**Purpose**: Keep only required columns and standardize naming.

**Actions**:
1. Select only the relevant columns for each dataset
2. Rename columns to standardized format:
   - All date columns → `date`
   - Value columns → standardized names (e.g., `cpi`, `fedfunds`, `usd_inr`, etc.)
3. Clean Gold Price: Remove commas from numbers (e.g., "4,386.30" → 4386.30)

**Result**: Each dataset now has exactly 2 columns: `date` and the feature value.

**Example**:
```python
# Before:
CPI: ['observation_date', 'CPIAUCSL', ...]
# After:
CPI: ['date', 'cpi']
```

---

### **STEP 3: DATE STANDARDIZATION**

**Purpose**: Convert all dates to datetime format and create merge key.

**Actions**:
1. **Parse Dates**: Convert date strings (DD-MM-YYYY format) to pandas datetime
   - Handles different date formats automatically
   - Returns `NaT` for invalid dates

2. **Remove Invalid Dates**: Drop rows where date parsing failed

3. **Remove Duplicates**:
   - Remove duplicate rows
   - Remove duplicate dates (keep first occurrence)

4. **Sort Ascending**: Sort all datasets by date (oldest → newest)

5. **Create Merge Key**: Create `year_month` column in format "YYYY-MM"
   - Example: "2025-01-15" → "2025-01"
   - This becomes the **ONLY merge key** for combining datasets

**Result**: All datasets are chronologically ordered with standardized `year_month` key.

---

### **STEP 4: HANDLE DIFFERENT FREQUENCIES**

**Purpose**: Convert all data to monthly frequency for consistency.

#### **Monthly Files (Use As-Is)**
- CPI, Fed Funds, USD-INR are already monthly
- Group by `year_month` and take mean if duplicates exist (safety measure)

#### **Daily Files → Monthly Conversion**

**Strategy**: Use **first available trading day of each month**

**Logic**:
```python
For each month (year_month):
  1. Group all daily records for that month
  2. Sort by date (ascending)
  3. Find first row with non-null value
  4. Use that value as the monthly value
```

**Why First Trading Day?**
- Handles missing 1st of month (weekends/holidays)
- Preserves market opening signal
- Captures early-month market sentiment
- More stable than using average or last day

**Result**: All daily files converted to monthly frequency (one record per month).

---

### **STEP 5: GOLD UNIT CONVERSION**

**Purpose**: Convert gold prices from USD per ounce to USD per gram.

**Conversion**:
```python
gold_usd_per_gram = gold_usd / 31.1035
```

**Why?**
- Raw data: Gold price in USD per ounce
- Target: Gold price in USD per gram (model requirement)
- Conversion factor: 1 ounce = 31.1035 grams

**Example**:
- If gold = $2000/ounce
- Then gold_per_gram = 2000 / 31.1035 = $64.32/gram

---

### **STEP 6: MISSING VALUE HANDLING**

**Purpose**: Fill missing values before merging.

**Strategy**:
1. **Forward Fill (ffill)**: Fill missing values with previous valid value
   - Example: If day 3 is missing, use day 2's value

2. **Backward Fill (bfill)**: Fill remaining missing values with next valid value
   - Example: If first value is missing, use second value

**Why This Approach?**
- Preserves time-series continuity
- Better than dropping rows (maintains time coverage)
- Better than mean imputation (preserves temporal patterns)

---

### **STEP 7: MERGE & FINAL MASTER DATASET**

**Purpose**: Combine all datasets into one unified dataset.

#### **Merge Strategy**:

1. **Start with Gold Dataset** as base (starts from 1975)
   - Initial columns: `year_month`, `date`, `gold_usd_per_gram`

2. **Sequential Outer Joins**:
   ```python
   master_df = gold_df
   master_df = master_df.merge(cpi_df, on='year_month', how='outer')
   master_df = master_df.merge(fedfunds_df, on='year_month', how='outer')
   master_df = master_df.merge(usd_inr_df, on='year_month', how='outer')
   # ... continue for all datasets
   ```

3. **Outer Join Benefits**:
   - Keeps all months from all datasets
   - Creates NaNs where data is missing (handled later)

4. **Post-Merge Cleaning**:
   - **Sort**: By `year_month` ascending
   - **Remove Duplicates**: Keep first occurrence per month
   - **Filter Date Range**: 1975-01 to 2025-12
   - **Drop Sparse Rows**: Remove rows with >50% missing values
   - **Final Fill**: Forward fill + backward fill remaining NaNs

#### **Final Master Dataset Structure**:

```
Columns:
- year_month (YYYY-MM format) - Merge key
- date (datetime)
- gold_usd_per_gram (target variable)
- cpi
- fedfunds
- usd_inr
- usd_aud
- usd_cad
- usd_jpy
- usd_chf
- usd_gbp
- treasury_10y
```

**Final Statistics**:
- **Rows**: 612 months (1975-01 to 2025-12)
- **Columns**: 12 (year_month, date, 10 features)
- **Missing Values**: 0 (all filled)

---

## Visual Flow Diagram

```
Raw CSV Files (10 files)
        ↓
[STEP 1] Load & Inspect
        ↓
[STEP 2] Select & Rename Columns
        ↓
[STEP 3] Standardize Dates → Create year_month
        ↓
[STEP 4] Convert Frequencies
    ├── Monthly files → Keep as-is
    └── Daily files → First trading day per month
        ↓
[STEP 5] Gold Conversion (oz → gram)
        ↓
[STEP 6] Handle Missing Values (ffill + bfill)
        ↓
[STEP 7] Merge All Datasets (outer join on year_month)
        ↓
    Clean & Filter
        ↓
master_df.csv (612 rows × 12 columns)
```

---

## Key Design Decisions

### 1. **Why First Trading Day for Daily → Monthly?**
- More reliable than averaging (no weekend/holiday issues)
- Captures market opening sentiment
- Standard practice in financial time-series

### 2. **Why year_month as Merge Key?**
- Single standard format (YYYY-MM)
- Works regardless of specific day within month
- Prevents merge mismatches

### 3. **Why Outer Join?**
- Preserves all time periods
- Missing values handled with forward/backward fill
- Maintains maximum time coverage

### 4. **Why Filter to 1975-2025?**
- Gold data starts in 1975 (boundary)
- Ensures all features have data for this period
- Standard time range for analysis

### 5. **Why Drop Rows with >50% Missing?**
- Rows with too many missing values are unreliable
- Better to remove than impute everything
- Maintains data quality

---

## Output: `master_df.csv`

**Location**: `outputs/master_df.csv`

**Format**:
```csv
year_month,date,gold_usd_per_gram,cpi,fedfunds,usd_inr,usd_aud,usd_cad,usd_jpy,usd_chf,usd_gbp,treasury_10y
1975-01,1975-01-02,5.883582,52.3,7.13,8.13,1.324,0.9917,300.66,2.5727,2.333,7.42
1975-02,1975-01-02,5.883582,52.6,6.24,7.9717,1.34,0.9999,297.44,2.5202,2.375,7.49
...
2025-12,2025-01-01,91.475236,325.031,3.72,90.035,0.6554,1.3982,155.28,0.8035,1.3229,4.19
```

**Characteristics**:
- ✅ All dates in chronological order
- ✅ Monthly frequency (one row per month)
- ✅ No missing values
- ✅ Standardized column names
- ✅ Gold in USD per gram
- ✅ Ready for feature engineering

---

## Next Steps

After `master_df.csv` is created:
1. **Feature Engineering** (Phase 2): Create derived features, lags, technical indicators
2. **Model Training** (Phase 4): Use this clean dataset to train ML models

---

## Running the Preprocessing

```bash
python phase1_data_preprocessing.py
```

**Expected Output**:
- Progress messages for each step
- Dataset inspection results
- Final master dataset statistics
- Saved file: `outputs/master_df.csv`

---

## Summary

The preprocessing pipeline:
1. ✅ Handles 10 different datasets with varying formats
2. ✅ Standardizes dates and creates merge key
3. ✅ Converts daily data to monthly (first trading day)
4. ✅ Converts gold prices to USD per gram
5. ✅ Handles missing values intelligently
6. ✅ Merges all datasets into unified master file
7. ✅ Produces clean, ready-to-use dataset for ML

**Result**: A clean, unified monthly time-series dataset ready for machine learning!

