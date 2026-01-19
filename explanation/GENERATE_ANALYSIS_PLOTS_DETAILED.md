# GENERATE_ANALYSIS_PLOTS.PY - DETAILED EXPLANATION

## Overview
`generate_analysis_plots.py` generates visualizations and analysis data for the Flask application's About page. It creates correlation heatmaps, feature importance plots, model prediction comparisons, and dataset summaries. All plots are saved as high-resolution PNG images in the `static/` directory for display on the web application.

---

## STEP-BY-STEP BREAKDOWN

### SECTION 1: IMPORTS AND SETUP

#### Lines 8-14: Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from sklearn.inspection import permutation_importance
```
**Explanation:**
- **pandas/numpy**: Data manipulation and numerical operations
- **matplotlib.pyplot**: Plotting library (creates figures and charts)
- **seaborn**: Statistical visualization library (enhances matplotlib)
- **pickle**: Loads saved models
- **Path**: Path handling (cross-platform file paths)
- **permutation_importance**: Calculates feature importance for SVR (which doesn't have built-in importance)

#### Lines 20-23: Directory Setup
```python
input_dir = Path("outputs")
models_dir = Path("models")
plots_dir = Path("static")
plots_dir.mkdir(exist_ok=True)
```
**Explanation:**
- **input_dir**: Directory containing processed data (from previous phases)
- **models_dir**: Directory containing saved models
- **plots_dir**: Directory for saving plots (static folder for Flask)
- **mkdir(exist_ok=True)**: Creates directory if it doesn't exist (no error if exists)
- All generated plots will be saved to `static/` directory

#### Lines 25-26: Configure Plotting Style
```python
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
```
**Explanation:**
- **sns.set_style("whitegrid")**: Sets seaborn style to white background with grid
  - Makes plots cleaner and more professional
  - Grid helps with reading values
- **plt.rcParams['figure.figsize']**: Sets default figure size to 12×8 inches
  - Larger figures for better readability
  - Can be overridden for specific plots

---

### SECTION 2: LOAD DATA AND MODELS

#### Lines 30-33: Load Data Files
```python
features_df = pd.read_csv(input_dir / "features_df.csv")
test_df = pd.read_csv(input_dir / "test_df.csv")
X_test = pd.read_csv(input_dir / "X_test.csv")
y_test = pd.read_csv(input_dir / "y_test.csv").iloc[:, 0]
```
**Explanation:**
- **features_df**: Feature-engineered dataset (from Phase 2)
- **test_df**: Full test dataset with year_month (from Phase 3)
- **X_test**: Test feature matrix (from Phase 3)
- **y_test**: Test target vector (from Phase 3)
- **.iloc[:, 0]**: Extracts first column as Series (y_test CSV has single column)
- All data needed for analysis and visualization

#### Lines 35-48: Load Models
```python
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
```
**Explanation:**
- **Context managers (`with`)**: Automatically close files after loading
- **'rb' mode**: Read binary (required for pickle files)
- **pickle.load()**: Deserializes Python objects (restores trained models)
- **Models loaded**: XGBoost, LightGBM, Random Forest, SVR
- **Scaler loaded**: StandardScaler for SVR (needed for scaling features)

#### Lines 47-57: Load Feature Columns and Create Dictionary
```python
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
```
**Explanation:**
- **feature_cols**: List of feature column names (in correct order)
- **models_dict**: Dictionary organizing all models and related data
- **Purpose**: Easy access to models and metadata
- **Used for**: Iterating over models, accessing scaler, getting feature names

---

### SECTION 3: CORRELATION ANALYSIS

#### Lines 64-81: Prepare Data for Correlation
```python
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
```
**Explanation:**
- **master_df**: Loads original cleaned dataset (from Phase 1)
- **Feature creation**: Creates missing features if not present
  - **log_gold**: Natural log of gold price
  - **cpi_yoy**: CPI year-over-year inflation rate
  - **real_rate**: Real interest rate (fedfunds - cpi_yoy)
- **corr_cols**: Columns to include in correlation analysis
- **gold_label**: Label for display purposes
- **Purpose**: Analyzes relationships between gold prices and macro indicators

#### Lines 81-90: Compute and Visualize Correlation
```python
corr_data = master_df[corr_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title(f'Correlation Analysis: {gold_label} vs Macro Indicators', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(plots_dir / "correlation_analysis.png", dpi=300, bbox_inches='tight')
plt.close()
```
**Explanation:**
- **.corr()**: Computes correlation matrix (Pearson correlation coefficients)
- **plt.figure()**: Creates new figure with specified size (10×8 inches)
- **sns.heatmap()**: Creates correlation heatmap
  - **annot=True**: Shows correlation values in cells
  - **fmt='.3f'**: Formats numbers to 3 decimal places
  - **cmap='coolwarm'**: Color scheme (blue=negative, red=positive, white=zero)
  - **center=0**: Centers colormap at zero (neutral correlation)
  - **square=True**: Makes cells square (better appearance)
  - **linewidths=1**: Adds lines between cells
  - **cbar_kws**: Colorbar settings (shrink=0.8 makes it smaller)
- **plt.title()**: Adds title with font size and padding
- **plt.tight_layout()**: Adjusts layout to prevent label cutoff
- **plt.savefig()**: Saves plot as PNG
  - **dpi=300**: High resolution (300 dots per inch)
  - **bbox_inches='tight'**: Removes extra whitespace
- **plt.close()**: Closes figure (frees memory)

**Correlation Interpretation:**
- **Range**: -1 to +1
- **+1**: Perfect positive correlation (move together)
- **-1**: Perfect negative correlation (move opposite)
- **0**: No correlation (independent)
- **High absolute value**: Strong relationship

---

### SECTION 4: FEATURE IMPORTANCE

#### Lines 95-130: Generate Feature Importance for Tree-Based Models
```python
print("\n[3] Generating feature importance plots...")

models_names = {
    'xgboost': 'XGBoost',
    'lightgbm': 'LightGBM',
    'random_forest': 'Random Forest'
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes = axes.flatten()
```
**Explanation:**
- **models_names**: Maps model keys to display names
- **plt.subplots(1, 3)**: Creates figure with 1 row, 3 columns (3 subplots)
- **figsize=(18, 6)**: Figure size (18 inches wide, 6 inches tall)
- **axes.flatten()**: Converts 2D axes array to 1D (easier indexing)
- **Purpose**: Creates side-by-side comparison of feature importance

#### Lines 106-125: Plot Feature Importance for Each Model
```python
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
```
**Explanation:**
- **Loop**: Iterates over tree-based models (XGBoost, LightGBM, Random Forest)
- **.feature_importances_**: Attribute containing importance scores
  - Tree-based models compute importance automatically
  - Measures how much each feature contributes to predictions
- **np.argsort(importances)[::-1][:15]**: 
  - Sorts indices by importance (descending)
  - `[::-1]` reverses order (highest first)
  - `[:15]` takes top 15 features
- **top_features/top_importances**: Extracts top 15 feature names and scores
- **axes[idx].barh()**: Creates horizontal bar chart
  - **range(len(top_features))**: Y-axis positions (0, 1, 2, ...)
  - **top_importances**: Bar lengths (importance values)
- **set_yticks/set_yticklabels()**: Sets Y-axis labels (feature names)
- **set_xlabel()**: Sets X-axis label
- **set_title()**: Sets subplot title
- **invert_yaxis()**: Inverts Y-axis (highest importance at top)

**Feature Importance Explained:**
- **Higher importance**: Feature has more influence on predictions
- **Tree-based models**: Use information gain/split quality to compute importance
- **Interpretation**: Identifies which features are most predictive

#### Lines 127-130: Save Feature Importance Plot
```python
plt.tight_layout()
plt.savefig(plots_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Feature importance plot saved")
```
**Explanation:**
- **tight_layout()**: Adjusts spacing between subplots
- **savefig()**: Saves as PNG (high resolution)
- **close()**: Closes figure
- **Result**: Single image with 3 side-by-side bar charts

---

### SECTION 5: SVR PERMUTATION IMPORTANCE

#### Lines 132-153: Calculate and Plot SVR Permutation Importance
```python
print("\n[4] Calculating SVR permutation importance (this may take a while)...")
svr_model = models_dict['svr']
X_test_scaled = scaler.transform(X_test)

perm_importance = permutation_importance(svr_model, X_test_scaled, y_test, 
                                        n_repeats=10, random_state=42, n_jobs=-1)
```
**Explanation:**
- **Why permutation importance?**: SVR doesn't have built-in feature importance
- **Permutation importance**: Measures importance by randomly shuffling each feature
  - If shuffling a feature decreases model performance → high importance
  - If shuffling has little effect → low importance
- **X_test_scaled**: Scales test features (SVR requires scaled features)
- **permutation_importance()**: Computes importance
  - **n_repeats=10**: Shuffles each feature 10 times (for stability)
  - **random_state=42**: Seed for reproducibility
  - **n_jobs=-1**: Uses all CPU cores (faster computation)
- **Note**: This can take several minutes (computationally intensive)

#### Lines 140-152: Extract and Plot Top Features
```python
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
```
**Explanation:**
- **importances_mean**: Average importance across n_repeats (more stable)
- **np.argsort()[::-1][:15]**: Gets top 15 features (same as tree models)
- **plt.barh()**: Creates horizontal bar chart
- **plt.gca()**: Gets current axes object
- **invert_yaxis()**: Highest importance at top
- **Result**: Separate plot for SVR feature importance

**Permutation Importance vs Built-in Importance:**
- **Built-in (tree models)**: Fast, model-specific
- **Permutation (SVR)**: Slower, model-agnostic, more interpretable
- **Permutation method**: Works for any model (universal approach)

---

### SECTION 6: MODEL PREDICTIONS COMPARISON

#### Lines 158-190: Generate Predictions from All Models
```python
print("\n[5] Generating model predictions comparison...")

predictions_log = {}

try:
    predictions_log['XGBoost'] = models_dict['xgboost'].predict(X_test)
    print("  ✓ XGBoost predictions generated")
except Exception as e:
    print(f"  ⚠️  Warning: XGBoost prediction failed: {e}")
    predictions_log['XGBoost'] = np.zeros(len(X_test)) * np.nan
```
**Explanation:**
- **predictions_log**: Dictionary to store log-scale predictions
- **Try-except blocks**: Handles prediction failures gracefully
- **.predict(X_test)**: Generates predictions on test set
- **Error handling**: If prediction fails, creates NaN array (prevents script crash)
- **Warning messages**: Alerts user to issues (helps debugging)
- **Repeated for**: XGBoost, LightGBM, Random Forest, SVR

**Special Handling:**
- **SVR**: Uses scaled features (`scaler.transform(X_test)`)
- **Random Forest**: Includes compatibility warning (sklearn version issues)
- **Error tolerance**: Script continues even if one model fails

#### Lines 192-199: Convert to Original Scale
```python
y_test_original = np.exp(y_test.values)
predictions_original = {}
for k, v in predictions_log.items():
    if isinstance(v, np.ndarray) and not np.isnan(v).all():
        predictions_original[k] = np.exp(v)
    else:
        predictions_original[k] = v
```
**Explanation:**
- **y_test_original**: Converts test targets from log to original scale
- **Back-transformation**: exp(log_value) = original_value
- **predictions_original**: Converts all predictions to original scale
- **NaN handling**: Skips conversion if predictions are all NaN
- **Purpose**: Plot in original scale (USD/gram) for interpretability

#### Lines 201-206: Create Comparison DataFrame
```python
test_dates = pd.to_datetime(test_df['year_month'])
comparison_df = pd.DataFrame({
    'Date': test_dates,
    'Actual': y_test_original,
    **{k: v for k, v in predictions_original.items()}
})
```
**Explanation:**
- **test_dates**: Converts year_month to datetime objects
- **comparison_df**: DataFrame with dates, actual values, and all predictions
- **Dictionary unpacking (`**{...}`)**: Adds all prediction columns dynamically
- **Result**: DataFrame with columns: Date, Actual, XGBoost, LightGBM, Random Forest, SVR
- **Purpose**: Easy plotting and analysis

#### Lines 208-222: Create Comparison Plot
```python
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
```
**Explanation:**
- **plt.figure()**: Creates figure (14×8 inches, larger for time-series)
- **plt.plot()**: Creates line plots
  - **Actual**: 'ko-' = black circles with solid line
  - **Predictions**: 'b--', 'r--', 'g--', 'm--' = blue, red, green, magenta dashed lines
  - **linewidth=2**: Thicker lines (more visible)
  - **markersize=8**: Larger markers for actual (easier to see)
- **plt.xlabel/ylabel()**: Axis labels with font size
- **plt.title()**: Plot title with font size and padding
- **plt.legend()**: Adds legend (identifies each line)
- **plt.grid()**: Adds grid (alpha=0.3 makes it subtle)
- **plt.xticks(rotation=45)**: Rotates date labels (prevents overlap)
- **savefig()**: Saves high-resolution PNG
- **Result**: Time-series plot showing actual vs predicted values

**Plot Interpretation:**
- **Closer lines**: Better predictions (model matches actual)
- **Overlapping predictions**: Models agree on predictions
- **Diverging predictions**: Models disagree (uncertainty indicator)
- **Trends**: Shows if models capture price trends correctly

---

### SECTION 7: DATASET SUMMARY

#### Lines 228-253: Generate Dataset Summary
```python
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
        # ... (similar for other columns)
    ]
}

dataset_summary_df = pd.DataFrame(dataset_summary)
dataset_summary_df.to_csv(input_dir / "dataset_summary.csv", index=False)
```
**Explanation:**
- **Purpose**: Creates summary table of data sources
- **Dictionary structure**: Maps column names to lists of values
- **Dataset names**: Lists all 10 data sources
- **Frequency**: Original frequency (Monthly or Daily→Monthly)
- **Date Range**: Uses first_valid_index() and last_valid_index()
  - **Note**: This approach uses row indices (not actual dates)
  - **Better approach**: Would use actual date ranges from year_month column
- **DataFrame creation**: Converts dictionary to DataFrame
- **to_csv()**: Saves summary to CSV file
  - **index=False**: Doesn't save row numbers
- **Result**: CSV file with dataset information (for About page display)

**Note on Date Range:**
- Current implementation uses row indices (0, 611) rather than actual dates
- Could be improved to use actual year_month values
- Still serves purpose of documenting data sources

---

## OUTPUT FILES

All files are saved to the `static/` directory (except dataset_summary.csv):

1. **correlation_analysis.png** (300 DPI)
   - Correlation heatmap: Gold vs Macro Indicators
   - Shows relationships between variables
   - Color-coded (coolwarm colormap)

2. **feature_importance.png** (300 DPI)
   - Side-by-side comparison (3 subplots)
   - Top 15 features for XGBoost, LightGBM, Random Forest
   - Horizontal bar charts

3. **svr_permutation_importance.png** (300 DPI)
   - Separate plot for SVR feature importance
   - Top 15 features using permutation method
   - Horizontal bar chart

4. **model_predictions_comparison.png** (300 DPI)
   - Time-series plot: Actual vs All Model Predictions
   - Test period (2020-2025)
   - Multiple lines (actual + 4 models)

5. **dataset_summary.csv** (saved to outputs/)
   - Summary table of data sources
   - Columns: Dataset, Frequency, Date Range
   - 10 rows (one per data source)

---

## KEY CONCEPTS

### Correlation Analysis:
- **Pearson correlation**: Measures linear relationships
- **Range**: -1 to +1
- **Heatmap**: Visual representation of correlation matrix
- **Color coding**: Red (positive), Blue (negative), White (zero)

### Feature Importance:
- **Tree-based models**: Built-in importance (information gain)
- **SVR**: Permutation importance (model-agnostic)
- **Interpretation**: Higher importance = more predictive
- **Top features**: Most influential features identified

### Permutation Importance:
- **Method**: Shuffle feature, measure performance drop
- **Computational cost**: Expensive (requires multiple predictions)
- **Advantage**: Works for any model
- **Stability**: Uses mean across multiple repeats

### Prediction Comparison:
- **Time-series plot**: Shows predictions over time
- **Multi-model comparison**: All models on same plot
- **Actual vs Predicted**: Visual performance assessment
- **Trend analysis**: Identifies if models capture trends

### Visualization Best Practices:
- **High resolution**: 300 DPI for publication quality
- **Clear labels**: Axis labels, titles, legends
- **Color coding**: Distinct colors for different models
- **Grid lines**: Helps read values accurately
- **Layout**: Tight layout prevents label cutoff

---

## SUMMARY

`generate_analysis_plots.py` accomplishes:

1. **Loads data and models** (from previous phases)
2. **Generates correlation analysis** (heatmap of relationships)
3. **Computes feature importance** (for all models)
4. **Creates prediction comparisons** (time-series plots)
5. **Generates dataset summary** (data source documentation)
6. **Saves high-resolution plots** (300 DPI PNG files)
7. **Handles errors gracefully** (continues if one model fails)

**Key Features:**
- Comprehensive visualizations (correlation, importance, predictions)
- High-quality output (300 DPI, professional styling)
- Error handling (robust to prediction failures)
- Multiple importance methods (built-in and permutation)
- Time-series analysis (prediction comparison over time)

**Output**: 4 PNG images + 1 CSV file, ready for Flask About page display

**Usage**: Run after Phase 4 (model training) to generate analysis visualizations

