# PHASE 4: MODEL TRAINING - DETAILED EXPLANATION

## Overview
Phase 4 trains multiple machine learning models to predict gold prices. It uses regularized hyperparameters to prevent overfitting, trains on log-scale targets, evaluates models using proper metrics, and selects the best model based on validation performance. Four models are trained: XGBoost, LightGBM, Random Forest, and SVR.

---

## STEP-BY-STEP BREAKDOWN

### STEP 1: SETUP AND LOAD DATA

#### Lines 26-31: Directory Setup
```python
input_dir = Path("outputs")
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
```
**Explanation:**
- Sets input/output directories (both use 'outputs')
- Creates 'models' directory for saving trained models
- All trained models will be saved as pickle files

#### Lines 33-40: Load Train/Validation/Test Splits
```python
X_train = pd.read_csv(input_dir / "X_train.csv")
y_train_log = pd.read_csv(input_dir / "y_train.csv").iloc[:, 0]
X_val = pd.read_csv(input_dir / "X_val.csv")
y_val_log = pd.read_csv(input_dir / "y_val.csv").iloc[:, 0]
X_test = pd.read_csv(input_dir / "X_test.csv")
y_test_log = pd.read_csv(input_dir / "y_test.csv").iloc[:, 0]
```
**Explanation:**
- Loads feature matrices (X) and target vectors (y) from Phase 3
- **.iloc[:, 0]**: Extracts first column as Series (y files have single column)
- **Suffix "_log"**: Reminds that targets are in log scale
- All data is ready for training

#### Lines 42-44: Convert to Original Scale (for Evaluation)
```python
y_train_original = np.exp(y_train_log)
y_val_original = np.exp(y_val_log)
y_test_original = np.exp(y_test_log)
```
**Explanation:**
- **Back-transformation**: exp(log_value) converts log scale to original scale
- **Purpose**: Evaluate models on original scale (USD/gram)
- **Why needed?**: 
  - Models predict in log scale (more stable)
  - But we want to evaluate in original scale (more interpretable)
  - RMSE and MAE need original scale for meaningful interpretation

**Mathematical Property**: 
- If log_prediction = 4.248, then original_prediction = exp(4.248) ≈ 70 USD/gram

#### Lines 46-47: Load Feature Column Names
```python
with open(input_dir / "feature_columns.pkl", 'rb') as f:
    feature_cols = pickle.load(f)
```
**Explanation:**
- Loads feature column names from Phase 3
- **'rb' mode**: Read binary (required for pickle)
- **Purpose**: Ensures correct feature order (important for predictions)

#### Lines 49-52: Print Dataset Information
```python
print(f"  Train:   {X_train.shape} (y in log scale)")
print(f"  Validation: {X_val.shape} (y in log scale)")
print(f"  Test:    {X_test.shape} (y in log scale)")
print(f"\n  ⚠️  IMPORTANT: Predicting log scale, will back-transform for RMSE/MAE")
```
**Explanation:**
- Displays dataset shapes
- Reminds that predictions are in log scale
- Notes that metrics will be computed on original scale

---

### STEP 2: EVALUATION FUNCTION

#### Lines 54-72: Evaluate Model Function
```python
def evaluate_model(y_true_log, y_pred_log, y_true_original, model_name, dataset_name):
    """
    Evaluate model on both log scale (R²) and original scale (RMSE, MAE)
    y_true_log, y_pred_log: predictions in log scale
    y_true_original: actual gold prices in original scale (for RMSE/MAE)
    """
    r2_log = r2_score(y_true_log, y_pred_log)
    
    y_pred_original = np.exp(y_pred_log)
    
    rmse_original = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
    mae_original = mean_absolute_error(y_true_original, y_pred_original)
    
    print(f"\n  {model_name} - {dataset_name}:")
    print(f"    RMSE (original scale): {rmse_original:.6f}")
    print(f"    MAE  (original scale): {mae_original:.6f}")
    print(f"    R²   (log scale):      {r2_log:.6f}")
    
    return {'rmse': rmse_original, 'mae': mae_original, 'r2': r2_log}
```
**Explanation:**
- **Purpose**: Comprehensive model evaluation function
- **Parameters**:
  - `y_true_log`: Actual values (log scale)
  - `y_pred_log`: Predictions (log scale)
  - `y_true_original`: Actual values (original scale, for RMSE/MAE)
  - `model_name`: Name of model (for printing)
  - `dataset_name`: Name of dataset (Train/Val/Test)
- **Metrics computed**:
  1. **R² (log scale)**: Coefficient of determination (measures fit quality)
  2. **RMSE (original scale)**: Root Mean Squared Error (penalizes large errors)
  3. **MAE (original scale)**: Mean Absolute Error (average error magnitude)
- **Why R² on log scale?**: 
  - R² measures correlation/fit quality (works well on log scale)
  - More stable metric for time-series
- **Why RMSE/MAE on original scale?**: 
  - More interpretable (error in USD/gram)
  - Users care about dollar errors, not log errors
- **Returns**: Dictionary with all metrics

**Metrics Explained:**
- **R²**: 0 to 1 (higher is better), measures proportion of variance explained
- **RMSE**: Always positive (lower is better), penalizes large errors more
- **MAE**: Always positive (lower is better), average error magnitude

---

### STEP 3: MODEL 1 - XGBOOST REGRESSOR

#### Lines 81-91: XGBoost Model Configuration
```python
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=3,              # FIXED: Reduced from 6 to 3 (prevent overfitting)
    learning_rate=0.03,       # FIXED: Reduced from 0.1 to 0.03 (slower learning)
    subsample=0.7,            # FIXED: Added subsample (prevent overfitting)
    colsample_bytree=0.7,     # FIXED: Added column subsampling
    reg_alpha=1.0,            # FIXED: Added L1 regularization
    reg_lambda=1.0,           # FIXED: Added L2 regularization
    random_state=42,
    n_jobs=-1
)
```
**Explanation:**
- **XGBoost**: Gradient boosting algorithm (ensemble of decision trees)
- **Hyperparameters (regularized to prevent overfitting)**:
  - `n_estimators=300`: Number of trees (boosting rounds)
  - `max_depth=3`: Maximum tree depth (reduced from 6 → prevents overfitting)
  - `learning_rate=0.03`: Step size (reduced from 0.1 → slower, more stable learning)
  - `subsample=0.7`: Use 70% of rows per tree (prevents overfitting)
  - `colsample_bytree=0.7`: Use 70% of features per tree (prevents overfitting)
  - `reg_alpha=1.0`: L1 regularization (penalizes large coefficients)
  - `reg_lambda=1.0`: L2 regularization (penalizes large coefficients)
  - `random_state=42`: Seed for reproducibility
  - `n_jobs=-1`: Use all CPU cores (faster training)

**Why Regularization?**
- Prevents overfitting (model memorizing training data)
- Improves generalization (works better on new data)
- More stable predictions

#### Lines 93-101: Train and Evaluate XGBoost
```python
xgb_model.fit(X_train, y_train_log)

xgb_train_pred_log = xgb_model.predict(X_train)
xgb_val_pred_log = xgb_model.predict(X_val)
xgb_test_pred_log = xgb_model.predict(X_test)

xgb_train_metrics = evaluate_model(y_train_log, xgb_train_pred_log, y_train_original, "XGBoost", "Train")
xgb_val_metrics = evaluate_model(y_val_log, xgb_val_pred_log, y_val_original, "XGBoost", "Validation")
xgb_test_metrics = evaluate_model(y_test_log, xgb_test_pred_log, y_test_original, "XGBoost", "Test")
```
**Explanation:**
- **.fit()**: Trains model on training data (X_train, y_train_log)
- **.predict()**: Makes predictions on train/val/test sets
- **Predictions in log scale**: All predictions are log values
- **Evaluation**: Computes metrics for each dataset
- **Metrics stored**: Saved for later comparison

---

### STEP 4: MODEL 2 - LIGHTGBM REGRESSOR

#### Lines 110-120: LightGBM Model Configuration
```python
lgb_model = lgb.LGBMRegressor(
    n_estimators=300,
    num_leaves=15,        
    max_depth=3,              
    learning_rate=0.03,      
    subsample=0.7,           
    colsample_bytree=0.7,    
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
```
**Explanation:**
- **LightGBM**: Fast gradient boosting (similar to XGBoost but faster)
- **Hyperparameters (similar regularization)**:
  - `n_estimators=300`: Number of trees
  - `num_leaves=15`: Maximum leaves per tree (controls complexity)
  - `max_depth=3`: Maximum tree depth (regularization)
  - `learning_rate=0.03`: Step size (slow learning)
  - `subsample=0.7`: Row subsampling
  - `colsample_bytree=0.7`: Feature subsampling
  - `verbose=-1`: Suppress output (no training logs)

**LightGBM vs XGBoost:**
- LightGBM: Faster training, uses leaf-wise growth
- XGBoost: More robust, uses level-wise growth
- Both are gradient boosting algorithms

#### Lines 122-130: Train and Evaluate LightGBM
```python
lgb_model.fit(X_train, y_train_log)

lgb_train_pred_log = lgb_model.predict(X_train)
lgb_val_pred_log = lgb_model.predict(X_val)
lgb_test_pred_log = lgb_model.predict(X_test)

lgb_train_metrics = evaluate_model(y_train_log, lgb_train_pred_log, y_train_original, "LightGBM", "Train")
lgb_val_metrics = evaluate_model(y_val_log, lgb_val_pred_log, y_val_original, "LightGBM", "Validation")
lgb_test_metrics = evaluate_model(y_test_log, lgb_test_pred_log, y_test_original, "LightGBM", "Test")
```
**Explanation:**
- Same process as XGBoost
- Train, predict, evaluate on all three datasets

---

### STEP 5: MODEL 3 - RANDOM FOREST REGRESSOR

#### Lines 139-145: Random Forest Model Configuration
```python
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=5,              # FIXED: Reduced from 10 to 5 (prevent overfitting)
    min_samples_leaf=20,      # FIXED: Added min_samples_leaf (require more samples per leaf)
    random_state=42,
    n_jobs=-1
)
```
**Explanation:**
- **Random Forest**: Ensemble of decision trees (bagging method)
- **Hyperparameters (regularized)**:
  - `n_estimators=300`: Number of trees
  - `max_depth=5`: Maximum tree depth (reduced from 10 → prevents overfitting)
  - `min_samples_leaf=20`: Minimum samples per leaf (prevents overfitting)
  - `random_state=42`: Reproducibility
  - `n_jobs=-1`: Parallel processing

**Random Forest vs Gradient Boosting:**
- Random Forest: Parallel trees, bagging (bootstrap aggregation)
- Gradient Boosting: Sequential trees, boosting (error correction)
- Random Forest: More robust to overfitting, faster training

#### Lines 147-155: Train and Evaluate Random Forest
```python
rf_model.fit(X_train, y_train_log)

rf_train_pred_log = rf_model.predict(X_train)
rf_val_pred_log = rf_model.predict(X_val)
rf_test_pred_log = rf_model.predict(X_test)

rf_train_metrics = evaluate_model(y_train_log, rf_train_pred_log, y_train_original, "Random Forest", "Train")
rf_val_metrics = evaluate_model(y_val_log, rf_val_pred_log, y_val_original, "Random Forest", "Validation")
rf_test_metrics = evaluate_model(y_test_log, rf_test_pred_log, y_test_original, "Random Forest", "Test")
```
**Explanation:**
- Same process as previous models
- Train, predict, evaluate

---

### STEP 6: MODEL 4 - SUPPORT VECTOR REGRESSION (SVR)

#### Lines 161-164: Scale Features for SVR
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```
**Explanation:**
- **StandardScaler**: Standardizes features (mean=0, std=1)
- **Why needed?**: SVR is sensitive to feature scale
- **.fit_transform()**: Fit on training data, then transform
- **.transform()**: Apply same scaling to val/test (no fitting!)
- **Important**: Never fit scaler on validation/test data (data leakage!)

**Standardization Formula:**
- z = (x - mean) / std
- All features have mean=0, std=1 after scaling

#### Lines 166-171: SVR Model Configuration
```python
svr_model = SVR(
    kernel='rbf',
    C=10,
    gamma=0.1,
    epsilon=0.01
)
```
**Explanation:**
- **SVR**: Support Vector Regression (non-linear regression)
- **Hyperparameters**:
  - `kernel='rbf'`: Radial Basis Function (non-linear kernel)
  - `C=10`: Regularization parameter (higher = less regularization)
  - `gamma=0.1`: Kernel coefficient (controls influence radius)
  - `epsilon=0.01`: Error tolerance (predictions within epsilon are not penalized)
- **Why SVR?**: Baseline model (linear methods), handles non-linear relationships

#### Lines 172-180: Train and Evaluate SVR
```python
svr_model.fit(X_train_scaled, y_train_log)

svr_train_pred_log = svr_model.predict(X_train_scaled)
svr_val_pred_log = svr_model.predict(X_val_scaled)
svr_test_pred_log = svr_model.predict(X_test_scaled)

svr_train_metrics = evaluate_model(y_train_log, svr_train_pred_log, y_train_original, "SVR", "Train")
svr_val_metrics = evaluate_model(y_val_log, svr_val_pred_log, y_val_original, "SVR", "Validation")
svr_test_metrics = evaluate_model(y_test_log, svr_test_pred_log, y_test_original, "SVR", "Test")
```
**Explanation:**
- Uses scaled features (X_train_scaled, etc.)
- Same evaluation process

---

### STEP 7: NAÏVE BASELINE MODEL

#### Lines 189-192: Load Data for Naïve Baseline
```python
master_df_for_naive = pd.read_csv(input_dir / "master_df.csv")
master_df_for_naive['year_month'] = pd.to_datetime(master_df_for_naive['year_month'])
master_df_for_naive['log_gold'] = np.log(master_df_for_naive['gold_usd_per_gram'])
```
**Explanation:**
- Loads master dataset (has gold prices)
- Converts year_month to datetime
- Creates log_gold column

#### Lines 194-199: Load Split DataFrames
```python
train_df = pd.read_csv(input_dir / "train_df.csv")
val_df = pd.read_csv(input_dir / "val_df.csv")
test_df = pd.read_csv(input_dir / "test_df.csv")
# Convert year_month to datetime
```
**Explanation:**
- Loads full DataFrames (with year_month)
- Needed to merge with master_df for naïve predictions

#### Lines 201-207: Create Naïve Predictions
```python
train_merged = train_df[['year_month']].merge(master_df_for_naive[['year_month', 'log_gold']], on='year_month', how='left')
val_merged = val_df[['year_month']].merge(master_df_for_naive[['year_month', 'log_gold']], on='year_month', how='left')
test_merged = test_df[['year_month']].merge(master_df_for_naive[['year_month', 'log_gold']], on='year_month', how='left')

train_naive_pred_log = train_merged['log_gold'].values
val_naive_pred_log = val_merged['log_gold'].values
test_naive_pred_log = test_merged['log_gold'].values
```
**Explanation:**
- **Naïve Baseline**: Predicts next month = current month (no change)
- **Logic**: gold(t+1) = gold(t)
- **Purpose**: Benchmark to compare ML models against
- **Merge**: Gets current month's gold price for each split

**Why Naïve Baseline?**
- Simple benchmark (if model can't beat this, it's useless)
- Measures improvement over "no prediction" approach
- Standard practice in time-series forecasting

#### Lines 209-226: Align Lengths and Evaluate
```python
min_len_train = min(len(train_naive_pred_log), len(y_train_log))
train_naive_pred_log = train_naive_pred_log[:min_len_train]
y_train_log_for_naive = y_train_log[:min_len_train]
# ... (similar for val and test)

naive_train_metrics = evaluate_model(y_train_log_for_naive, train_naive_pred_log, y_train_original_for_naive, "Naïve Baseline", "Train")
naive_val_metrics = evaluate_model(y_val_log_for_naive, val_naive_pred_log, y_val_original_for_naive, "Naïve Baseline", "Validation")
naive_test_metrics = evaluate_model(y_test_log_for_naive, test_naive_pred_log, y_test_original_for_naive, "Naïve Baseline", "Test")
```
**Explanation:**
- Aligns lengths (handles any length mismatches)
- Evaluates naïve baseline on all datasets
- Results stored for comparison

---

### STEP 8: MODEL COMPARISON

#### Lines 235-240: Rank Models by Validation MAE
```python
models_metrics_mae = {
    'XGBoost': xgb_val_metrics['mae'],
    'LightGBM': lgb_val_metrics['mae'],
    'Random Forest': rf_val_metrics['mae'],
    'SVR': svr_val_metrics['mae']
}

sorted_models_mae = sorted(models_metrics_mae.items(), key=lambda x: x[1])
```
**Explanation:**
- Creates dictionary of validation MAE values
- **Sorts by MAE**: Lower MAE = better model
- **key=lambda x: x[1]**: Sorts by second element (MAE value)
- Result: List of (model_name, mae) tuples, sorted by MAE

#### Lines 242-249: Print Model Rankings
```python
print("\n  Models ranked by Validation MAE (original scale, lowest is best):")
for i, (model_name, mae) in enumerate(sorted_models_mae, 1):
    print(f"    {i}. {model_name}: {mae:.6f} USD/gram")

best_model_name = sorted_models_mae[0][0]
best_val_mae = sorted_models_mae[0][1]
print(f"\n  ✓ Best Model: {best_model_name} (Lowest Validation MAE: {best_val_mae:.6f} USD/gram)")
```
**Explanation:**
- Displays ranked models (1 = best, 4 = worst)
- Identifies best model (lowest validation MAE)
- **Why Validation MAE?**: 
  - MAE is interpretable (error in USD/gram)
  - Validation set is used for model selection
  - Prevents overfitting to test set

#### Lines 251-256: Compare ML Models vs Naïve Baseline
```python
print("\n  📊 ML Models vs Naïve Baseline Comparison (Validation MAE):")
print(f"    Naïve Baseline: {naive_val_metrics['mae']:.6f} USD/gram")
for model_name, mae in sorted_models_mae:
    improvement = ((naive_val_metrics['mae'] - mae) / naive_val_metrics['mae']) * 100
    status = "✓ IMPROVES" if improvement > 0 else "✗ WORSE"
    print(f"    {model_name}: {mae:.6f} USD/gram ({improvement:+.2f}% vs baseline) {status}")
```
**Explanation:**
- Compares each ML model against naïve baseline
- **Improvement calculation**: (baseline_mae - model_mae) / baseline_mae × 100
- **Positive improvement**: Model is better than baseline
- **Negative improvement**: Model is worse than baseline (shouldn't happen!)

**Example:**
- If baseline MAE = 10 and model MAE = 7
- Improvement = (10-7)/10 × 100 = 30% improvement

#### Lines 258-267: Print R² Comparison
```python
models_r2 = {
    'XGBoost': xgb_val_metrics['r2'],
    'LightGBM': lgb_val_metrics['r2'],
    'Random Forest': rf_val_metrics['r2'],
    'SVR': svr_val_metrics['r2'],
    'Naïve Baseline': naive_val_metrics['r2']
}
for model_name, r2 in models_r2.items():
    print(f"    {model_name}: {r2:.6f}")
```
**Explanation:**
- Displays R² scores for all models
- R² measures fit quality (higher is better, 0-1 range)
- Helps understand model performance

---

### STEP 9: SAVE MODELS

#### Lines 276-294: Save All Models
```python
with open(models_dir / "xgboost_model.pkl", 'wb') as f:
    pickle.dump(xgb_model, f)
# ... (similar for other models)

with open(models_dir / "scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)

with open(output_dir / "feature_columns.pkl", 'wb') as f:
    pickle.dump(feature_cols, f)
```
**Explanation:**
- **Pickle format**: Saves Python objects (models, scaler)
- **Models saved**: XGBoost, LightGBM, Random Forest, SVR
- **Scaler saved**: Needed for SVR predictions (scaling)
- **Feature columns saved**: Needed for predictions (feature order)

**Why Pickle?**
- Preserves exact model state (weights, parameters)
- Easy to load: `pickle.load()` restores model
- Standard format for Python ML models

---

### STEP 10: SAVE EVALUATION RESULTS

#### Lines 300-320: Create Results DataFrame
```python
results_df = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM', 'Random Forest', 'SVR', 'Naïve Baseline'],
    'Train_RMSE': [xgb_train_metrics['rmse'], lgb_train_metrics['rmse'], ...],
    'Train_MAE': [xgb_train_metrics['mae'], lgb_train_metrics['mae'], ...],
    'Train_R2': [xgb_train_metrics['r2'], lgb_train_metrics['r2'], ...],
    'Val_RMSE': [xgb_val_metrics['rmse'], lgb_val_metrics['rmse'], ...],
    'Val_MAE': [xgb_val_metrics['mae'], lgb_val_metrics['mae'], ...],
    'Val_R2': [xgb_val_metrics['r2'], lgb_val_metrics['r2'], ...],
    'Test_RMSE': [xgb_test_metrics['rmse'], lgb_test_metrics['rmse'], ...],
    'Test_MAE': [xgb_test_metrics['mae'], lgb_test_metrics['mae'], ...],
    'Test_R2': [xgb_test_metrics['r2'], lgb_test_metrics['r2'], ...]
})
```
**Explanation:**
- Creates comprehensive results table
- **Columns**: Model name, metrics for Train/Val/Test
- **Metrics**: RMSE, MAE, R² for each dataset
- **Rows**: One row per model (5 models total)

#### Lines 322-327: Save Results
```python
print("\n  📊 Evaluation Metrics Note:")
print("    - RMSE/MAE: On original scale (USD per gram)")
print("    - R²: On log scale (proper evaluation metric)")

results_df.to_csv(output_dir / "model_evaluation_results.csv", index=False)
print("  ✓ Evaluation results saved to outputs/model_evaluation_results.csv")
```
**Explanation:**
- Saves results to CSV file
- Notes about metric scales (important for interpretation)
- File will be used in app.py (display results on About page)

---

## MODEL COMPARISON SUMMARY

### Models Trained:
1. **XGBoost**: Gradient boosting (usually best performer)
2. **LightGBM**: Fast gradient boosting (similar to XGBoost)
3. **Random Forest**: Ensemble bagging (robust, fast)
4. **SVR**: Support Vector Regression (baseline, non-linear)
5. **Naïve Baseline**: No-change prediction (benchmark)

### Selection Criteria:
- **Best Model**: Lowest Validation MAE (Mean Absolute Error)
- **Why Validation MAE?**: 
  - Interpretable (error in USD/gram)
  - Validation set used for model selection
  - Prevents overfitting to test set

### Evaluation Metrics:
- **RMSE**: Root Mean Squared Error (penalizes large errors)
- **MAE**: Mean Absolute Error (average error)
- **R²**: Coefficient of Determination (fit quality)

### Key Principles:
- **Regularization**: All models use regularized hyperparameters (prevent overfitting)
- **Log Scale**: Training and predictions in log scale (more stable)
- **Back-Transform**: Metrics computed on original scale (more interpretable)
- **Time-Series Split**: Train/Val/Test split preserves temporal order

---

## OUTPUT FILES

### Models (saved in `models/` directory):
1. **xgboost_model.pkl**: Trained XGBoost model
2. **lightgbm_model.pkl**: Trained LightGBM model
3. **random_forest_model.pkl**: Trained Random Forest model
4. **svr_model.pkl**: Trained SVR model
5. **scaler.pkl**: StandardScaler (for SVR)

### Results (saved in `outputs/` directory):
6. **model_evaluation_results.csv**: Comprehensive evaluation results
7. **feature_columns.pkl**: Feature column names (updated)

**Total**: 7 files saved

---

## SUMMARY

Phase 4 accomplishes:

1. **Loads train/val/test splits** from Phase 3
2. **Trains 4 ML models** (XGBoost, LightGBM, Random Forest, SVR)
3. **Uses regularized hyperparameters** (prevents overfitting)
4. **Trains on log scale** (more stable predictions)
5. **Evaluates on original scale** (interpretable metrics)
6. **Compares models** (ranks by validation MAE)
7. **Compares vs naïve baseline** (benchmark)
8. **Saves all models** (for deployment)
9. **Saves evaluation results** (for analysis)

**Key Principles:**
- Regularization (prevent overfitting)
- Log scale training (stability)
- Original scale evaluation (interpretability)
- Validation-based selection (best model)
- Comprehensive evaluation (all metrics, all datasets)

**Output**: Trained models and evaluation results, ready for deployment (app.py)

