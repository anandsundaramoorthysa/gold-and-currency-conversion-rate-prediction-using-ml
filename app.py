"""
FLASK WEB APPLICATION
Gold Price Prediction System
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

models_dir = Path("models")
outputs_dir = Path("outputs")

print("Loading models and data...")

print("  Loading XGBoost model...")
with open(models_dir / "xgboost_model.pkl", 'rb') as f:
    xgb_model = pickle.load(f)

print("  Loading LightGBM model...")
with open(models_dir / "lightgbm_model.pkl", 'rb') as f:
    lgb_model = pickle.load(f)

print("  Loading Random Forest model...")
with open(models_dir / "random_forest_model.pkl", 'rb') as f:
    rf_model = pickle.load(f)

print("  Loading SVR model...")
with open(models_dir / "svr_model.pkl", 'rb') as f:
    svr_model = pickle.load(f)

print("  Loading scaler...")
with open(models_dir / "scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

with open(outputs_dir / "feature_columns.pkl", 'rb') as f:
    feature_cols = pickle.load(f)

eval_results = pd.read_csv(outputs_dir / "model_evaluation_results.csv")

ml_eval_results = eval_results[eval_results['Model'] != 'Naïve Baseline'].copy()

best_model_row = ml_eval_results.loc[ml_eval_results['Val_MAE'].idxmin()]
best_model_name = str(best_model_row['Model'])
best_model_val_r2_internal = float(best_model_row['Val_R2'])
best_model_accuracy = max(0.0, best_model_val_r2_internal)
best_model_val_mae = float(best_model_row['Val_MAE'])
best_model_val_rmse = float(best_model_row['Val_RMSE'])

models_dict = {
    'xgboost': xgb_model,
    'lightgbm': lgb_model,
    'random_forest': rf_model,
    'svr': svr_model
}

master_df = pd.read_csv(outputs_dir / "master_df.csv")
master_df['year_month'] = pd.to_datetime(master_df['year_month'])
if hasattr(master_df['year_month'].dtype, 'freq'):
    master_df['year_month'] = master_df['year_month'].dt.to_timestamp()

features_df = pd.read_csv(outputs_dir / "features_df.csv")
features_df['year_month'] = pd.to_datetime(features_df['year_month'])
if hasattr(features_df['year_month'].dtype, 'freq'):
    features_df['year_month'] = features_df['year_month'].dt.to_timestamp()

dataset_summary = pd.read_csv(outputs_dir / "dataset_summary.csv")

print(f"✓ Models and data loaded")
print(f"  Best Model: {best_model_name} (Val MAE: {best_model_val_mae:.4f}, Val R²: {best_model_val_r2_internal:.4f})")


def compute_features_from_gold_history(gold_history, macro_data_row):
    """
    Compute features dynamically from a gold price history and macro data.
    This ensures features are updated with predicted values during recursive forecasting.
    
    Args:
        gold_history: List of gold prices (most recent last) - needs at least 3 values for lags
        macro_data_row: Series with macro variables (fedfunds, cpi, cpi_yoy, real_rate, yield_spread, treasury_10y)
    
    Returns:
        Dictionary of computed features
    """
    if len(gold_history) == 0:
        raise ValueError("gold_history cannot be empty")
    
    gold_current = gold_history[-1]
    gold_lag1 = gold_history[-2] if len(gold_history) >= 2 else gold_current
    gold_lag3 = gold_history[-4] if len(gold_history) >= 4 else gold_lag1
    
    log_gold = np.log(gold_current)
    log_gold_lag1 = np.log(gold_lag1)
    gold_return_1m = log_gold - log_gold_lag1
    
    ma3_window = gold_history[-3:] if len(gold_history) >= 3 else gold_history
    gold_ma3 = np.mean(ma3_window)
    
    real_rate = macro_data_row.get('real_rate', np.nan)
    if pd.isna(real_rate):
        fedfunds = macro_data_row.get('fedfunds', np.nan)
        cpi_yoy = macro_data_row.get('cpi_yoy', np.nan)
        real_rate = fedfunds - cpi_yoy if not pd.isna(fedfunds) and not pd.isna(cpi_yoy) else np.nan
    
    fedfunds_lag1 = macro_data_row.get('fedfunds', np.nan)
    cpi_yoy_lag1 = macro_data_row.get('cpi_yoy', np.nan)
    real_rate_lag1 = macro_data_row.get('real_rate', np.nan)
    if pd.isna(real_rate_lag1):
        real_rate_lag1 = real_rate
    
    yield_spread = macro_data_row.get('yield_spread', np.nan)
    if pd.isna(yield_spread):
        treasury_10y = macro_data_row.get('treasury_10y', np.nan)
        fedfunds = macro_data_row.get('fedfunds', np.nan)
        yield_spread = treasury_10y - fedfunds if not pd.isna(treasury_10y) and not pd.isna(fedfunds) else np.nan
    
    computed_features = {
        'gold_lag1': gold_lag1,
        'gold_lag3': gold_lag3,
        'gold_return_1m': gold_return_1m,
        'gold_ma3': gold_ma3,
        'real_rate': real_rate,
        'real_rate_lag1': real_rate_lag1,
        'fedfunds_lag1': fedfunds_lag1,
        'cpi_yoy_lag1': cpi_yoy_lag1,
        'yield_spread': yield_spread
    }
    
    return computed_features


def get_features_for_recursive_prediction(target_month_str):
    """
    Get features for prediction using recursive forecasting approach.
    For future months, builds features dynamically using predicted values from previous months.
    
    CRITICAL FIX: Implements recursive forecasting where:
    1. Start from last historical data
    2. Predict each intermediate month up to target
    3. Update features using predictions
    4. Return features for target month
    """
    target_month = pd.to_datetime(target_month_str)
    if isinstance(target_month, (pd.Series, np.ndarray)):
        target_month = pd.Timestamp(target_month.iloc[0] if isinstance(target_month, pd.Series) else target_month[0])
    elif not isinstance(target_month, pd.Timestamp):
        target_month = pd.Timestamp(target_month)
    
    assert 'log_gold_next' not in feature_cols, "LEAKAGE: target column 'log_gold_next' in features!"
    assert 'gold_usd_per_gram_next_month' not in feature_cols, "LEAKAGE: target column 'gold_usd_per_gram_next_month' in features!"
    
    features_df_local = features_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(features_df_local['year_month']):
        features_df_local['year_month'] = pd.to_datetime(features_df_local['year_month'])
    historical_data = features_df_local[features_df_local['year_month'] < target_month].copy()
    if len(historical_data) == 0:
        return None, None
    
    last_historical_row = historical_data.iloc[-1]
    last_historical_month_val = last_historical_row['year_month']
    if isinstance(last_historical_month_val, pd.Series):
        last_historical_month_val = last_historical_month_val.iloc[0]
    last_historical_month = pd.to_datetime(last_historical_month_val)
    
    gold_history = []
    for i in range(min(4, len(historical_data))):
        row_idx = len(historical_data) - 1 - i
        if row_idx >= 0:
            gold_val = historical_data.iloc[row_idx].get('gold_usd_per_gram', np.nan)
            if not pd.isna(gold_val):
                gold_history.insert(0, gold_val)
    
    if len(gold_history) == 0:
        master_historical = master_df[master_df['year_month'] < target_month].copy()
        if len(master_historical) >= 1:
            for i in range(min(4, len(master_historical))):
                row_idx = len(master_historical) - 1 - i
                if row_idx >= 0:
                    gold_val = master_historical.iloc[row_idx].get('gold_usd_per_gram', np.nan)
                    if not pd.isna(gold_val):
                        gold_history.insert(0, gold_val)
    
    if len(gold_history) == 0:
        return None, None
    
    if isinstance(last_historical_month, (pd.Series, np.ndarray)):
        last_historical_month = pd.Timestamp(last_historical_month.iloc[0] if isinstance(last_historical_month, pd.Series) else last_historical_month[0])
    elif not isinstance(last_historical_month, pd.Timestamp):
        last_historical_month = pd.Timestamp(last_historical_month)
    
    master_df_for_filter = master_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(master_df_for_filter['year_month']):
        master_df_for_filter['year_month'] = pd.to_datetime(master_df_for_filter['year_month'])
    
    comparison_mask = master_df_for_filter['year_month'] <= last_historical_month
    master_filtered = master_df_for_filter[comparison_mask]
    last_master_row = master_filtered.iloc[-1] if len(master_filtered) > 0 else master_df.iloc[-1]
    
    if not isinstance(target_month, pd.Timestamp):
        if isinstance(target_month, (pd.Series, np.ndarray, list)):
            target_month = pd.Timestamp(target_month.iloc[0] if isinstance(target_month, pd.Series) else target_month[0])
        else:
            target_month = pd.Timestamp(target_month)
    
    assert isinstance(target_month, pd.Timestamp), f"target_month must be Timestamp, got {type(target_month)}"
    assert isinstance(last_historical_month, pd.Timestamp), f"last_historical_month must be Timestamp, got {type(last_historical_month)}"
    
    if target_month <= last_historical_month:
        feature_values = last_historical_row[feature_cols].values.reshape(1, -1)
        return feature_values, last_historical_row
    
    current_month = last_historical_month
    
    model_key_map = {
        'XGBoost': 'xgboost',
        'LightGBM': 'lightgbm',
        'Random Forest': 'random_forest',
        'SVR': 'svr'
    }
    best_model_key = model_key_map.get(best_model_name, 'xgboost')
    best_model_for_recursive = models_dict[best_model_key]
    
    while current_month < target_month:
        computed_features = compute_features_from_gold_history(gold_history, last_master_row)
        
        current_features_dict = {}
        for col in feature_cols:
            if col in computed_features:
                current_features_dict[col] = computed_features[col]
            elif col in last_historical_row.index:
                current_features_dict[col] = last_historical_row[col]
            else:
                current_features_dict[col] = np.nan
        
        feature_vector = np.array([current_features_dict[col] for col in feature_cols]).reshape(1, -1)
        
        if np.isnan(feature_vector).any():
            last_historical_features = last_historical_row[feature_cols].values.reshape(1, -1)
            nan_mask = np.isnan(feature_vector[0])
            feature_vector[0][nan_mask] = last_historical_features[0][nan_mask]
        
        if best_model_key == 'svr':
            log_pred = best_model_for_recursive.predict(scaler.transform(feature_vector))[0]
        else:
            log_pred = best_model_for_recursive.predict(feature_vector)[0]
        
        log_last_gold = np.log(gold_history[-1])
        log_change = log_pred - log_last_gold
        log_change_clipped = np.clip(log_change, np.log(0.95), np.log(1.10))
        predicted_gold = gold_history[-1] * np.exp(log_change_clipped)
        
        gold_history.append(predicted_gold)
        if len(gold_history) > 4:
            gold_history = gold_history[-4:]
        
        if current_month.month == 12:
            current_month = pd.Timestamp(year=current_month.year + 1, month=1, day=1)
        else:
            current_month = pd.Timestamp(year=current_month.year, month=current_month.month + 1, day=1)
    
    final_computed_features = compute_features_from_gold_history(gold_history, last_master_row)
    
    final_features_dict = {}
    for col in feature_cols:
        if col in final_computed_features:
            final_features_dict[col] = final_computed_features[col]
        elif col in last_historical_row.index:
            final_features_dict[col] = last_historical_row[col]
        else:
            final_features_dict[col] = np.nan
    
    feature_values = np.array([final_features_dict[col] for col in feature_cols]).reshape(1, -1)
    
    if np.isnan(feature_values).any():
        last_historical_features = last_historical_row[feature_cols].values.reshape(1, -1)
        nan_mask = np.isnan(feature_values[0])
        feature_values[0][nan_mask] = last_historical_features[0][nan_mask]
    
    final_row = pd.Series(final_features_dict)
    
    return feature_values, final_row


def get_fx_rates_for_month(target_month_str):
    """Get FX rates for a given month (latest available)"""
    target_month = pd.to_datetime(target_month_str)
    
    available_data = master_df[master_df['year_month'] <= target_month].copy()
    
    if len(available_data) == 0:
        return None
    
    latest_row = available_data.iloc[-1]
    
    fx_rates = {
        'INR': latest_row.get('usd_inr', np.nan),
        'AUD': latest_row.get('usd_aud', np.nan),
        'CAD': latest_row.get('usd_cad', np.nan),
        'JPY': latest_row.get('usd_jpy', np.nan),
        'CHF': latest_row.get('usd_chf', np.nan),
        'GBP': latest_row.get('usd_gbp', np.nan)
    }
    
    return fx_rates


def calculate_confidence_score(gold_price, val_rmse):
    """
    Calculate confidence score based on validation RMSE.
    Confidence = max(0, 100 - (val_rmse / gold_price) * 100)
    Higher confidence when RMSE is small relative to price.
    """
    if gold_price > 0:
        relative_error = val_rmse / gold_price
        confidence = max(0, min(100, (1 - relative_error) * 100))
    else:
        confidence = 50.0
    
    return round(float(confidence), 2)


@app.route('/')
def index():
    """Home page with prediction UI"""
    return render_template('index.html', 
                         best_model=best_model_name,
                         best_model_accuracy=best_model_accuracy,
                         best_model_rmse=best_model_val_rmse)


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        month_year = data.get('month_year')
        
        if not month_year:
            return jsonify({'error': 'Month and year required'}), 400
        
        try:
            target_date = pd.to_datetime(month_year + '-01')
            if isinstance(target_date, (pd.Series, np.ndarray)):
                target_date = pd.Timestamp(target_date.iloc[0] if isinstance(target_date, pd.Series) else target_date[0])
            elif not isinstance(target_date, pd.Timestamp):
                target_date = pd.Timestamp(target_date)
        except Exception as e:
            return jsonify({'error': f'Invalid date format. Use YYYY-MM. Error: {str(e)}'}), 400
        
        try:
            feature_values, latest_row = get_features_for_recursive_prediction(month_year)
        except Exception as e:
            import traceback
            error_msg = f"Error in feature computation: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return jsonify({'error': f'Error computing features: {str(e)}'}), 500
        
        if feature_values is None:
            return jsonify({'error': 'Insufficient historical data for prediction'}), 400
        
        if np.isnan(feature_values).any():
            historical_data = features_df[features_df['year_month'] < pd.to_datetime(month_year + '-01')].copy()
            if len(historical_data) > 0:
                last_historical_features = historical_data.iloc[-1][feature_cols].values.reshape(1, -1)
                nan_mask = np.isnan(feature_values[0])
                feature_values[0][nan_mask] = last_historical_features[0][nan_mask]
        
        predictions_log = {}
        
        xgb_pred_log = float(models_dict['xgboost'].predict(feature_values)[0])
        predictions_log['XGBoost'] = xgb_pred_log
        
        lgb_pred_log = float(models_dict['lightgbm'].predict(feature_values)[0])
        predictions_log['LightGBM'] = lgb_pred_log
        
        rf_pred_log = float(models_dict['random_forest'].predict(feature_values)[0])
        predictions_log['Random Forest'] = rf_pred_log
        
        svr_pred_log = float(models_dict['svr'].predict(scaler.transform(feature_values))[0])
        predictions_log['SVR'] = svr_pred_log
        
        target_date = pd.to_datetime(month_year + '-01')
        available_master_data = master_df[master_df['year_month'] < target_date]
        
        if len(available_master_data) == 0:
            last_gold_price = 70.0
        else:
            last_master_row = available_master_data.iloc[-1]
            last_gold_price = float(last_master_row['gold_usd_per_gram'])
        
        log_last_gold = np.log(last_gold_price)
        
        if best_model_name == 'XGBoost':
            main_prediction_log = predictions_log['XGBoost']
        elif best_model_name == 'LightGBM':
            main_prediction_log = predictions_log['LightGBM']
        elif best_model_name == 'Random Forest':
            main_prediction_log = predictions_log['Random Forest']
        else:
            main_prediction_log = predictions_log['SVR']
        
        log_change = main_prediction_log - log_last_gold
        log_change_clipped = np.clip(log_change, np.log(0.95), np.log(1.10))
        main_prediction = last_gold_price * np.exp(log_change_clipped)
        
        predictions_original = {}
        for model_name, log_pred in predictions_log.items():
            log_change_model = log_pred - log_last_gold
            log_change_model_clipped = np.clip(log_change_model, np.log(0.95), np.log(1.10))
            predictions_original[model_name] = last_gold_price * np.exp(log_change_model_clipped)
        
        all_preds = [main_prediction] + list(predictions_original.values())
        pred_range = max(all_preds) - min(all_preds)
        pred_std = np.std(all_preds)
        
        min_variation = max(0.1, last_gold_price * 0.001)
        if pred_range < min_variation and pred_std < min_variation * 0.5:
            import warnings
            warnings.warn(f"WARNING: Predictions are flat (range={pred_range:.4f}, std={pred_std:.4f}). "
                         f"This may indicate features are not updating correctly. "
                         f"Features used: {dict(zip(feature_cols, feature_values[0]))}")
        
        macro_features = ['real_rate', 'real_rate_lag1', 'fedfunds_lag1', 'cpi_yoy_lag1', 'yield_spread']
        macro_values = []
        for f in macro_features:
            if f in feature_cols:
                idx = feature_cols.index(f)
                val = feature_values[0][idx]
                if isinstance(val, (np.ndarray, pd.Series)):
                    val = val.item() if hasattr(val, 'item') else (val.iloc[0] if isinstance(val, pd.Series) else val[0] if len(val) > 0 else np.nan)
                macro_values.append(float(val) if not pd.isna(val) else np.nan)
            else:
                macro_values.append(np.nan)
        
        non_nan_values = [v for v in macro_values if not pd.isna(v)]
        if len(non_nan_values) > 0:
            first_val = non_nan_values[0]
            if all(abs(float(v) - float(first_val)) < 1e-6 for v in non_nan_values):
                import warnings
                warnings.warn(f"WARNING: Macro features may not be influencing predictions. "
                             f"Macro values: {dict(zip(macro_features, macro_values))}")
        
        confidence = calculate_confidence_score(main_prediction, best_model_val_rmse)
        
        fx_rates = get_fx_rates_for_month(month_year)
        
        if fx_rates is None:
            fx_rates = {
                'INR': np.nan, 'AUD': np.nan, 'CAD': np.nan,
                'JPY': np.nan, 'CHF': np.nan, 'GBP': np.nan
            }
        
        return jsonify({
            'prediction': round(float(main_prediction), 4),
            'confidence': float(confidence),
            'best_model': str(best_model_name),
            'fx_rates': {k: round(float(v), 4) if not np.isnan(v) else None for k, v in fx_rates.items()},
            'all_predictions': {k: round(float(v), 4) for k, v in predictions_original.items()}
        })
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"ERROR in /predict route: {str(e)}")
        print(f"Traceback:\n{error_traceback}")
        return jsonify({'error': f'Prediction failed: {str(e)}. Check server logs for details.'}), 500


@app.route('/about')
def about():
    """About page with full project details"""
    model_comparison_raw = eval_results.to_dict('records')
    model_comparison = []
    for row in model_comparison_raw:
        row_display = row.copy()
        for r2_col in ['Train_R2', 'Val_R2', 'Test_R2']:
            if r2_col in row_display:
                row_display[r2_col] = max(0.0, float(row_display[r2_col])) if row_display[r2_col] is not None else 0.0
        model_comparison.append(row_display)
    
    master_df_for_corr = master_df.copy()
    
    if 'log_gold' not in master_df_for_corr.columns:
        master_df_for_corr['log_gold'] = np.log(master_df_for_corr['gold_usd_per_gram'])
    
    if 'cpi_yoy' not in master_df_for_corr.columns:
        master_df_for_corr['cpi_yoy'] = (master_df_for_corr['cpi'] / master_df_for_corr['cpi'].shift(12) - 1) * 100
    
    if 'real_rate' not in master_df_for_corr.columns:
        master_df_for_corr['real_rate'] = master_df_for_corr['fedfunds'] - master_df_for_corr['cpi_yoy']
    
    corr_cols = ['log_gold', 'cpi_yoy', 'real_rate', 'fedfunds']
    gold_col_for_corr = 'log_gold'
    
    corr_data = master_df_for_corr[corr_cols].corr()
    
    xgb_model = models_dict['xgboost']
    importances = xgb_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    top_features_xgb = [{'feature': feature_cols[i], 'importance': float(importances[i])} 
                        for i in indices]
    
    lgb_model = models_dict['lightgbm']
    lgb_importances = lgb_model.feature_importances_
    lgb_indices = np.argsort(lgb_importances)[::-1][:10]
    top_features_lgb = [{'feature': feature_cols[i], 'importance': float(lgb_importances[i])} 
                        for i in lgb_indices]
    
    rf_model = models_dict['random_forest']
    rf_importances = rf_model.feature_importances_
    rf_indices = np.argsort(rf_importances)[::-1][:10]
    top_features_rf = [{'feature': feature_cols[i], 'importance': float(rf_importances[i])} 
                       for i in rf_indices]
    
    X_test = pd.read_csv(outputs_dir / "X_test.csv")
    y_test_log = pd.read_csv(outputs_dir / "y_test.csv").iloc[:, 0]
    test_df = pd.read_csv(outputs_dir / "test_df.csv")
    
    predictions_log = {}
    try:
        predictions_log['XGBoost'] = models_dict['xgboost'].predict(X_test)
    except Exception as e:
        predictions_log['XGBoost'] = np.full(len(X_test), np.nan)
    
    try:
        predictions_log['LightGBM'] = models_dict['lightgbm'].predict(X_test)
    except Exception as e:
        predictions_log['LightGBM'] = np.full(len(X_test), np.nan)
    
    try:
        predictions_log['Random Forest'] = models_dict['random_forest'].predict(X_test)
    except Exception as e:
        predictions_log['Random Forest'] = np.full(len(X_test), np.nan)
    
    try:
        predictions_log['SVR'] = models_dict['svr'].predict(scaler.transform(X_test))
    except Exception as e:
        predictions_log['SVR'] = np.full(len(X_test), np.nan)
    
    y_test_original = np.exp(y_test_log.values)
    predictions_original = {}
    for k, v in predictions_log.items():
        if isinstance(v, np.ndarray) and not np.isnan(v).all():
            predictions_original[k] = np.exp(v)
        else:
            predictions_original[k] = np.full(len(X_test), np.nan)
    
    test_dates = pd.to_datetime(test_df['year_month']) if 'year_month' in test_df.columns else pd.date_range(start='2020-01', periods=len(X_test), freq='MS')
    comparison_df = pd.DataFrame({
        'Date': test_dates,
        'Actual': y_test_original,
        **predictions_original
    })
    
    comparison_df['Date'] = comparison_df['Date'].dt.strftime('%Y-%m-%d')
    predictions_data = comparison_df.to_dict('records')
    
    dataset_summary_data = dataset_summary.to_dict('records')
    
    correlations = {
        'gold_cpi_yoy': float(corr_data.loc[gold_col_for_corr, 'cpi_yoy']),
        'gold_real_rate': float(corr_data.loc[gold_col_for_corr, 'real_rate']),
        'gold_fedfunds': float(corr_data.loc[gold_col_for_corr, 'fedfunds'])
    }
    
    return render_template('about.html',
                         model_comparison=model_comparison,
                         correlations=correlations,
                         top_features_xgb=top_features_xgb,
                         top_features_lgb=top_features_lgb,
                         top_features_rf=top_features_rf,
                         predictions_data=predictions_data,
                         dataset_summary_data=dataset_summary_data,
                         best_model=best_model_name,
                         best_model_accuracy=best_model_accuracy,
                         best_model_mae=best_model_val_mae,
                         best_model_rmse=best_model_val_rmse)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

