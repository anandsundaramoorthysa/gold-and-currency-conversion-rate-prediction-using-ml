# Gold Price Prediction System - Project Summary

## ✅ Project Completion Status

All phases have been successfully completed according to the requirements.

## 📋 Completed Phases

### ✅ Phase 1: Data Preprocessing
- **Status**: Complete
- **Output**: `outputs/master_df.csv` (612 rows, 12 columns)
- **Date Range**: 1975-01 to 2025-12
- **Key Operations**:
  - Loaded and inspected all 10 CSV datasets
  - Standardized dates and created year_month merge key
  - Converted daily files to monthly (first available trading day)
  - Converted gold from USD/ounce to USD/gram
  - Handled missing values (forward/backward fill)
  - Merged all datasets

### ✅ Phase 2: Feature Engineering
- **Status**: Complete
- **Output**: `outputs/features_df.csv` (597 rows, 30 columns)
- **Features Created**:
  - **Macro Features**: CPI YoY, Real Rate, Yield Spread
  - **Lag Features**: Gold (1,2,3), Fed Funds (1,2), CPI YoY (1,2), Real Rate (1), USD-INR (1,2)
  - **Technical Features**: MA3, MA12, Volatility 3M
  - **Target**: gold_usd_per_gram_next_month

### ✅ Phase 3: Train/Validation/Test Split
- **Status**: Complete
- **Splits**:
  - **Train**: 1977-2020 (528 rows)
  - **Validation**: 2021-2023 (36 rows)
  - **Test**: 2024-2025 (23 rows)
- **Output Files**: X_train.csv, y_train.csv, X_val.csv, y_val.csv, X_test.csv, y_test.csv

### ✅ Phase 4: Model Training
- **Status**: Complete
- **Models Trained**:
  1. **XGBoost Regressor** ⭐ (Best Model - Validation RMSE: 7.254)
  2. LightGBM Regressor (Validation RMSE: 10.835)
  3. Random Forest Regressor (Validation RMSE: 8.780)
  4. Support Vector Regression (Validation RMSE: 21.370)
- **Evaluation Metrics**: RMSE, MAE, R² for Train, Validation, and Test sets
- **Saved Models**: `models/all_models.pkl`, `models/best_model.pkl`

### ✅ Phase 5: Flask Web Application
- **Status**: Complete
- **Routes Implemented**:
  - `/` - Home page with prediction interface
  - `/about` - Comprehensive project documentation
  - `/predict` - API endpoint (POST) for predictions
- **Features**:
  - Interactive prediction UI (month/year input)
  - Displays predicted gold price in USD per gram
  - Shows USD exchange rate equivalences (INR, AUD, CAD, JPY, CHF, GBP)
  - Displays best model name and confidence score
  - About page with full project details, analysis, and visualizations

### ✅ Phase 6: Analysis & Documentation
- **Status**: Complete
- **Generated Analysis**:
  - Correlation analysis (Gold vs CPI YoY, Real Rate, Fed Funds)
  - Feature importance plots for all models
  - Model predictions comparison chart
  - Dataset summary table
  - Model evaluation results

## 📊 Model Performance Summary

| Model | Train RMSE | Val RMSE | Test RMSE | Status |
|-------|------------|----------|-----------|--------|
| **XGBoost** | 0.029 | **7.254** | 29.491 | ⭐ Best |
| Random Forest | 1.344 | 8.780 | 31.616 | ✓ Good |
| LightGBM | 1.373 | 10.835 | 35.311 | ✓ Good |
| SVR | 2.968 | 21.370 | 49.111 | ✓ Acceptable |

**Best Model**: XGBoost (Lowest Validation RMSE)

## 🎯 Key Requirements Met

✅ **Gold price prediction** (USD per gram) - Only ML prediction  
✅ **USD exchange rate display** - Shows equivalence for 6 currencies  
✅ **Macroeconomic indicators** - CPI, Fed Funds, Real Rate used for prediction  
✅ **FX rates for display only** - Not used in prediction model  
✅ **Flask web application** - No Streamlit  
✅ **4 models trained** - XGBoost, LightGBM, Random Forest, SVR  
✅ **Time-series safe split** - Train (1977-2020), Val (2021-2023), Test (2024-2025)  
✅ **Complete about page** - Project details, methodology, analysis, visualizations  

## 📁 Project Structure

```
ML_Project/
├── using_data/              # Input CSV files (10 datasets)
├── outputs/                 # Processed data and results
│   ├── master_df.csv
│   ├── features_df.csv
│   ├── train_df.csv, val_df.csv, test_df.csv
│   ├── X_train.csv, y_train.csv, etc.
│   ├── model_evaluation_results.csv
│   └── model_predictions_comparison.csv
├── models/                  # Saved ML models
│   ├── all_models.pkl
│   └── best_model.pkl
├── static/                  # CSS and plots
│   ├── style.css
│   ├── correlation_analysis.png
│   ├── feature_importance.png
│   └── model_predictions_comparison.png
├── templates/               # HTML templates
│   ├── index.html
│   └── about.html
├── phase1_data_preprocessing.py
├── phase2_feature_engineering.py
├── phase3_train_test_split.py
├── phase4_model_training.py
├── generate_analysis_plots.py
├── app.py                   # Flask application
├── test_app.py              # Verification script
├── requirements.txt
├── README.md
└── PROJECT_SUMMARY.md
```

## 🚀 How to Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run preprocessing** (if not already done):
   ```bash
   python phase1_data_preprocessing.py
   python phase2_feature_engineering.py
   python phase3_train_test_split.py
   python phase4_model_training.py
   python generate_analysis_plots.py
   ```

3. **Start Flask app**:
   ```bash
   python app.py
   ```

4. **Access application**:
   - Home: http://localhost:5000
   - About: http://localhost:5000/about

## 📝 Notes

- All phases executed exactly as per requirements
- Gold prices converted from USD/ounce to USD/gram (÷ 31.1035)
- FX rates displayed in format: "To get 1 USD, how much currency is required"
- Best model (XGBoost) selected based on lowest Validation RMSE
- All visualizations and analysis generated automatically
- Flask app includes error handling and user-friendly interface

## ✨ Ready for Deployment

The project is complete and ready for:
- Local testing and demonstration
- Deployment to cloud platforms (Render, Heroku, etc.)
- GitHub repository submission
- Academic presentation

---

**Project Status**: ✅ **COMPLETE**  
**Last Updated**: 2025  
**Best Model**: XGBoost Regressor

