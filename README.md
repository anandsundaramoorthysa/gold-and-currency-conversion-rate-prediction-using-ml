# Gold Price Prediction System

A machine learning system that predicts gold price per gram in USD using macroeconomic indicators and time-series features. Deployed as a Flask web application.

## 🎯 Project Overview

This system:
- **Predicts** gold price per gram in USD (monthly time-series)
- **Displays** USD exchange rate equivalents (INR, AUD, CAD, JPY, CHF, GBP)
- Uses **macroeconomic indicators** (CPI, Fed Funds, Real Rate) for prediction
- Uses **FX rates only for display**, not prediction
- Deployed as a **Flask web application** (NO Streamlit)

## 📊 Datasets Used

1. Consumer Price Index (CPI) - Monthly
2. Federal Funds Rate (FEDFUNDS) - Monthly
3. Foreign Exchange Rates - Daily/Monthly:
   - USD-INR (Monthly)
   - USD-AUD, USD-CAD, USD-JPY, USD-CHF, USD-GBP (Daily → Monthly)
4. Gold Futures Historical Data (1975-2025) - Daily → Monthly
5. US 10-Year Treasury Yield - Daily → Monthly

## 🔧 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ML_Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Running the Application

### Step 1: Data Preprocessing
```bash
python phase1_data_preprocessing.py
```

### Step 2: Feature Engineering
```bash
python phase2_feature_engineering.py
```

### Step 3: Train/Test Split
```bash
python phase3_train_test_split.py
```

### Step 4: Model Training
```bash
python phase4_model_training.py
```

### Step 5: Generate Analysis Plots
```bash
python generate_analysis_plots.py
```

### Step 6: Run Flask Application
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## 📁 Project Structure

```
ML_Project/
├── using_data/              # Input CSV files
├── outputs/                 # Processed data and results
│   ├── master_df.csv
│   ├── features_df.csv
│   ├── X_train.csv, y_train.csv
│   ├── X_val.csv, y_val.csv
│   ├── X_test.csv, y_test.csv
│   └── model_evaluation_results.csv
├── models/                  # Saved models
│   ├── all_models.pkl
│   └── best_model.pkl
├── static/                  # Static files (CSS, plots)
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
├── requirements.txt
└── README.md
```

## 🤖 Models Trained

1. **XGBoost Regressor** (Best Model - Lowest Validation RMSE)
2. LightGBM Regressor
3. Random Forest Regressor
4. Support Vector Regression (RBF kernel)

## 📈 Features

### Macro Features
- CPI YoY Inflation
- Real Interest Rate
- Yield Spread

### Lag Features
- Gold (1, 2, 3 months)
- Fed Funds (1, 2 months)
- CPI YoY (1, 2 months)
- Real Rate (1 month)
- USD-INR (1, 2 months)

### Technical Features
- Moving averages (3, 12 months)
- Volatility (3 months)

## 🌐 Web Application Routes

- `/` - Home page with prediction interface
- `/about` - Comprehensive project documentation and analysis
- `/predict` - API endpoint for predictions (POST request)

## 📝 API Usage

```python
import requests

response = requests.post('http://localhost:5000/predict', 
    json={'month_year': '2025-12'})
data = response.json()

print(f"Predicted Gold Price: ${data['prediction']}")
print(f"Best Model: {data['best_model']}")
print(f"Confidence: {data['confidence']}%")
print(f"FX Rates: {data['fx_rates']}")
```

## 📊 Evaluation Metrics

Models are evaluated using:
- **RMSE** (Root Mean Squared Error) - Primary metric
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)

Best model selected based on **lowest Validation RMSE**.

## 🎓 Methodology

1. **Data Preprocessing**: Standardize dates, handle frequencies, merge datasets
2. **Feature Engineering**: Create macro, lag, and technical features
3. **Time-Series Split**: Train (1977-2020), Validation (2021-2023), Test (2024-2025)
4. **Model Training**: Train 4 different models and evaluate
5. **Deployment**: Flask web application with interactive UI

## 📄 License

This project is for educational purposes.

## 👤 Author

College ML Project

