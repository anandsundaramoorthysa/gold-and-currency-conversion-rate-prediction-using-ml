---
title: Gold Price Prediction
emoji: 🏆
colorFrom: amber
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Gold Price Prediction System

Predict **gold price per gram (USD)** using macroeconomic indicators and ML. Built with Flask and tree models (XGBoost, LightGBM, Random Forest).

## Features

- **Predict** gold price for any future month (YYYY-MM)
- **Best model** auto-selected by validation MAE (XGBoost / LightGBM / Random Forest)
- **FX display**: USD equivalents in INR, AUD, CAD, JPY, CHF, GBP
- **About** page: model comparison, feature importance, correlations, test-period charts

## How to use

1. Open the **Home** tab and choose a **Month & Year** (e.g. `2025-12`).
2. Click **Predict** to get the gold price (USD/gram), confidence, and FX rates.
3. Visit **About** for methodology, metrics, and plots.

## API

```bash
curl -X POST https://YOUR-SPACE-URL/predict \
  -H "Content-Type: application/json" \
  -d '{"month_year": "2025-12"}'
```

Response includes `prediction`, `confidence`, `best_model`, `fx_rates`, `all_predictions`.

## Models

- Target: next-month **log return**; price = current_gold × exp(return).
- Features: gold returns (1m/3m/6m/12m), volatility, MA ratio, real rate, yield spread, macro lags.
- Train: 1977–2019; Validation: 2020–2022; Test: 2023–2025.

---

*College ML Project — Flask + XGBoost/LightGBM/Random Forest*
