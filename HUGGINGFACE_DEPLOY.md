# Deploy on Hugging Face Spaces

Follow these steps to host the Gold Price Prediction app on Hugging Face.

## 1. Prepare assets (run once locally)

You need the **data** in `using_data/` (all CSV files) and then run the full pipeline so `outputs/`, `models/`, and `static/` are built:

```bash
# From project root (e.g. ML_Project/)
python prepare_for_huggingface.py
```

This runs, in order:

- `phase1_data_preprocessing.py`
- `phase2_feature_engineering.py`
- `phase3_train_test_split.py`
- `phase4_model_training.py`
- `generate_analysis_plots.py`

After it finishes, the repo will have:

- `outputs/` — master_df, features, splits, feature_columns.pkl, evaluation results, dataset_summary
- `models/` — xgboost, lightgbm, random_forest pickle files
- `static/` — correlation and feature importance plots, model comparison plot, style.css

## 2. Create a Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces).
2. Click **Create new Space**.
3. Set:
   - **Name**: e.g. `gold-price-prediction`
   - **SDK**: **Docker**
   - **Visibility**: Public or Private
4. Create the Space.

## 3. Push your code to the Space

Push everything **except** raw data (you can omit `using_data/` if you don’t want it on HF):

- `app.py`
- `templates/`
- `static/`
- `outputs/`
- `models/`
- `requirements.txt`
- `Dockerfile`
- `SPACE_README.md` — use this as the **Space README** (copy content into the Space’s README.md so the Space card shows the right title and `sdk: docker`).

Optional but useful:

- `prepare_for_huggingface.py`
- `HUGGINGFACE_DEPLOY.md` (this file)

Using Git:

```bash
# Clone your Space (replace with your Space URL)
git clone https://huggingface.co/spaces/YOUR_USERNAME/gold-price-prediction
cd gold-price-prediction

# Copy app files + Dockerfile + requirements + SPACE_README content
# Then add, commit, push:
git add app.py templates/ static/ outputs/ models/ requirements.txt Dockerfile README.md
git commit -m "Add Gold Price Prediction app"
git push
```

**Important:** The Space’s `README.md` should contain the **YAML front matter** and content from `SPACE_README.md` (including `sdk: docker` and `app_port: 7860`).

## 4. Run the Space

After the push, Hugging Face will build the Docker image and run:

```text
gunicorn -w 2 -b 0.0.0.0:7860 --timeout 120 app:app
```

The app will be at:  
`https://YOUR_USERNAME-gold-price-prediction.hf.space` (or the URL shown on the Space page).

## Checklist

- [ ] `using_data/` CSVs present locally
- [ ] `python prepare_for_huggingface.py` completed without errors
- [ ] `outputs/`, `models/`, `static/` exist and are committed (or uploaded)
- [ ] Space created with **SDK: Docker**
- [ ] `Dockerfile`, `requirements.txt`, `app.py`, `templates/`, `static/`, `outputs/`, `models/` in the Space repo
- [ ] Space `README.md` contains `sdk: docker` and `app_port: 7860` (from `SPACE_README.md`)

## Troubleshooting

- **Build fails:** Check that `outputs/` and `models/` are in the repo and not empty.
- **App crashes at startup:** Ensure `outputs/dataset_summary.csv` exists (created by `generate_analysis_plots.py`).
- **502 / timeout:** Increase `--timeout` in the Dockerfile `CMD` if predictions are slow.
