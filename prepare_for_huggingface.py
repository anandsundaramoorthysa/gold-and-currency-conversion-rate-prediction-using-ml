"""
Prepare the Gold Price Prediction project for Hugging Face Spaces.
Runs the full pipeline (Phase 1 → 4 + analysis) so outputs/, models/, and static/ are ready to push.
Run this once locally (with using_data/ CSVs present), then push the repo to a Docker Space.
"""

import subprocess
import sys
from pathlib import Path

def run_script(name: str, script: str) -> bool:
    """Run a Python script; return True on success."""
    print(f"\n{'='*60}")
    print(f"Running: {script}")
    print('='*60)
    result = subprocess.run([sys.executable, script], cwd=Path(__file__).resolve().parent)
    if result.returncode != 0:
        print(f"ERROR: {script} failed with exit code {result.returncode}")
        return False
    print(f"OK: {script} completed.")
    return True

def main():
    root = Path(__file__).resolve().parent
    scripts = [
        "phase1_data_preprocessing.py",
        "phase2_feature_engineering.py",
        "phase3_train_test_split.py",
        "phase4_model_training.py",
        "generate_analysis_plots.py",
    ]
    for script in scripts:
        path = root / script
        if not path.exists():
            print(f"SKIP: {script} not found.")
            continue
        if not run_script(script, script):
            sys.exit(1)

    # Sanity check
    outputs = root / "outputs"
    models = root / "models"
    required = [
        outputs / "master_df.csv",
        outputs / "features_df.csv",
        outputs / "feature_columns.pkl",
        outputs / "model_evaluation_results.csv",
        outputs / "dataset_summary.csv",
        models / "xgboost_model.pkl",
        models / "lightgbm_model.pkl",
        models / "random_forest_model.pkl",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("\nWARNING: Some required files are missing after pipeline:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    print("\n" + "="*60)
    print("Preparation complete. You can now push this repo to Hugging Face Spaces.")
    print("  - Create a new Space, choose SDK: Docker")
    print("  - Push this folder (including outputs/, models/, templates/, static/)")
    print("="*60)

if __name__ == "__main__":
    main()
