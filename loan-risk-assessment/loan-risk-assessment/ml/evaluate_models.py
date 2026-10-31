"""
evaluate_models.py
--------------------
Standalone script to print a human-readable evaluation report from the
artifacts already produced by train_models.py (model_comparison.csv,
tuning_before_after.csv, feature_importance.json, model_meta.json).

Run this any time you just want the report without re-training:
    python3 evaluate_models.py
"""
import json
import os

import pandas as pd

ARTIFACT_DIR = os.path.dirname(__file__)


def main():
    print("=" * 70)
    print("INTELLIGENT LOAN RISK ASSESSMENT — MODEL EVALUATION REPORT")
    print("=" * 70)

    comparison = pd.read_csv(os.path.join(ARTIFACT_DIR, "model_comparison.csv"))
    print("\n--- Model Comparison (test set) ---")
    print(comparison.to_string(index=False))

    with open(os.path.join(ARTIFACT_DIR, "model_meta.json")) as f:
        meta = json.load(f)
    print(f"\n--- Selected Model: {meta['best_model_name']} ---")
    print("Selection rule: highest ROC-AUC, tie-broken by weighted F1-score")
    print("(NOT simply the highest accuracy — accuracy is misleading on")
    print(" imbalanced classes even after SMOTE rebalancing the training set).")

    tuning_path = os.path.join(ARTIFACT_DIR, "tuning_before_after.csv")
    if os.path.exists(tuning_path):
        print("\n--- Hyperparameter Tuning: Before vs After ---")
        print(pd.read_csv(tuning_path).to_string(index=False))

    fi_path = os.path.join(ARTIFACT_DIR, "feature_importance.json")
    if os.path.exists(fi_path):
        with open(fi_path) as f:
            fi = json.load(f)
        print("\n--- Top Contributing Features (Explainability) ---")
        for row in fi[:10]:
            print(f"  {row['feature']:<40s} {row['importance']}")

    print("\nPlots (confusion matrices, ROC curves) saved under ml/plots/")
    print("=" * 70)


if __name__ == "__main__":
    main()
