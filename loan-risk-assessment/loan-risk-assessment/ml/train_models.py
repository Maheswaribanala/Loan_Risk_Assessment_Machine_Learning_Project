"""
train_models.py
-----------------
End-to-end training pipeline:
 1. Load dataset
 2. Clean (duplicates, outliers)
 3. Feature engineering
 4. Train/test split
 5. Preprocess (impute/encode/scale) via ColumnTransformer
 6. Balance classes with SMOTE (train set only)
 7. Train Logistic Regression, Random Forest, Gradient Boosting
    ("XGBoost slot" — see note below)
 8. Evaluate with Accuracy/Precision/Recall/F1/ROC-AUC/Confusion Matrix/CV
 9. Hyperparameter-tune the best model with RandomizedSearchCV
10. Save the winning pipeline (preprocessor + model) with joblib
11. Save metrics/model_comparison.csv + confusion matrices + ROC curves
    for the dashboard

NOTE ON XGBOOST:
`xgboost` could not be installed in this offline sandbox (no internet
access, no cached wheel available). `GradientBoostingClassifier` from
scikit-learn is used in its place — it is trained, tuned and evaluated
exactly like the other two models and plays the "XGBoost" role in the
comparison table. On a machine with internet access, install xgboost
(`pip install xgboost`) and swap in `from xgboost import XGBClassifier`
in the MODELS dict below — no other code changes are required because
XGBClassifier implements the same scikit-learn estimator interface.
"""

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelBinarizer

sys.path.append(os.path.dirname(__file__))
from feature_engineering import engineer_features, select_model_columns
from preprocessing import build_preprocessor, clean_raw_data
from smote_custom import SMOTE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "loan_dataset.csv")
ARTIFACT_DIR = os.path.dirname(__file__)
PLOTS_DIR = os.path.join(ARTIFACT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

RANDOM_STATE = 42
CLASS_ORDER = ["Low", "Medium", "High"]  # fixed label order used everywhere


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df = clean_raw_data(df)
    df = engineer_features(df)

    y = df["risk_category"]
    X = select_model_columns(df)
    return X, y


def evaluate_model(name, model, X_test, y_test, classes):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    lb = LabelBinarizer()
    lb.fit(classes)
    y_test_bin = lb.transform(y_test)
    try:
        roc_auc = roc_auc_score(y_test_bin, y_proba, average="weighted", multi_class="ovr")
    except ValueError:
        roc_auc = float("nan")

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=classes, ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"confusion_matrix_{name.replace(' ', '_')}.png"))
    plt.close(fig)

    # ROC curve (one-vs-rest, per class)
    fig, ax = plt.subplots(figsize=(5, 4))
    for i, cls in enumerate(lb.classes_):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, list(model.classes_).index(cls)])
        ax.plot(fpr, tpr, label=f"{cls}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve (OvR) — {name}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"roc_curve_{name.replace(' ', '_')}.png"))
    plt.close(fig)

    return {
        "model": name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc == roc_auc else None,
    }


def main():
    print("Loading & preparing data...")
    X, y = load_and_prepare()
    print("Feature matrix:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    # ColumnTransformer may return sparse matrices (OneHotEncoder) — densify
    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()
        X_test_proc = X_test_proc.toarray()

    print("Before SMOTE:", pd.Series(y_train).value_counts().to_dict())
    smote = SMOTE(k_neighbors=5, random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train.to_numpy())
    print("After SMOTE:", pd.Series(y_train_bal).value_counts().to_dict())

    MODELS = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        # Plays the "XGBoost" role — see module docstring for why.
        "XGBoost": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    results = []
    fitted_models = {}
    cv_scores = {}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, model in MODELS.items():
        print(f"\nTraining {name} ...")
        t0 = time.time()
        model.fit(X_train_bal, y_train_bal)
        fitted_models[name] = model

        cv = cross_val_score(model, X_train_bal, y_train_bal, cv=skf, scoring="f1_weighted")
        cv_scores[name] = round(cv.mean(), 4)

        metrics = evaluate_model(name, model, X_test_proc, y_test, CLASS_ORDER)
        metrics["cv_f1_mean"] = cv_scores[name]
        metrics["train_time_sec"] = round(time.time() - t0, 2)
        results.append(metrics)
        print(metrics)

    comparison_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    comparison_df.to_csv(os.path.join(ARTIFACT_DIR, "model_comparison.csv"), index=False)
    print("\n=== MODEL COMPARISON ===")
    print(comparison_df)

    # Best model selection: primarily ROC-AUC, tie-break on F1 — NOT accuracy alone
    best_row = comparison_df.sort_values(["roc_auc", "f1_score"], ascending=False).iloc[0]
    best_name = best_row["model"]
    print(f"\nBest model before tuning: {best_name}")

    # ---- Hyperparameter tuning on the best model ----
    param_distributions = {
        "Logistic Regression": {
            "C": [0.01, 0.1, 0.5, 1, 2, 5, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        },
        "Random Forest": {
            "n_estimators": [200, 300, 400, 500],
            "max_depth": [None, 8, 12, 16, 20],
            "min_samples_split": [2, 4, 6],
            "min_samples_leaf": [1, 2, 4],
        },
        "XGBoost": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.02, 0.05, 0.1, 0.2],
            "max_depth": [2, 3, 4, 5],
            "subsample": [0.7, 0.85, 1.0],
        },
    }

    base_model = MODELS[best_name].__class__(random_state=RANDOM_STATE) if best_name != "Logistic Regression" \
        else LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_distributions[best_name],
        n_iter=12,
        scoring="f1_weighted",
        cv=skf,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train_bal, y_train_bal)
    tuned_model = search.best_estimator_
    print(f"\nBest params for {best_name}: {search.best_params_}")

    tuned_metrics = evaluate_model(best_name + " (Tuned)", tuned_model, X_test_proc, y_test, CLASS_ORDER)
    print("Tuned metrics:", tuned_metrics)

    before_after = pd.DataFrame([
        {**dict(comparison_df[comparison_df["model"] == best_name].iloc[0]), "stage": "before_tuning"},
        {**tuned_metrics, "stage": "after_tuning"},
    ])
    before_after.to_csv(os.path.join(ARTIFACT_DIR, "tuning_before_after.csv"), index=False)
    print(before_after)

    # Only keep the tuned model if it's actually at least as good
    final_model = tuned_model if tuned_metrics["f1_score"] >= best_row["f1_score"] else fitted_models[best_name]
    final_name = best_name

    # ---- Feature importance (explainability) ----
    feature_names = preprocessor.get_feature_names_out()
    importance_records = []
    if hasattr(final_model, "feature_importances_"):
        importances = final_model.feature_importances_
    elif hasattr(final_model, "coef_"):
        importances = np.mean(np.abs(final_model.coef_), axis=0)
    else:
        importances = np.zeros(len(feature_names))

    for fname, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1])[:15]:
        importance_records.append({"feature": fname, "importance": round(float(imp), 5)})
    with open(os.path.join(ARTIFACT_DIR, "feature_importance.json"), "w") as f:
        json.dump(importance_records, f, indent=2)
    print("\nTop features:", importance_records[:8])

    # ---- Save final artifacts ----
    joblib.dump(preprocessor, os.path.join(ARTIFACT_DIR, "preprocessor.pkl"))
    joblib.dump(final_model, os.path.join(ARTIFACT_DIR, "model.pkl"))
    with open(os.path.join(ARTIFACT_DIR, "model_meta.json"), "w") as f:
        json.dump({
            "best_model_name": final_name,
            "classes": list(final_model.classes_),
            "metrics": tuned_metrics if final_model is tuned_model else
                       comparison_df[comparison_df["model"] == best_name].iloc[0].to_dict(),
            "trained_at": pd.Timestamp.now().isoformat(),
        }, f, indent=2, default=str)

    print(f"\nSaved model.pkl + preprocessor.pkl. Winning model: {final_name}")


if __name__ == "__main__":
    main()
