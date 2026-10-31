"""
predictor.py
-------------
Loads the trained preprocessor + model saved by ml/train_models.py and
runs REAL predictions (no hard-coded / fake results). Uses the exact
same feature_engineering.py module used during training, so there is no
training/prediction mismatch.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

ML_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "ml")
sys.path.append(os.path.abspath(ML_DIR))

from feature_engineering import engineer_features, select_model_columns  # noqa: E402

MODEL_PATH = os.path.join(ML_DIR, "model.pkl")
PREPROCESSOR_PATH = os.path.join(ML_DIR, "preprocessor.pkl")
META_PATH = os.path.join(ML_DIR, "model_meta.json")
FEATURE_IMPORTANCE_PATH = os.path.join(ML_DIR, "feature_importance.json")

_model = None
_preprocessor = None
_meta = None
_feature_importance = None


class ModelNotLoadedError(Exception):
    pass


def load_artifacts():
    global _model, _preprocessor, _meta, _feature_importance
    if not (os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH)):
        raise ModelNotLoadedError(
            "Trained model artifacts not found. Run `python3 ml/train_models.py` first."
        )
    _model = joblib.load(MODEL_PATH)
    _preprocessor = joblib.load(PREPROCESSOR_PATH)
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            _meta = json.load(f)
    if os.path.exists(FEATURE_IMPORTANCE_PATH):
        with open(FEATURE_IMPORTANCE_PATH) as f:
            _feature_importance = json.load(f)
    return _model, _preprocessor


def is_loaded():
    return _model is not None and _preprocessor is not None


RISK_ICON = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

RECOMMENDATIONS = {
    "Low": "Loan application appears relatively low risk. Standard verification is sufficient.",
    "Medium": "Additional financial verification is recommended before approval.",
    "High": "Applicant has a higher estimated repayment risk. Recommend collateral, a "
            "co-signer, or a reduced loan amount.",
}

# Human-friendly labels for the top contributing raw fields, used to explain
# a specific prediction (as opposed to the global feature_importance.json).
FACTOR_LABELS = {
    "credit_history": "Credit history",
    "credit_score": "Credit score",
    "applicant_income": "Applicant income",
    "total_income": "Total household income",
    "debt_to_income_ratio": "Debt-to-income ratio",
    "loan_to_income_ratio": "Loan-to-income ratio",
    "existing_loans": "Existing loans",
    "repayment_history": "Repayment history",
    "employment_status": "Employment status",
    "income_stability_score": "Income stability",
    "loan_amount": "Loan amount",
    "monthly_repayment_estimate": "Estimated monthly repayment",
}


def _top_factors_for_applicant(raw_row: dict, n=5):
    """Cheap, transparent, per-applicant explanation: rank the applicant's
    own field values against the GLOBAL feature-importance ranking learned
    at training time, and surface the top overlapping ones in plain
    English. (Global importances come from feature_importance.json,
    produced by train_models.py from the winning model's
    feature_importances_/coef_.)"""
    if not _feature_importance:
        return []

    factors = []
    for item in _feature_importance:
        fname = item["feature"]
        base = fname.split("__")[-1]
        for key, label in FACTOR_LABELS.items():
            if base.startswith(key):
                value = raw_row.get(key)
                if value is not None:
                    factors.append({"factor": label, "value": value, "importance": item["importance"]})
                break
    # de-duplicate keeping the highest-importance mention of each factor
    seen = {}
    for f in factors:
        if f["factor"] not in seen or f["importance"] > seen[f["factor"]]["importance"]:
            seen[f["factor"]] = f
    ranked = sorted(seen.values(), key=lambda x: -x["importance"])
    return ranked[:n]


def predict(applicant_dict: dict) -> dict:
    if not is_loaded():
        load_artifacts()

    df = pd.DataFrame([applicant_dict])
    df = engineer_features(df)
    X = select_model_columns(df)

    X_proc = _preprocessor.transform(X)
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    pred_class = _model.predict(X_proc)[0]
    proba = _model.predict_proba(X_proc)[0]
    class_index = list(_model.classes_).index(pred_class)
    probability = float(proba[class_index])

    all_probs = {cls: round(float(p), 4) for cls, p in zip(_model.classes_, proba)}

    top_factors = _top_factors_for_applicant(df.iloc[0].to_dict())

    model_name = _meta.get("best_model_name", _model.__class__.__name__) if _meta else _model.__class__.__name__

    return {
        "risk_category": pred_class,
        "risk_icon": RISK_ICON.get(pred_class, ""),
        "risk_probability": round(probability * 100, 2),
        "all_class_probabilities": all_probs,
        "recommendation": RECOMMENDATIONS.get(pred_class, ""),
        "top_factors": top_factors,
        "model_used": model_name,
    }
