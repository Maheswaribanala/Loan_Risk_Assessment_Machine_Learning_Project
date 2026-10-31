"""
feature_engineering.py
-----------------------
Shared feature-engineering logic used by BOTH the training pipeline
(train_models.py) and the Flask backend at prediction time
(backend/app.py). Keeping this logic in one importable module is what
guarantees there is no training/prediction mismatch.

Derived features created here and why they matter for loan risk:

- debt_to_income_ratio     : Higher DTI means a larger share of income is
                              already committed to debt -> higher default risk.
- loan_to_income_ratio     : Large loans relative to annual income signal
                              over-leverage.
- monthly_repayment_estimate: Approximate EMI using an amortization formula;
                              lets the model reason about affordability
                              instead of raw loan amount alone.
- income_stability_score   : Employment type is a strong proxy for how
                              stable/predictable future income is.
- total_income             : Combined household income capacity.
- credit_history (binary)  : Derived from credit_score when not provided
                              directly; a clean/poor credit history is one
                              of the strongest predictors of repayment risk.
"""

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "age", "dependents", "applicant_income", "coapplicant_income",
    "loan_amount", "loan_term", "credit_score", "existing_loans",
    "debt_to_income_ratio", "loan_to_income_ratio",
    "monthly_repayment_estimate", "income_stability_score",
    "total_income", "credit_history",
]

CATEGORICAL_FEATURES = [
    "gender", "marital_status", "education", "employment_status",
    "self_employed", "repayment_history", "property_area",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

EMPLOYMENT_STABILITY_MAP = {
    "Salaried": 0.85,
    "Self-Employed": 0.55,
    "Business Owner": 0.60,
    "Unemployed": 0.10,
}

ANNUAL_INTEREST_RATE = 0.10  # assumed flat rate used only for the EMI estimate feature


def _monthly_emi(loan_amount, loan_term_months, annual_rate=ANNUAL_INTEREST_RATE):
    r = annual_rate / 12
    loan_term_months = np.where(loan_term_months <= 0, 12, loan_term_months)
    with np.errstate(divide="ignore", invalid="ignore"):
        emi = (loan_amount * r) / (1 - (1 + r) ** (-loan_term_months))
    return np.nan_to_num(emi, nan=0.0, posinf=0.0, neginf=0.0)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns to a raw applicant dataframe (idempotent)."""
    df = df.copy()

    # Fill sensible defaults for anything missing before deriving features
    if "coapplicant_income" not in df:
        df["coapplicant_income"] = 0.0
    if "credit_history" not in df or df["credit_history"].isnull().all():
        df["credit_history"] = (df["credit_score"].fillna(0) >= 650).astype(int)

    df["total_income"] = df["applicant_income"].fillna(0) + df["coapplicant_income"].fillna(0)

    df["monthly_repayment_estimate"] = _monthly_emi(
        df["loan_amount"].fillna(0).to_numpy(dtype=float),
        df["loan_term"].fillna(12).to_numpy(dtype=float),
    )

    df["debt_to_income_ratio"] = (
        (df["monthly_repayment_estimate"] + df.get("existing_loans", 0).fillna(0) * 150)
        / (df["total_income"] + 1)
    ).round(3)

    df["loan_to_income_ratio"] = (
        df["loan_amount"].fillna(0) / (df["total_income"] * 12 + 1)
    ).round(3)

    if "employment_status" in df:
        df["income_stability_score"] = df["employment_status"].map(
            EMPLOYMENT_STABILITY_MAP
        ).fillna(0.4)
    else:
        df["income_stability_score"] = 0.4

    return df


def select_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the columns the model pipeline expects, in order."""
    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    return df[ALL_FEATURES]
