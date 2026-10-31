"""
preprocessing.py
-----------------
Data cleaning (missing values, duplicates, outliers) and the
ColumnTransformer used to encode/scale features consistently between
training and inference.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineering import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle duplicates and obvious bad values on the RAW training data.
    (Missing-value imputation itself happens inside the sklearn Pipeline
    so that the exact same logic applies at prediction time.)"""
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows")

    # Outlier handling: cap extreme numeric values at the 1st/99th percentile
    # (winsorizing) rather than dropping rows, so we don't lose scarce
    # high-risk examples.
    for col in ["applicant_income", "coapplicant_income", "loan_amount"]:
        if col in df.columns:
            low, high = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(lower=low, upper=high)

    # Basic sanity filters
    if "age" in df.columns:
        df = df[(df["age"].isna()) | ((df["age"] >= 18) & (df["age"] <= 100))]
    if "loan_amount" in df.columns:
        df = df[(df["loan_amount"].isna()) | (df["loan_amount"] > 0)]

    return df.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    """Builds the ColumnTransformer: median-impute + scale numeric,
    most-frequent-impute + one-hot encode categorical."""
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return preprocessor
