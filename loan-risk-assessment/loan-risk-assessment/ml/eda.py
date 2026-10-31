"""
eda.py — Exploratory Data Analysis for the Loan Risk dataset.

Run: python3 eda.py
Outputs a text summary + saves a few plots into ml/eda_outputs/
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = "../dataset/loan_dataset.csv"
OUT_DIR = "eda_outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def run_eda():
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("SHAPE:", df.shape)
    print("=" * 60)
    print("\nCOLUMN DTYPES:\n", df.dtypes)
    print("\nMISSING VALUES:\n", df.isnull().sum()[df.isnull().sum() > 0])
    print("\nDUPLICATE ROWS:", df.duplicated().sum())
    print("\nTARGET DISTRIBUTION:\n", df["risk_category"].value_counts(normalize=True))
    print("\nNUMERIC SUMMARY:\n", df.describe().T)

    # Target distribution plot
    plt.figure(figsize=(5, 4))
    df["risk_category"].value_counts().plot(kind="bar", color=["#2ecc71", "#f1c40f", "#e74c3c"])
    plt.title("Risk Category Distribution")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/risk_distribution.png")
    plt.close()

    # Correlation heatmap (numeric only)
    numeric_df = df.select_dtypes(include="number")
    plt.figure(figsize=(9, 7))
    corr = numeric_df.corr()
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/correlation_heatmap.png")
    plt.close()

    # Income vs risk
    plt.figure(figsize=(6, 4))
    df.boxplot(column="applicant_income", by="risk_category")
    plt.title("Applicant Income by Risk Category")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/income_by_risk.png")
    plt.close()

    print(f"\nEDA plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    run_eda()
