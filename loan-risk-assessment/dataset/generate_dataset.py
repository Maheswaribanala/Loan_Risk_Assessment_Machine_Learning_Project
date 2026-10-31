"""
generate_dataset.py
--------------------
Generates a realistic synthetic loan-applicant dataset for the
Intelligent Loan Risk Assessment System.

WHY SYNTHETIC DATA?
This project runs in an offline / sandboxed environment with no internet
access, so a public dataset (e.g. Kaggle's Loan Prediction dataset) cannot
be downloaded. Instead we generate a large, statistically realistic dataset
using domain-informed rules (income, credit score, DTI, employment, etc.)
with controlled randomness and noise, so the ML pipeline has real signal
to learn from (not just random labels).

To use a real dataset instead: drop any CSV with compatible columns into
this folder as `loan_dataset.csv` and re-run the ML pipeline in ml/.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 6000

genders = np.random.choice(["Male", "Female"], size=N, p=[0.58, 0.42])
marital = np.random.choice(["Single", "Married", "Divorced"], size=N, p=[0.38, 0.52, 0.10])
education = np.random.choice(["Graduate", "Not Graduate"], size=N, p=[0.68, 0.32])
employment = np.random.choice(
    ["Salaried", "Self-Employed", "Business Owner", "Unemployed"],
    size=N, p=[0.55, 0.22, 0.15, 0.08]
)
dependents = np.random.choice([0, 1, 2, 3], size=N, p=[0.45, 0.25, 0.20, 0.10])
property_area = np.random.choice(["Urban", "Semiurban", "Rural"], size=N, p=[0.40, 0.35, 0.25])

age = np.clip(np.random.normal(38, 11, N), 21, 68).astype(int)

# Income correlated with employment type and education
base_income = np.random.gamma(shape=4.0, scale=1500, size=N)
edu_bonus = np.where(education == "Graduate", 1.25, 1.0)
emp_multiplier = np.select(
    [employment == "Salaried", employment == "Self-Employed",
     employment == "Business Owner", employment == "Unemployed"],
    [1.0, 1.15, 1.35, 0.25]
)
applicant_income = np.round(base_income * edu_bonus * emp_multiplier, 2)

has_coapplicant = np.random.choice([0, 1], size=N, p=[0.35, 0.65])
coapplicant_income = np.where(
    has_coapplicant == 1,
    np.round(np.random.gamma(shape=3.0, scale=900, size=N), 2),
    0.0
)

total_income = applicant_income + coapplicant_income

# Loan amount tends to scale with income but has its own noise (aspiration/need)
loan_amount = np.round(
    np.clip(total_income * np.random.uniform(1.5, 6.5, N) / 1000, 20, 700) * 1000, -2
)

loan_term = np.random.choice([12, 36, 60, 84, 120, 180, 240, 360], size=N,
                              p=[0.05, 0.10, 0.15, 0.15, 0.20, 0.15, 0.10, 0.10])

# Credit score correlated loosely with income stability & employment,
# with substantial independent randomness (credit history is its own signal)
credit_base = 300 + emp_multiplier * 120 + (np.log1p(applicant_income) * 25)
credit_score = np.clip(credit_base + np.random.normal(0, 60, N), 300, 900).astype(int)

credit_history = np.where(credit_score >= 650, 1, np.where(credit_score >= 550,
                           np.random.choice([0, 1], N, p=[0.4, 0.6]), 0))

existing_loans = np.random.choice([0, 1, 2, 3, 4], size=N, p=[0.40, 0.28, 0.18, 0.09, 0.05])

repayment_history = np.select(
    [credit_score >= 720, credit_score >= 600, credit_score >= 480],
    ["Excellent", "Good", "Average"],
    default="Poor"
)

self_employed = np.where(employment == "Self-Employed", "Yes", "No")

# ---- Derived / engineered features ----
monthly_income = total_income
monthly_repayment_estimate = np.round(
    (loan_amount * (0.10 / 12)) / (1 - (1 + 0.10 / 12) ** (-loan_term)), 2
)
monthly_repayment_estimate = np.nan_to_num(monthly_repayment_estimate, nan=0, posinf=0, neginf=0)

debt_to_income_ratio = np.round(
    np.where(monthly_income > 0,
             (monthly_repayment_estimate + existing_loans * 150) / (monthly_income + 1),
             1.0), 3
)
loan_to_income_ratio = np.round(loan_amount / (total_income * 12 + 1), 3)

income_stability_score = np.select(
    [employment == "Salaried", employment == "Self-Employed",
     employment == "Business Owner", employment == "Unemployed"],
    [0.85, 0.55, 0.60, 0.10]
)

# ---- Target risk logic (ground truth generator) ----
# Weighted risk score built from real-world-plausible factors + noise.
risk_score = (
    (1 - credit_history) * 2.2
    + (900 - credit_score) / 900 * 2.5
    + debt_to_income_ratio * 3.0
    + loan_to_income_ratio * 1.5
    + existing_loans * 0.35
    + (1 - income_stability_score) * 1.8
    + np.where(repayment_history == "Poor", 1.5,
      np.where(repayment_history == "Average", 0.7, 0.0))
    + np.random.normal(0, 0.9, N)  # noise so the problem isn't trivially separable
)

low_thresh, high_thresh = np.quantile(risk_score, [0.45, 0.80])
risk_category = np.select(
    [risk_score <= low_thresh, risk_score <= high_thresh],
    ["Low", "Medium"],
    default="High"
)

names_first = ["Rahul", "Priya", "Aman", "Sneha", "Vikram", "Anjali", "Karan", "Divya",
               "Arjun", "Neha", "Rohit", "Pooja", "Sanjay", "Kavya", "Manoj", "Isha"]
names_last = ["Sharma", "Verma", "Patel", "Reddy", "Nair", "Gupta", "Iyer", "Singh",
              "Das", "Rao", "Mehta", "Joshi"]
applicant_name = [f"{np.random.choice(names_first)} {np.random.choice(names_last)}" for _ in range(N)]

df = pd.DataFrame({
    "applicant_name": applicant_name,
    "age": age,
    "gender": genders,
    "marital_status": marital,
    "education": education,
    "dependents": dependents,
    "employment_status": employment,
    "self_employed": self_employed,
    "applicant_income": applicant_income,
    "coapplicant_income": coapplicant_income,
    "loan_amount": loan_amount,
    "loan_term": loan_term,
    "credit_score": credit_score,
    "credit_history": credit_history,
    "existing_loans": existing_loans,
    "repayment_history": repayment_history,
    "property_area": property_area,
    "debt_to_income_ratio": debt_to_income_ratio,
    "loan_to_income_ratio": loan_to_income_ratio,
    "monthly_repayment_estimate": monthly_repayment_estimate,
    "income_stability_score": income_stability_score,
    "risk_category": risk_category,
})

# introduce a small amount of realistic missingness + a few duplicate rows
for col in ["credit_score", "coapplicant_income", "dependents", "self_employed"]:
    mask = np.random.rand(N) < 0.02
    df.loc[mask, col] = np.nan

dupes = df.sample(25, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)

df.to_csv("/home/claude/loan-risk-assessment/dataset/loan_dataset.csv", index=False)
print("Dataset generated:", df.shape)
print(df["risk_category"].value_counts())
