"""
validation.py — validates incoming applicant JSON before it ever touches
the ML pipeline or the database, so bad input produces a clean 400 error
instead of a traceback.
"""

REQUIRED_FIELDS = [
    "applicant_name", "age", "gender", "marital_status", "education",
    "dependents", "employment_status", "applicant_income", "loan_amount",
    "loan_term",
]

NUMERIC_FIELDS = [
    "age", "dependents", "applicant_income", "coapplicant_income",
    "loan_amount", "loan_term", "credit_score", "existing_loans",
]

ALLOWED_VALUES = {
    "gender": {"Male", "Female", "Other"},
    "marital_status": {"Single", "Married", "Divorced"},
    "education": {"Graduate", "Not Graduate"},
    "employment_status": {"Salaried", "Self-Employed", "Business Owner", "Unemployed"},
    "self_employed": {"Yes", "No"},
    "repayment_history": {"Excellent", "Good", "Average", "Poor", "No History"},
    "property_area": {"Urban", "Semiurban", "Rural"},
}


def validate_applicant_payload(data: dict):
    """Returns (is_valid, errors_list)."""
    errors = []

    if not isinstance(data, dict):
        return False, ["Request body must be a JSON object."]

    for field in REQUIRED_FIELDS:
        if field not in data or data[field] in (None, ""):
            errors.append(f"Missing required field: '{field}'")

    for field in NUMERIC_FIELDS:
        if field in data and data[field] not in (None, ""):
            try:
                float(data[field])
            except (TypeError, ValueError):
                errors.append(f"Field '{field}' must be numeric.")

    if "age" in data and data.get("age") not in (None, ""):
        try:
            age_val = float(data["age"])
            if not (18 <= age_val <= 100):
                errors.append("Field 'age' must be between 18 and 100.")
        except (TypeError, ValueError):
            pass

    if "applicant_income" in data and data.get("applicant_income") not in (None, ""):
        try:
            if float(data["applicant_income"]) < 0:
                errors.append("Field 'applicant_income' cannot be negative.")
        except (TypeError, ValueError):
            pass

    if "loan_amount" in data and data.get("loan_amount") not in (None, ""):
        try:
            if float(data["loan_amount"]) <= 0:
                errors.append("Field 'loan_amount' must be greater than 0.")
        except (TypeError, ValueError):
            pass

    for field, allowed in ALLOWED_VALUES.items():
        if field in data and data[field] not in (None, "") and data[field] not in allowed:
            errors.append(f"Field '{field}' must be one of {sorted(allowed)}.")

    return (len(errors) == 0), errors
