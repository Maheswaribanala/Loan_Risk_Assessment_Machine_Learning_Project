"""
db.py — MySQL data access layer for the Loan Risk Assessment System.
Connects to the real MySQL database (loan_risk_assessment) created via
schema_mysql.sql in MySQL Workbench.
"""

import json
import mysql.connector

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "Maheswari@2006", 
    "database": "loan_risk_assessment",
}


def _get_connection():
    return mysql.connector.connect(**DB_CONFIG)


def init_db():
    # Tables already created via schema_mysql.sql in MySQL Workbench.
    # This just confirms the connection works at Flask startup.
    conn = _get_connection()
    conn.close()
    print("Connected to MySQL database: loan_risk_assessment")


def insert_applicant(data: dict) -> int:
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO applicants
        (applicant_name, age, gender, marital_status, education, dependents,
         employment_status, self_employed, applicant_income, coapplicant_income,
         loan_amount, loan_term, credit_score, credit_history, existing_loans,
         repayment_history, property_area)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        data.get("applicant_name"), data.get("age"), data.get("gender"),
        data.get("marital_status"), data.get("education"), data.get("dependents"),
        data.get("employment_status"), data.get("self_employed"),
        data.get("applicant_income"), data.get("coapplicant_income"),
        data.get("loan_amount"), data.get("loan_term"), data.get("credit_score"),
        data.get("credit_history"), data.get("existing_loans"),
        data.get("repayment_history"), data.get("property_area"),
    ))
    applicant_id = cur.lastrowid
    conn.commit()
    cur.close()
    conn.close()
    return applicant_id


def insert_prediction(applicant_id, predicted_class, risk_category,
                       risk_probability, model_used, top_factors):
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO predictions
        (applicant_id, predicted_class, risk_category, risk_probability,
         model_used, top_factors)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (applicant_id, predicted_class, risk_category, risk_probability,
          model_used, json.dumps(top_factors)))
    prediction_id = cur.lastrowid
    conn.commit()
    cur.close()
    conn.close()
    return prediction_id


def get_applications(limit=100):
    conn = _get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT a.id as applicant_id, a.applicant_name, a.age, a.gender,
               a.applicant_income, a.coapplicant_income, a.loan_amount,
               a.credit_score, a.created_at,
               p.risk_category, p.risk_probability, p.model_used,
               p.prediction_date
        FROM applicants a
        LEFT JOIN predictions p ON p.applicant_id = a.id
        ORDER BY a.id DESC
        LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_dashboard_stats():
    conn = _get_connection()
    cur = conn.cursor(dictionary=True)

    cur.execute("SELECT COUNT(*) as total FROM applicants")
    total = cur.fetchone()["total"]

    cur.execute("""
        SELECT risk_category, COUNT(*) as count
        FROM predictions GROUP BY risk_category
    """)
    dist = {row["risk_category"]: row["count"] for row in cur.fetchall()}

    cur.execute("""
        SELECT a.applicant_name, p.risk_category, p.risk_probability,
               p.prediction_date
        FROM predictions p
        JOIN applicants a ON a.id = p.applicant_id
        ORDER BY p.id DESC LIMIT 10
    """)
    recent = cur.fetchall()

    cur.close()
    conn.close()
    return {
        "total_applications": total,
        "low_risk": dist.get("Low", 0),
        "medium_risk": dist.get("Medium", 0),
        "high_risk": dist.get("High", 0),
        "recent_predictions": recent,
    }