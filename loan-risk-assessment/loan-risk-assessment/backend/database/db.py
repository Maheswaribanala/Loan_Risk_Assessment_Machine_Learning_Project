"""
db.py
------
Database access layer for the Loan Risk Assessment System.

IMPORTANT NOTE ON MySQL vs SQLite:
This project is specified to use MySQL, and the schema in
`schema_mysql.sql` (same folder) is written for MySQL and matches the
spec exactly (database `loan_risk_assessment`, tables `applicants` and
`predictions`).

However, this build happened inside an offline sandbox with no internet
access and no MySQL server process available, so it cannot actually
start/connect to a MySQL instance here. To keep the whole system
genuinely runnable end-to-end for demonstration, this module uses
Python's built-in `sqlite3` (zero setup, ships with Python) with the
IDENTICAL table structure and column names as the MySQL schema.

TO SWITCH TO REAL MYSQL (recommended for your actual submission/deployment):
1. `pip install mysql-connector-python`
2. Run `schema_mysql.sql` against your MySQL server:
       mysql -u root -p < backend/database/schema_mysql.sql
3. Replace the `_get_connection()` function below with:
       import mysql.connector
       def _get_connection():
           return mysql.connector.connect(
               host="localhost", user="root", password="yourpassword",
               database="loan_risk_assessment"
           )
   Every other function in this file (insert_applicant, insert_prediction,
   get_applications, get_dashboard_stats...) uses standard parameterized
   SQL (`%s` placeholders work for both sqlite3's `?`-style via the
   PLACEHOLDER constant below) so the rest of the app does not need to change.
"""

import json
import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "loan_risk_assessment.db")
PLACEHOLDER = "?"  # would be "%s" for mysql-connector-python


def _get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = _get_connection()
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS applicants (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        applicant_name TEXT NOT NULL,
        age INTEGER,
        gender TEXT,
        marital_status TEXT,
        education TEXT,
        dependents INTEGER,
        employment_status TEXT,
        self_employed TEXT,
        applicant_income REAL,
        coapplicant_income REAL,
        loan_amount REAL,
        loan_term INTEGER,
        credit_score INTEGER,
        credit_history INTEGER,
        existing_loans INTEGER,
        repayment_history TEXT,
        property_area TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        applicant_id INTEGER NOT NULL,
        predicted_class TEXT,
        risk_category TEXT,
        risk_probability REAL,
        model_used TEXT,
        top_factors TEXT,
        prediction_date TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (applicant_id) REFERENCES applicants (id)
    );
    """)
    conn.commit()
    conn.close()


def insert_applicant(data: dict) -> int:
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO applicants
        (applicant_name, age, gender, marital_status, education, dependents,
         employment_status, self_employed, applicant_income, coapplicant_income,
         loan_amount, loan_term, credit_score, credit_history, existing_loans,
         repayment_history, property_area)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        VALUES (?, ?, ?, ?, ?, ?)
    """, (applicant_id, predicted_class, risk_category, risk_probability,
          model_used, json.dumps(top_factors)))
    prediction_id = cur.lastrowid
    conn.commit()
    conn.close()
    return prediction_id


def get_applications(limit=100):
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT a.id as applicant_id, a.applicant_name, a.age, a.gender,
               a.applicant_income, a.coapplicant_income, a.loan_amount,
               a.credit_score, a.created_at,
               p.risk_category, p.risk_probability, p.model_used,
               p.prediction_date
        FROM applicants a
        LEFT JOIN predictions p ON p.applicant_id = a.id
        ORDER BY a.id DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_dashboard_stats():
    conn = _get_connection()
    cur = conn.cursor()
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
    recent = [dict(r) for r in cur.fetchall()]

    conn.close()
    return {
        "total_applications": total,
        "low_risk": dist.get("Low", 0),
        "medium_risk": dist.get("Medium", 0),
        "high_risk": dist.get("High", 0),
        "recent_predictions": recent,
    }
