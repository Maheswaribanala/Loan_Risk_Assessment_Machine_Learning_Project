-- schema_mysql.sql
-- Run with: mysql -u root -p < schema_mysql.sql
-- Creates the production MySQL database & tables exactly per the spec.
-- (The app itself runs on SQLite by default in this offline build — see
--  db.py's docstring for the 3-step switch-over to this schema.)

CREATE DATABASE IF NOT EXISTS loan_risk_assessment
    CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE loan_risk_assessment;

CREATE TABLE IF NOT EXISTS applicants (
    id                  INT AUTO_INCREMENT PRIMARY KEY,
    applicant_name      VARCHAR(150) NOT NULL,
    age                 INT,
    gender              VARCHAR(20),
    marital_status      VARCHAR(20),
    education           VARCHAR(30),
    dependents          INT,
    employment_status   VARCHAR(30),
    self_employed       VARCHAR(5),
    applicant_income    DECIMAL(14,2),
    coapplicant_income  DECIMAL(14,2),
    loan_amount         DECIMAL(14,2),
    loan_term           INT,
    credit_score        INT,
    credit_history      TINYINT,
    existing_loans      INT,
    repayment_history   VARCHAR(20),
    property_area       VARCHAR(20),
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS predictions (
    id                  INT AUTO_INCREMENT PRIMARY KEY,
    applicant_id        INT NOT NULL,
    predicted_class     VARCHAR(20),
    risk_category       VARCHAR(20),
    risk_probability    DECIMAL(6,4),
    model_used          VARCHAR(60),
    top_factors         JSON,
    prediction_date     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (applicant_id) REFERENCES applicants(id)
) ENGINE=InnoDB;

CREATE INDEX idx_predictions_applicant ON predictions(applicant_id);
CREATE INDEX idx_predictions_risk ON predictions(risk_category);
