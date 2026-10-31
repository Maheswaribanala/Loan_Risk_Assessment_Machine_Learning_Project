"""
app.py — Flask backend for the Intelligent Loan Risk Assessment System.

Serves the static frontend AND exposes the REST API:
  POST /api/predict            -> validate, preprocess, predict, store, return result
  POST /api/applicant           -> store an applicant record without predicting (rarely needed;
                                    /api/predict already stores both applicant + prediction)
  GET  /api/applications        -> application history (for the dashboard table)
  GET  /api/dashboard           -> aggregate stats for dashboard cards/charts
  GET  /api/model-performance   -> model comparison table + tuning + feature importance

Run:
    cd backend
    python3 app.py
Then open http://localhost:5000/
"""

import json
import os
import sys

from flask import Flask, jsonify, request, send_from_directory

sys.path.append(os.path.dirname(__file__))
from database import db  # noqa: E402
from utils import predictor  # noqa: E402
from utils.validation import validate_applicant_payload  # noqa: E402

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
ML_DIR = os.path.join(os.path.dirname(__file__), "..", "ml")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")


# ---------------------------------------------------------------- startup --
def _startup():
    db.init_db()
    try:
        predictor.load_artifacts()
        print("Model artifacts loaded successfully.")
    except predictor.ModelNotLoadedError as e:
        print(f"WARNING: {e}")


_startup()


# ------------------------------------------------------------- error utils --
def error_response(message, status=400, details=None):
    payload = {"success": False, "error": message}
    if details:
        payload["details"] = details
    return jsonify(payload), status


@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/api/"):
        return error_response("Endpoint not found.", 404)
    # Fallback to SPA-style index for unknown frontend routes
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.errorhandler(500)
def server_error(e):
    return error_response("Internal server error. Please try again.", 500)


# ------------------------------------------------------------ frontend routes --
@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def frontend_files(filename):
    full_path = os.path.join(FRONTEND_DIR, filename)
    if os.path.isfile(full_path):
        return send_from_directory(FRONTEND_DIR, filename)
    return error_response("File not found", 404)


# ------------------------------------------------------------------ API: predict --
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not request.is_json:
        return error_response("Request must be JSON.")

    data = request.get_json(silent=True)
    if data is None:
        return error_response("Invalid or malformed JSON body.")

    is_valid, errors = validate_applicant_payload(data)
    if not is_valid:
        return error_response("Validation failed.", 400, errors)

    try:
        result = predictor.predict(data)
    except predictor.ModelNotLoadedError as e:
        return error_response(str(e), 503)
    except Exception as e:
        return error_response(f"Prediction failed: {e}", 500)

    try:
        applicant_id = db.insert_applicant(data)
        db.insert_prediction(
            applicant_id=applicant_id,
            predicted_class=result["risk_category"],
            risk_category=result["risk_category"],
            risk_probability=result["risk_probability"],
            model_used=result["model_used"],
            top_factors=result["top_factors"],
        )
    except Exception as e:
        # Prediction succeeded even if storage failed — tell the user, don't crash.
        return jsonify({
            "success": True,
            "warning": f"Prediction succeeded but could not be saved to the database: {e}",
            "applicant_name": data.get("applicant_name"),
            **result,
        })

    return jsonify({
        "success": True,
        "applicant_id": applicant_id,
        "applicant_name": data.get("applicant_name"),
        **result,
    })


# ------------------------------------------------------------------ API: applicant --
@app.route("/api/applicant", methods=["POST"])
def api_applicant():
    data = request.get_json(silent=True)
    if data is None:
        return error_response("Invalid or malformed JSON body.")
    is_valid, errors = validate_applicant_payload(data)
    if not is_valid:
        return error_response("Validation failed.", 400, errors)
    try:
        applicant_id = db.insert_applicant(data)
    except Exception as e:
        return error_response(f"Database error: {e}", 500)
    return jsonify({"success": True, "applicant_id": applicant_id})


# ------------------------------------------------------------------ API: applications --
@app.route("/api/applications", methods=["GET"])
def api_applications():
    try:
        limit = int(request.args.get("limit", 100))
    except ValueError:
        limit = 100
    try:
        rows = db.get_applications(limit=limit)
    except Exception as e:
        return error_response(f"Database error: {e}", 500)
    return jsonify({"success": True, "count": len(rows), "applications": rows})


# ------------------------------------------------------------------ API: dashboard --
@app.route("/api/dashboard", methods=["GET"])
def api_dashboard():
    try:
        stats = db.get_dashboard_stats()
    except Exception as e:
        return error_response(f"Database error: {e}", 500)
    return jsonify({"success": True, **stats})


# ------------------------------------------------------------------ API: model performance --
@app.route("/api/model-performance", methods=["GET"])
def api_model_performance():
    comparison_path = os.path.join(ML_DIR, "model_comparison.csv")
    meta_path = os.path.join(ML_DIR, "model_meta.json")
    tuning_path = os.path.join(ML_DIR, "tuning_before_after.csv")
    fi_path = os.path.join(ML_DIR, "feature_importance.json")

    if not os.path.exists(comparison_path):
        return error_response("Model performance data not available. Train the models first.", 503)

    import csv

    def read_csv(path):
        with open(path) as f:
            return list(csv.DictReader(f))

    response = {"success": True, "comparison": read_csv(comparison_path)}

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            response["best_model"] = json.load(f)
    if os.path.exists(tuning_path):
        response["tuning_before_after"] = read_csv(tuning_path)
    if os.path.exists(fi_path):
        with open(fi_path) as f:
            response["feature_importance"] = json.load(f)

    return jsonify(response)


@app.route("/ml-plots/<path:filename>")
def ml_plots(filename):
    plots_dir = os.path.join(ML_DIR, "plots")
    full_path = os.path.join(plots_dir, filename)
    if os.path.isfile(full_path):
        return send_from_directory(plots_dir, filename)
    return error_response("Plot not found", 404)


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({
        "success": True,
        "model_loaded": predictor.is_loaded(),
        "status": "ok",
    })


if __name__ == "__main__":
    import logging
    import flask.cli

    logging.getLogger("werkzeug").setLevel(logging.ERROR)   # WARNING, Running on, request logs hide
    flask.cli.show_server_banner = lambda *args, **kwargs: None   # "Serving Flask app", "Debug mode" hide

    print("Server running on http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)