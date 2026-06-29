"""Flask web application for Rising Water Prediction."""

import json
import sys
from pathlib import Path

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import BEST_MODEL_PATH, CATEGORICAL_FEATURES, Config, NUMERIC_FEATURES, REPORTS_DIR, TARGET_COLUMN
from src.dataset import (
    REQUIRED_COLUMNS,
    append_row,
    get_dataset_preview,
    import_csv,
    load_or_create_dataset,
)
from src.predict import predict_water_rise

app = Flask(__name__)
app.config.from_object(Config)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def model_ready() -> bool:
    return BEST_MODEL_PATH.exists()


@app.route("/")
def index():
    return render_template("index.html", model_ready=model_ready())


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    if request.method == "POST":
        if not model_ready():
            flash("Model not trained yet. Run: python -m src.train", "warning")
            return redirect(url_for("predict"))

        try:
            data = {key: request.form[key] for key in request.form}
            for col in NUMERIC_FEATURES:
                if col in data:
                    data[col] = float(data[col])
            result = predict_water_rise(data)
        except (ValueError, RuntimeError) as exc:
            flash(str(exc), "danger")

    return render_template(
        "prediction.html",
        result=result,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        model_ready=model_ready(),
    )


@app.route("/dataset", methods=["GET", "POST"])
def dataset():
    if request.method == "POST":
        action = request.form.get("action", "manual")

        try:
            if action == "upload":
                file = request.files.get("dataset_file")
                if not file or not file.filename:
                    raise ValueError("Please choose a CSV file to upload.")

                filename = secure_filename(file.filename)
                if not filename.lower().endswith(".csv"):
                    raise ValueError("Only CSV files are supported.")

                result = import_csv(file.read())
                flash(
                    f"Uploaded {result['added']} record(s). Dataset now has {result['total']} rows. "
                    "Retrain with python -m src.train to update predictions.",
                    "success",
                )
            else:
                data = {key: request.form[key] for key in request.form if key != "action"}
                total = append_row(data)
                flash(
                    f"Record added successfully. Dataset now has {total} rows. "
                    "Retrain with python -m src.train to update predictions.",
                    "success",
                )
        except ValueError as exc:
            flash(str(exc), "danger")

        return redirect(url_for("dataset"))

    df = load_or_create_dataset()
    return render_template(
        "dataset.html",
        total_rows=len(df),
        required_columns=REQUIRED_COLUMNS,
        preview=get_dataset_preview(),
        model_ready=model_ready(),
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not model_ready():
        return jsonify({"error": "Model not trained"}), 503

    payload = request.get_json(silent=True) or {}
    try:
        for col in NUMERIC_FEATURES:
            if col in payload:
                payload[col] = float(payload[col])
        return jsonify(predict_water_rise(payload))
    except (ValueError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/dataset", methods=["POST"])
def api_dataset():
    payload = request.get_json(silent=True) or {}
    try:
        total = append_row(payload)
        return jsonify({"message": "Record added", "total_rows": total})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/dashboard")
def dashboard():
    df = load_or_create_dataset()
    metrics = None
    metrics_path = REPORTS_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())

    summary = {
        "rows": len(df),
        "avg_rainfall": round(df["rainfall_mm"].mean(), 2) if "rainfall_mm" in df.columns else None,
        "avg_water_rise": round(df["water_rise_cm"].mean(), 2) if TARGET_COLUMN in df.columns else None,
    }
    return render_template("dashboard.html", summary=summary, metrics=metrics, model_ready=model_ready())


@app.route("/about")
def about():
    return render_template("about.html")


@app.errorhandler(404)
def not_found(error):
    return render_template("error.html", code=404, message="Page not found"), 404


@app.errorhandler(500)
def server_error(error):
    return render_template("error.html", code=500, message="Internal server error"), 500


if __name__ == "__main__":
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
