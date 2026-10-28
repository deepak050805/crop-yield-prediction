"""HTTP adapters for the historical explorer and prediction service."""

import json
from pathlib import Path

from flask import jsonify, render_template, request

from backend.validation import ValidationError, parse_prediction, parse_selection


def register_routes(app):
    store = app.extensions["fieldwise"]["store"]

    @app.errorhandler(ValidationError)
    def invalid_input(error):
        return jsonify(error=str(error), code="invalid_input", field=error.field), 400

    def page(template, active):
        return render_template(
            template,
            active=active,
            districts=store.districts,
            crops=store.crops,
            first_year=store.first_year,
            last_year=store.last_year,
        )

    @app.get("/")
    def home():
        return page("index.html", "overview")

    @app.get("/predictor")
    def predictor():
        return page("predictor.html", "predictor")

    @app.get("/history")
    def history():
        return page("history.html", "history")

    @app.post("/get_data")
    def get_data():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ValidationError("Send a JSON object with crop and district.")
        selection = parse_selection(data, store.crops, store.districts)
        return jsonify(store.records(selection.crop, selection.district))

    @app.get("/model_info")
    def model_info():
        path = Path(app.config["METRICS_PATH"])
        if not path.exists():
            return jsonify(
                error="Model evaluation is not available.", code="model_unavailable"
            ), 503
        return jsonify(json.loads(path.read_text(encoding="utf-8")))

    @app.post("/predict")
    def predict():
        values = parse_prediction(request.form, store.crops, store.districts)
        try:
            return jsonify(
                app.extensions["fieldwise"]["predictor"].predict(
                    values.crop,
                    values.district,
                    values.rainfall,
                    values.temperature,
                    values.humidity,
                )
            )
        except Exception:
            app.logger.exception("Prediction failed")
            return jsonify(
                error="Prediction could not be completed. Please try again.",
                code="prediction_failed",
            ), 500
