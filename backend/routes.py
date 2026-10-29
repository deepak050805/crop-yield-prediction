"""HTTP adapters for the historical explorer and prediction service."""

import json
from pathlib import Path

from flask import jsonify, render_template, request

from backend.request_data import read_payload
from backend.validation import parse_prediction, parse_selection


def register_routes(app):
    store = app.extensions["fieldwise"]["store"]

    @app.get("/health/live")
    def health_live():
        return jsonify(status="alive")

    @app.get("/health/ready")
    def health_ready():
        resources = app.extensions.get("fieldwise", {})
        predictor = resources.get("predictor")
        loaded_store = resources.get("store")
        ready = (
            loaded_store is not None
            and not loaded_store.frame.empty
            and predictor is not None
            and predictor.model is not None
        )
        return jsonify(status="ready" if ready else "not_ready"), 200 if ready else 503

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
        data = read_payload()
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
        # Preserve the existing empty-form 400 response while accepting JSON too.
        data = (
            {}
            if not request.mimetype and not request.content_length
            else read_payload(allow_form=True)
        )
        values = parse_prediction(data, store.crops, store.districts)
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
