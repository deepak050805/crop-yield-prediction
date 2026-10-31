"""Construct isolated Flask applications and initialize their local resources."""

from pathlib import Path

from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix

from backend.config import FRONTEND_DIR, load_config
from backend.data_store import DataStore
from backend.http import register_http_errors
from backend.ml.artifacts import load_model_bundle
from backend.prediction import PredictionService
from backend.routes import register_routes


def create_app(config=None):
    app = Flask(
        __name__,
        template_folder=str(FRONTEND_DIR / "templates"),
        static_folder=str(FRONTEND_DIR / "static"),
        static_url_path="/static",
    )
    app.config.update(load_config())
    app.config.update(config or {})
    if app.config["TRUST_PROXY"]:
        # Trust exactly one controlled proxy for client address and scheme only.
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=0, x_port=0, x_prefix=0)
    app.logger.setLevel(app.config["LOG_LEVEL"])
    store = DataStore(Path(app.config["DATA_PATH"]))
    model, report = load_model_bundle(
        app.config["MODEL_PATH"], app.config["METRICS_PATH"], app.config["DATA_PATH"]
    )
    app.extensions["fieldwise"] = {
        "store": store,
        "predictor": PredictionService(store, model),
        "model_info": report,
    }

    register_routes(app)
    register_http_errors(app)
    return app
