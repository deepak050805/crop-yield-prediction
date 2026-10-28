"""Construct isolated Flask applications and initialize their local resources."""

import pickle
from pathlib import Path

from flask import Flask

from backend.config import FRONTEND_DIR, load_config
from backend.data_store import DataStore
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
    app.logger.setLevel(app.config["LOG_LEVEL"])
    store = DataStore(Path(app.config["DATA_PATH"]))
    with Path(app.config["MODEL_PATH"]).open("rb") as stream:
        model = pickle.load(stream)
    app.extensions["fieldwise"] = {"store": store, "predictor": PredictionService(store, model)}

    register_routes(app)
    return app
