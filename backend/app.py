"""Compatible WSGI entry point; use create_app for isolated application tests."""

import sys
from pathlib import Path

# Keep the documented `python backend/app.py` entry point working.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import BASE_DIR  # noqa: E402, F401
from backend.factory import create_app  # noqa: E402
from backend.ml.schema import prepare_input  # noqa: E402, F401

app = create_app()

# Existing integrations import these names from backend.app.
store = app.extensions["fieldwise"]["store"]
df, crops, districts = store.frame, store.crops, store.districts
model = app.extensions["fieldwise"]["predictor"].model

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=app.config["PORT"], debug=False)
