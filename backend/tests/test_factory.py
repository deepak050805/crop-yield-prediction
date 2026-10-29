import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.config import BASE_DIR
from backend.factory import create_app
from backend.ml.artifacts import ArtifactValidationError


class ApplicationFactoryTests(unittest.TestCase):
    def test_instances_have_isolated_configuration_and_resources(self):
        first = create_app({"TESTING": True, "APP_LABEL": "first"})
        second = create_app({"TESTING": True, "APP_LABEL": "second"})
        self.assertEqual(first.config["APP_LABEL"], "first")
        self.assertEqual(second.config["APP_LABEL"], "second")
        self.assertIsNot(
            first.extensions["fieldwise"]["store"], second.extensions["fieldwise"]["store"]
        )
        self.assertIsNot(
            first.extensions["fieldwise"]["predictor"], second.extensions["fieldwise"]["predictor"]
        )
        self.assertFalse(first.debug)

    def test_missing_resources_fail_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "absent"
            for key in ["DATA_PATH", "MODEL_PATH", "METRICS_PATH"]:
                with self.subTest(resource=key), self.assertRaises((OSError, ValueError)):
                    create_app({"TESTING": True, key: missing})

    def test_routes_and_assets_work_in_a_fresh_app(self):
        client = create_app({"TESTING": True}).test_client()
        for path in ["/", "/predictor", "/history", "/model_info", "/static/js/dashboard.js"]:
            with self.subTest(path=path), client.get(path) as response:
                self.assertEqual(response.status_code, 200)

    def test_configured_release_corruption_prevents_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory)
            for name in ["yield_model.pkl", "model_metrics.json", "model_manifest.json"]:
                shutil.copyfile(BASE_DIR / "models" / name, release / name)
            (release / "model_metrics.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ArtifactValidationError):
                create_app(
                    {
                        "TESTING": True,
                        "MODEL_PATH": release / "yield_model.pkl",
                        "METRICS_PATH": release / "model_metrics.json",
                    }
                )

    def test_report_uses_the_verified_snapshot_without_reopening_files(self):
        app = create_app({"TESTING": True})
        expected = app.extensions["fieldwise"]["model_info"]["model_version"]
        with patch("pathlib.Path.read_text", side_effect=AssertionError("unexpected disk read")):
            response = app.test_client().get("/model_info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["model_version"], expected)
