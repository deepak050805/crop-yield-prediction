import tempfile
import unittest
from pathlib import Path

from backend.factory import create_app


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
            for key in ["DATA_PATH", "MODEL_PATH"]:
                with self.subTest(resource=key), self.assertRaises((OSError, ValueError)):
                    create_app({"TESTING": True, key: missing})

    def test_routes_and_assets_work_in_a_fresh_app(self):
        client = create_app({"TESTING": True}).test_client()
        for path in ["/", "/predictor", "/history", "/model_info", "/static/js/dashboard.js"]:
            with self.subTest(path=path), client.get(path) as response:
                self.assertEqual(response.status_code, 200)
