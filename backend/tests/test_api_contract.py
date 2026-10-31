import unittest
from unittest.mock import patch

import numpy as np
from werkzeug.datastructures import MultiDict

from backend.factory import create_app


class ApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app({"TESTING": True})
        cls.payload = dict(
            crop="Wheat", district="Amritsar", rainfall=700, temperature=27, humidity=70
        )

    def setUp(self):
        self.client = self.app.test_client()

    def test_json_and_form_predictions_are_equivalent(self):
        json_response = self.client.post("/predict", json=self.payload)
        form_response = self.client.post("/predict", data=self.payload)
        self.assertEqual(json_response.status_code, 200)
        self.assertEqual(json_response.get_json(), form_response.get_json())

    def test_duplicate_fields_are_http_errors(self):
        for route in ["/predict", "/get_data"]:
            response = self.client.post(
                route, data='{"crop":"Wheat","crop":"Rice"}', content_type="application/json"
            )
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.get_json()["field"], "crop")
        response = self.client.post(
            "/predict", data=MultiDict([*self.payload.items(), ("rainfall", 100)])
        )
        self.assertEqual(response.status_code, 400)

    def test_media_type_and_body_limits_have_structured_errors(self):
        response = self.client.post("/predict", data="value", content_type="text/plain")
        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.get_json()["code"], "unsupported_media_type")
        for path in ["/predict", "/get_data"]:
            response = self.client.post(path, data=" " * 20000, content_type="application/json")
            self.assertEqual(response.status_code, 413)
            self.assertEqual(response.get_json()["code"], "request_entity_too_large")

    def test_errors_preserve_protocol_headers_and_hide_static_artifacts(self):
        response = self.client.get("/predict")
        self.assertEqual(response.status_code, 405)
        self.assertIn("POST", response.headers["Allow"])
        for path in ["/missing", "/static/../backend/models/yield_model.pkl", "/static/../.env"]:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 404)
            self.assertTrue(response.is_json)

    def test_invalid_model_outputs_return_structured_prediction_errors(self):
        model = self.app.extensions["fieldwise"]["predictor"].model
        for output in [
            np.array([4.0, np.nan, 3.0]),
            np.array([4.0, np.inf, 3.0]),
            np.array([[4.0], [5.0], [3.0]]),
            np.array([-1e308, 1e308, 0]),
        ]:
            with self.subTest(output=output):
                with patch.object(model, "predict", return_value=output):
                    with self.assertLogs(self.app.logger, level="ERROR"):
                        response = self.client.post("/predict", json=self.payload)
                self.assertEqual(response.status_code, 500)
                self.assertEqual(response.get_json(), {
                    "error": "Prediction could not be completed. Please try again.",
                    "code": "prediction_failed",
                    "request_id": response.headers["X-Request-ID"],
                })

    def test_negative_model_outputs_preserve_estimates_and_warn(self):
        model = self.app.extensions["fieldwise"]["predictor"].model
        with patch.object(model, "predict", return_value=np.array([-4.0, -3.5, -5.0])):
            response = self.client.post("/predict", json=self.payload)
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["predicted"], -4.0)
        self.assertEqual(result["optimal"], -3.5)
        self.assertEqual(result["gap"], 0.5)
        self.assertTrue(any("negative yield estimate" in warning for warning in result["warnings"]))
