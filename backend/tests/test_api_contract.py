import unittest

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
