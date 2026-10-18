import json
import math
import unittest

from app.app import BASE_DIR, app, crops, df, model, prepare_input


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.payload = dict(crop="Wheat", district="Amritsar", rainfall="735",
                            temperature="27.5", humidity="67")

    def test_pages_and_assets(self):
        for path in ["/", "/predictor", "/history", "/static/dashboard.js",
                     "/static/dashboard.css", "/static/favicon.svg"]:
            with self.subTest(path=path):
                with self.client.get(path) as response:
                    self.assertEqual(response.status_code, 200)
        for path, text in [("/", b"Your season at a glance"),
                           ("/predictor", b"Weather conditions"),
                           ("/history", b"Export CSV")]:
            response = self.client.get(path)
            self.assertIn(text, response.data)
            self.assertNotIn(b"zapier", response.data.lower())

    def test_historical_records_are_filtered_and_sorted(self):
        response = self.client.post("/get_data", json={"crop": "Wheat", "district": "Amritsar"})
        self.assertEqual(response.status_code, 200)
        records = response.get_json()
        self.assertEqual(len(records), 10)
        self.assertEqual([r["Year"] for r in records], list(range(2015, 2025)))
        self.assertTrue(all(r["Crop"] == "Wheat" and r["District"] == "Amritsar" for r in records))
        for invalid in [{}, {"crop": "Unknown", "district": "Amritsar"}, ["Wheat"]]:
            self.assertEqual(self.client.post("/get_data", json=invalid).status_code, 400)

    def test_predictions_for_all_crops_use_matching_features(self):
        self.assertNotIn("Year", model.feature_names_in_)
        for crop in crops:
            with self.subTest(crop=crop):
                payload = {**self.payload, "crop": crop}
                response = self.client.post("/predict", data=payload)
                self.assertEqual(response.status_code, 200)
                result = response.get_json()
                expected = float(model.predict(prepare_input(27.5, 735, 67, crop, "Amritsar"))[0])
                self.assertAlmostEqual(result["predicted"], expected, places=3)
                self.assertTrue(math.isfinite(result["predicted"]))
                self.assertGreaterEqual(result["optimal"], result["predicted"])
                self.assertAlmostEqual(result["gap"], result["optimal"] - result["predicted"], delta=.0011)
                observed = df[df["Crop"] == crop]
                for key, column, original in [("best_rain", "Rainfall", 735),
                                               ("best_temp", "Temperature", 27.5),
                                               ("best_hum", "Humidity", 67)]:
                    if result[key] != original:
                        self.assertGreaterEqual(result[key], observed[column].min())
                        self.assertLessEqual(result[key], observed[column].max())

    def test_invalid_inputs_and_extrapolation(self):
        for field, value in [("rainfall", "nan"), ("temperature", "inf"),
                             ("humidity", "101"), ("rainfall", "-1"), ("rainfall", "1e308"),
                             ("crop", "Unknown"), ("temperature", "bad")]:
            with self.subTest(field=field, value=value):
                response = self.client.post("/predict", data={**self.payload, field: value})
                self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.post("/predict", data={}).status_code, 400)
        result = self.client.post("/predict", data={**self.payload, "rainfall": "1500"}).get_json()
        self.assertTrue(result["warnings"])

    def test_report_has_separate_years_and_crop_metrics(self):
        response = self.client.get("/model_info")
        self.assertEqual(response.status_code, 200)
        report = response.get_json()
        self.assertLess(max(report["development_years"]), min(report["holdout_years"]))
        self.assertTrue(set(report["validation_years"]).issubset(report["development_years"]))
        self.assertEqual(set(report["per_crop"]), set(crops))
        self.assertEqual(report["features"], list(model.feature_names_in_))
        self.assertTrue(all(math.isfinite(value) for value in report["holdout"].values()))
        saved = json.loads((BASE_DIR / "models" / "model_metrics.json").read_text())
        self.assertEqual(saved["model"], report["model"])


if __name__ == "__main__":
    unittest.main()
