import unittest
from unittest.mock import Mock

import numpy as np

from backend.prediction import PredictionService


class PredictionServiceTests(unittest.TestCase):
    def setUp(self):
        self.store = Mock()
        self.store.bounds.return_value = {
            "Rainfall": (650, 800), "Temperature": (24, 30), "Humidity": (60, 75),
        }
        self.model = Mock()
        self.model.predict.return_value = np.array([4.0, 4.3, 3.8])
        self.service = PredictionService(self.store, self.model)

    def test_one_batch_preserves_schema_and_clips_alternatives(self):
        result = self.service.predict("Wheat", "Amritsar", 750, 29, 72)
        self.assertEqual(result["predicted"], 4)
        self.assertEqual(result["optimal"], 4.3)
        self.assertEqual(result["gap"], .3)
        self.model.predict.assert_called_once()
        frame = self.model.predict.call_args.args[0]
        self.assertEqual(list(frame.columns), ["Temperature", "Rainfall", "Humidity", "Crop", "District"])
        self.assertEqual(frame.iloc[1]["Temperature"], 30)
        self.assertEqual(frame.iloc[1]["Rainfall"], 800)
        self.assertEqual(frame.iloc[1]["Humidity"], 75)

    def test_extrapolated_input_is_preserved_and_warned(self):
        result = self.service.predict("Wheat", "Amritsar", 1500, 29, 72)
        frame = self.model.predict.call_args.args[0]
        self.assertEqual(frame.iloc[0]["Rainfall"], 1500)
        self.assertTrue(result["warnings"])
        self.assertEqual(frame.iloc[1]["Rainfall"], 800)
        self.assertEqual(frame.iloc[2]["Rainfall"], 800)

    def test_rejects_malformed_model_outputs(self):
        outputs = [
            None, 4.0, [], [4.0, 4.3], [4.0, 4.3, 3.8, 5.0],
            [[4.0, 4.3, 3.8]], [[4.0], [4.3], [3.8]],
            [True, False, True], ["4.0", "4.3", "3.8"],
            [4.0 + 1j, 4.3, 3.8], np.array([4.0, 4.3, 3.8], dtype=object),
        ]
        for output in outputs:
            with self.subTest(output=output):
                self.model.predict.return_value = output
                with self.assertRaisesRegex(ValueError, "one numeric yield per scenario"):
                    self.service.predict("Wheat", "Amritsar", 750, 29, 72)

    def test_rejects_nonfinite_model_outputs_in_every_scenario(self):
        for value in [np.nan, np.inf, -np.inf]:
            for index in range(3):
                with self.subTest(value=value, index=index):
                    output = np.array([4.0, 4.3, 3.8])
                    output[index] = value
                    self.model.predict.return_value = output
                    with self.assertRaisesRegex(ValueError, "predictions must be finite"):
                        self.service.predict("Wheat", "Amritsar", 750, 29, 72)

    def test_rejects_overflowing_gap_from_finite_yields(self):
        self.model.predict.return_value = np.array([-1e308, 1e308, 0])
        with self.assertRaisesRegex(ValueError, "yield gap must be finite"):
            self.service.predict("Wheat", "Amritsar", 750, 29, 72)

    def test_accepts_numeric_lists_and_integer_model_outputs(self):
        for output in [[4.0, 4.3, 3.8], np.array([4, 5, 3])]:
            with self.subTest(output=output):
                self.model.predict.return_value = output
                result = self.service.predict("Wheat", "Amritsar", 750, 29, 72)
                self.assertEqual(result["predicted"], 4.0)
                self.assertEqual(result["optimal"], output[1])
                self.assertEqual(result["warnings"], [])
