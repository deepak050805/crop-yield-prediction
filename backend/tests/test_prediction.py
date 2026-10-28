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
