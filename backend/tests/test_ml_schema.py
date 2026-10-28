import unittest

from backend.ml.schema import FEATURES, prepare_input


class FeatureSchemaTests(unittest.TestCase):
    def test_inference_features_exclude_target_and_year(self):
        frame = prepare_input(27.5, 735, 67, "Wheat", "Amritsar")
        self.assertEqual(list(frame.columns), FEATURES)
        self.assertEqual(frame.iloc[0].tolist(), [27.5, 735, 67, "Wheat", "Amritsar"])
        self.assertNotIn("Yield", FEATURES)
        self.assertNotIn("Year", FEATURES)
