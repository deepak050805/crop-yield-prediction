import unittest

from backend.ml.schema import FEATURES, NUMERIC_BOUNDS, prepare_input, validate_numeric


class FeatureSchemaTests(unittest.TestCase):
    def test_inference_features_exclude_target_and_year(self):
        frame = prepare_input(27.5, 735, 67, "Wheat", "Amritsar")
        self.assertEqual(list(frame.columns), FEATURES)
        self.assertEqual(frame.iloc[0].tolist(), [27.5, 735, 67, "Wheat", "Amritsar"])
        self.assertNotIn("Yield", FEATURES)
        self.assertNotIn("Year", FEATURES)

    def test_weather_bounds_are_inclusive_and_parsed_consistently(self):
        for feature, (lower, upper) in NUMERIC_BOUNDS.items():
            self.assertEqual(validate_numeric(feature, str(lower)), lower)
            self.assertEqual(validate_numeric(feature, upper), upper)
            for invalid in [lower - 1, upper + 1, None, True, "NaN", "inf", "bad"]:
                with self.subTest(feature=feature, invalid=invalid):
                    with self.assertRaises(ValueError):
                        validate_numeric(feature, invalid)
