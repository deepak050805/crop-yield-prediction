import unittest

import pandas as pd

from backend.ml.dataset import DATA_COLUMNS, validate_dataset


def observation(**changes):
    return {
        "District": "Amritsar", "Crop": "Wheat", "Year": 2020,
        "Temperature": 27.5, "Rainfall": 735.0, "Humidity": 67.0, "Yield": 4.2,
        **changes,
    }


class DatasetValidationTests(unittest.TestCase):
    def test_preserves_observations_without_mutating_caller(self):
        frame = pd.DataFrame([observation(Year="2021"), observation(Year="2020")])
        result = validate_dataset(frame)
        self.assertEqual(list(result.columns), DATA_COLUMNS)
        self.assertEqual(result["Year"].tolist(), [2020, 2021])
        self.assertEqual(frame["Year"].tolist(), ["2021", "2020"])

    def test_fails_instead_of_dropping_bad_records(self):
        for field, value in [
            ("Rainfall", None), ("Humidity", float("inf")), ("Yield", -1),
            ("Temperature", "invalid"), ("Temperature", True), ("Rainfall", -1),
            ("Year", 2020.5), ("Year", 0), ("Crop", " Wheat"), ("District", ""),
            ("District", "Amritsar\n"),
        ]:
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    validate_dataset(pd.DataFrame([observation(**{field: value})]))

    def test_rejects_empty_missing_and_unexpected_columns(self):
        for frame in [
            pd.DataFrame(columns=DATA_COLUMNS),
            pd.DataFrame([observation()]).drop(columns="Humidity"),
            pd.DataFrame([observation(Leakage=4.2)]),
        ]:
            with self.assertRaises(ValueError):
                validate_dataset(frame)

    def test_rejects_exact_duplicates_and_conflicting_observation_keys(self):
        for second in [observation(), observation(Yield=9.0), observation(Year="2020")]:
            with self.subTest(second=second):
                with self.assertRaisesRegex(ValueError, "duplicate|conflicting"):
                    validate_dataset(pd.DataFrame([observation(), second]))

    def test_distinct_district_or_crop_is_a_separate_observation(self):
        frame = pd.DataFrame([
            observation(), observation(District="Ludhiana"), observation(Crop="Rice"),
        ])
        self.assertEqual(len(validate_dataset(frame)), 3)
