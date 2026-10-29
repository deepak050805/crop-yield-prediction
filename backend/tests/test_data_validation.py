import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backend.ml.dataset import DATA_COLUMNS, validate_dataset
from backend.ml.validate_data import main as validate_main


def observation(**changes):
    return {
        "District": "Amritsar",
        "Crop": "Wheat",
        "Year": 2020,
        "Temperature": 27.5,
        "Rainfall": 735.0,
        "Humidity": 67.0,
        "Yield": 4.2,
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
            ("Rainfall", None),
            ("Humidity", float("inf")),
            ("Yield", -1),
            ("Temperature", "invalid"),
            ("Temperature", True),
            ("Rainfall", -1),
            ("Year", 2020.5),
            ("Year", 0),
            ("Crop", " Wheat"),
            ("District", ""),
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
        frame = pd.DataFrame(
            [
                observation(),
                observation(District="Ludhiana"),
                observation(Crop="Rice"),
            ]
        )
        self.assertEqual(len(validate_dataset(frame)), 3)

    def test_complex_observations_never_discard_imaginary_components(self):
        for field in ["Temperature", "Rainfall", "Humidity", "Year", "Yield"]:
            for value in [complex(1, 2), complex(1, 0)]:
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, "real numbers"):
                        validate_dataset(pd.DataFrame([observation(**{field: value})]))

    def test_validation_command_reports_coverage_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.csv"
            pd.DataFrame([observation(Year=2021), observation(Year=2020)]).to_csv(
                source, index=False
            )
            original = source.read_bytes()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                validate_main([str(source)])
            report = json.loads(output.getvalue())
            self.assertEqual(report["records"], 2)
            self.assertEqual(report["years"], [2020, 2021])
            self.assertEqual(report["records_per_year"], {"2020": 1, "2021": 1})
            self.assertEqual(report["observed_ranges"]["Yield"], {"min": 4.2, "max": 4.2})
            self.assertEqual(report["yield_unit"], "undocumented")
            self.assertEqual(source.read_bytes(), original)

    def test_validation_command_fails_without_printing_a_partial_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "bad.csv"
            pd.DataFrame([observation(Yield=-1)]).to_csv(source, index=False)
            for path in [source, source.with_name("missing.csv")]:
                stdout, stderr = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as error:
                        validate_main([str(path)])
                self.assertEqual(error.exception.code, 1)
                self.assertEqual(stdout.getvalue(), "")
                self.assertIn("Dataset validation failed:", stderr.getvalue())
