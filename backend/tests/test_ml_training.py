import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from backend.ml.artifacts import load_model_bundle
from backend.ml.schema import FEATURES, NUMERIC
from backend.ml.train import build_pipeline, select_candidate, train


class TrainingTests(unittest.TestCase):
    def test_preprocessing_learns_only_fitted_training_categories_and_ranges(self):
        earlier = pd.DataFrame([
            [20, 700, 60, "Wheat", "A"], [22, 800, 70, "Rice", "B"],
        ], columns=FEATURES)
        pipeline = build_pipeline().fit(earlier, [2, 4])
        scaler = pipeline.named_steps["prepare"].named_transformers_["weather"]
        encoder = pipeline.named_steps["prepare"].named_transformers_["categories"]
        np.testing.assert_array_equal(scaler.mean_, [21, 750, 65])
        future = pd.DataFrame([[55, 1800, 95, "Potato", "C"]], columns=FEATURES)
        self.assertTrue(np.isfinite(pipeline.predict(future)).all())
        np.testing.assert_array_equal(scaler.mean_, earlier[NUMERIC].mean())
        self.assertNotIn("Potato", encoder.categories_[0])
        self.assertNotIn("C", encoder.categories_[1])

    def test_one_standard_error_selection_uses_declared_order_and_rejects_nan(self):
        results = {"mean_test_score": [-0.25, -0.24, -0.5], "std_test_score": [0.1, 0.03, 0.1]}
        self.assertEqual(select_candidate(results, 3), 0)
        with self.assertRaises(ValueError):
            select_candidate({"mean_test_score": [np.nan], "std_test_score": [0.0]}, 3)

    def test_holdout_targets_do_not_affect_selection_or_fold_baselines(self):
        rows = [
            {"District": district, "Crop": crop, "Year": year,
             "Temperature": 20 + (year - 2015), "Rainfall": 700 + (year - 2015),
             "Humidity": 60, "Yield": crop_index + 2 + (year - 2015) * 0.02}
            for year in range(2015, 2025)
            for district in ["A", "B"]
            for crop_index, crop in enumerate(["Wheat", "Rice"])
        ]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            data = pd.DataFrame(rows)
            reports = []
            for index, addition in enumerate([0, 100]):
                source = directory / f"source-{index}.csv"
                changed = data.copy()
                changed.loc[changed["Year"].ge(2023), "Yield"] += addition
                changed.to_csv(source, index=False)
                output = directory / f"release-{index}"
                grid = [{"regressor": [Ridge()], "regressor__alpha": [1.0, 10.0]}]
                with patch("backend.ml.train.candidate_grid", return_value=grid), contextlib.redirect_stdout(io.StringIO()):
                    report = train(source, output, trees=1)
                model, loaded_report = load_model_bundle(
                    output / "yield_model.pkl", output / "model_metrics.json", source
                )
                self.assertEqual(loaded_report, report)
                np.testing.assert_array_equal(
                    model.named_steps["prepare"].named_transformers_["weather"].mean_,
                    changed[NUMERIC].mean(),
                )
                self.assertEqual(sum(item["samples"] for item in report["per_crop"].values()), 8)
                reports.append(report)
            for key in ["candidates", "cv_mae", "baseline_cv_mae", "cv_folds"]:
                self.assertEqual(reports[0][key], reports[1][key])
            self.assertLess(reports[0]["holdout"]["mae"], reports[1]["holdout"]["mae"])

    def test_invalid_training_options_fail_before_reading_files(self):
        for settings in [{"trees": 0}, {"n_jobs": -1}, {"random_state": -1}, {"random_state": True}]:
            with self.assertRaises(ValueError):
                train(data_path="missing.csv", **settings)
