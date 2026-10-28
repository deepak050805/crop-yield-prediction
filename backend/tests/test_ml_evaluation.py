import unittest

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from backend.ml.evaluation import crop_mean_predictions, fold_comparisons, grouped_scores, scores


class RecordingRegressor(RegressorMixin, BaseEstimator):
    fitted_indices = []

    def fit(self, features, target):
        type(self).fitted_indices.append(set(features.index))
        self.mean_ = float(target.mean())
        return self

    def predict(self, features):
        return np.full(len(features), self.mean_)


class EvaluationTests(unittest.TestCase):
    def test_metrics_reject_nonfinite_or_incompatible_arrays(self):
        self.assertEqual(scores([1, 3], [2, 2]), {"mae": 1.0, "rmse": 1.0, "r2": 0.0})
        self.assertIsNone(scores([1], [2])["r2"])
        self.assertIsNone(scores([2, 2], [2, 2])["r2"])
        for actual, predicted in [([], []), ([1], []), ([1], [np.nan]), ([[1]], [[1]])]:
            with self.assertRaises(ValueError):
                scores(actual, predicted)

    def test_crop_baseline_never_reads_holdout_yields_and_handles_unseen_crops(self):
        earlier = pd.DataFrame({"Crop": ["Wheat", "Wheat", "Rice"], "Yield": [2, 4, 9]})
        later = pd.DataFrame({"Crop": ["Wheat", "Unknown"], "Yield": [-1000, 1000]})
        np.testing.assert_allclose(crop_mean_predictions(earlier, later), [3, 5])
        later["Yield"] = [10_000, -10_000]
        np.testing.assert_allclose(crop_mean_predictions(earlier, later), [3, 5])

    def test_subgroup_metrics_compare_identical_rows_with_sample_counts(self):
        later = pd.DataFrame({"District": ["A", "A", "B"], "Yield": [1, 3, 5]}, index=[9, 7, 5])
        result = grouped_scores(later, [2, 2, 4], [1, 2, 3], "District")
        self.assertEqual(result["A"]["samples"], 2)
        self.assertEqual(result["A"]["mae"], 1)
        self.assertEqual(result["A"]["baseline"]["mae"], 0.5)
        self.assertIsNone(result["B"]["r2"])
        self.assertEqual(sum(group["samples"] for group in result.values()), len(later))

    def test_each_temporal_baseline_and_estimator_fit_use_only_earlier_rows(self):
        frame = pd.DataFrame({
            "Year": [2018, 2019, 2020], "Yield": [1, 3, 100], "Crop": ["Wheat"] * 3,
            "District": ["A"] * 3, "Temperature": [20, 21, 22],
            "Rainfall": [700] * 3, "Humidity": [60] * 3,
        })
        RecordingRegressor.fitted_indices = []
        folds = [(np.array([0]), np.array([1])), (np.array([0, 1]), np.array([2]))]
        result = fold_comparisons(RecordingRegressor(), frame, folds)
        self.assertEqual(RecordingRegressor.fitted_indices, [{0}, {0, 1}])
        self.assertEqual([fold["baseline"]["mae"] for fold in result], [2, 98])
        self.assertTrue(all(fold["model"] == fold["baseline"] for fold in result))
        with self.assertRaises(ValueError):
            fold_comparisons(RecordingRegressor(), frame, [(np.array([2]), np.array([0]))])
