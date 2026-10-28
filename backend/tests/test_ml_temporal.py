import unittest

import pandas as pd

from backend.ml.temporal import temporal_partitions


class TemporalPartitionTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "Year": [year for year in range(2015, 2025) for _ in range(2)],
            "Yield": list(range(20)),
        }).sample(frac=1, random_state=42)

    def test_whole_year_holdout_and_expanding_folds_have_no_future_rows(self):
        development, holdout, folds = temporal_partitions(self.frame)
        self.assertEqual(holdout["Year"].unique().tolist(), [2023, 2024])
        self.assertEqual(len(development) + len(holdout), len(self.frame))
        self.assertTrue(set(development["Yield"]).isdisjoint(holdout["Yield"]))
        self.assertEqual(len(folds), 3)
        for train_indices, validation_indices in folds:
            earlier, later = development.iloc[train_indices], development.iloc[validation_indices]
            self.assertLess(earlier["Year"].max(), later["Year"].min())
            self.assertEqual(later["Year"].nunique(), 1)
            self.assertEqual(len(later), 2)
            self.assertTrue(set(train_indices).isdisjoint(validation_indices))

    def test_short_history_and_invalid_configuration_fail(self):
        with self.assertRaises(ValueError):
            temporal_partitions(self.frame[self.frame["Year"].ge(2020)])
        for count in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):
                temporal_partitions(self.frame, holdout_years=count)

    def test_calendar_gaps_do_not_move_future_observations_into_training(self):
        frame = self.frame.copy()
        frame["Year"] = frame["Year"].map(lambda year: year + 10 if year >= 2020 else year)
        development, holdout, folds = temporal_partitions(frame)
        self.assertLess(development["Year"].max(), holdout["Year"].min())
        self.assertLess(development.iloc[folds[0][0]]["Year"].max(), 2030)
