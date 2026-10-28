"""Expanding-window evaluation with an untouched final-year holdout."""

import numpy as np


def temporal_partitions(data, holdout_years=2, validation_years=3):
    """Split whole calendar years, keeping every observation in its partition.

    Fold indices address the returned development frame, never the source index.
    At least one year precedes the earliest validation year.
    """
    for name, count in [("holdout_years", holdout_years), ("validation_years", validation_years)]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"{name} must be a positive integer.")
    ordered = data.sort_values("Year", kind="stable").reset_index(drop=True)
    years = sorted(ordered["Year"].unique())
    required = holdout_years + validation_years + 1
    if len(years) < required:
        raise ValueError(f"At least {required} distinct years are required for temporal evaluation.")
    cutoff = years[-holdout_years]
    development = ordered[ordered["Year"].lt(cutoff)].reset_index(drop=True)
    holdout = ordered[ordered["Year"].ge(cutoff)].reset_index(drop=True)
    folds = []
    for year in sorted(development["Year"].unique())[-validation_years:]:
        train_indices = np.flatnonzero(development["Year"].lt(year))
        validation_indices = np.flatnonzero(development["Year"].eq(year))
        folds.append((train_indices, validation_indices))
    return development, holdout, folds
