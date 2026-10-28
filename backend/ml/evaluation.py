"""Metrics and training-only baselines for matched temporal partitions."""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

try:
    from .schema import FEATURES
except ImportError:
    from schema import FEATURES


def scores(actual, predicted):
    """Return finite regression errors; undefined R2 is represented by JSON null."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or predicted.shape != actual.shape or not len(actual):
        raise ValueError("Metrics require equal nonempty one-dimensional arrays.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics cannot include non-finite observations or predictions.")
    informative_r2 = len(actual) >= 2 and np.ptp(actual) > 0
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 4),
        "rmse": round(float(root_mean_squared_error(actual, predicted)), 4),
        "r2": round(float(r2_score(actual, predicted)), 4) if informative_r2 else None,
    }


def crop_mean_predictions(earlier, later):
    """Estimate crop means exclusively from the supplied training observations."""
    if earlier.empty:
        raise ValueError("The crop-mean baseline requires training observations.")
    crop_means = earlier.groupby("Crop", observed=True)["Yield"].mean()
    return later["Crop"].map(crop_means).fillna(float(earlier["Yield"].mean())).to_numpy()


def grouped_scores(holdout, predicted, baseline, column):
    """Compare identical holdout rows, retaining errors and sample counts by group."""
    predicted, baseline = np.asarray(predicted), np.asarray(baseline)
    # Validate global dimensions before subgroup masks can hide an invalid result.
    scores(holdout["Yield"], predicted)
    scores(holdout["Yield"], baseline)
    result = {}
    for label in sorted(holdout[column].unique()):
        mask = holdout[column].eq(label).to_numpy()
        result[str(label)] = {
            **scores(holdout.loc[mask, "Yield"], predicted[mask]),
            "samples": int(mask.sum()),
            "baseline": scores(holdout.loc[mask, "Yield"], baseline[mask]),
        }
    return result


def fold_comparisons(estimator, development, folds):
    """Refit all preprocessing per fold and compare the same validation records."""
    comparisons = []
    for train_indices, validation_indices in folds:
        earlier = development.iloc[train_indices]
        later = development.iloc[validation_indices]
        if earlier.empty or later.empty or earlier["Year"].max() >= later["Year"].min():
            raise ValueError("Temporal folds require nonempty strictly earlier training years.")
        fitted = clone(estimator).fit(earlier[FEATURES], earlier["Yield"])
        predictions = fitted.predict(later[FEATURES])
        baseline = crop_mean_predictions(earlier, later)
        comparisons.append({
            "training_years": sorted(int(year) for year in earlier["Year"].unique()),
            "validation_years": sorted(int(year) for year in later["Year"].unique()),
            "training_samples": len(earlier),
            "validation_samples": len(later),
            "model": scores(later["Yield"], predictions),
            "baseline": scores(later["Yield"], baseline),
        })
    return comparisons
