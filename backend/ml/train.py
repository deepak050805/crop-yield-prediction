"""Reproducible temporal model selection and an untouched final-years report."""

import argparse
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .artifacts import publish_bundle
    from .dataset import load_dataset
    from .evaluation import crop_mean_predictions, fold_comparisons, grouped_scores, scores
    from .schema import CATEGORICAL, FEATURES, NUMERIC
    from .temporal import temporal_partitions
except ImportError:  # Keep direct script execution compatible.
    from artifacts import publish_bundle
    from dataset import load_dataset
    from evaluation import crop_mean_predictions, fold_comparisons, grouped_scores, scores
    from schema import CATEGORICAL, FEATURES, NUMERIC
    from temporal import temporal_partitions

BASE_DIR = Path(__file__).resolve().parents[1]


def build_pipeline():
    """Keep all learned preprocessing inside each fitted cross-validation clone."""
    preprocessing = ColumnTransformer([
        ("weather", StandardScaler(), NUMERIC),
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ], remainder="drop")
    return Pipeline([("prepare", preprocessing), ("regressor", Ridge())])


def candidate_grid(random_state=42, trees=200):
    return [
        {"regressor": [Ridge()], "regressor__alpha": [1.0, 10.0, 100.0]},
        {"regressor": [RandomForestRegressor(n_estimators=trees, random_state=random_state, n_jobs=1)],
         "regressor__max_depth": [4, 8], "regressor__min_samples_leaf": [5, 15]},
        {"regressor": [ExtraTreesRegressor(n_estimators=trees, random_state=random_state, n_jobs=1)],
         "regressor__max_depth": [4, 8], "regressor__min_samples_leaf": [5, 15]},
    ]


def select_candidate(results, fold_count):
    """Prefer the first predeclared candidate within one SE of the best CV MAE."""
    means, deviations = np.asarray(results["mean_test_score"]), np.asarray(results["std_test_score"])
    if fold_count < 1 or not len(means) or not np.isfinite(means).all() or not np.isfinite(deviations).all():
        raise ValueError("Model selection requires finite temporal validation scores.")
    best_index = int(np.argmax(means))
    threshold = means[best_index] - deviations[best_index] / np.sqrt(fold_count)
    return next(index for index, value in enumerate(means) if value >= threshold)


def train(data_path=None, models_dir=None, *, random_state=42, trees=200,
          holdout_years=2, validation_years=3, n_jobs=1):
    """Evaluate on development folds, report holdout, then refit for serving."""
    for name, value in [("trees", trees), ("n_jobs", n_jobs)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if isinstance(random_state, bool) or not isinstance(random_state, int) or not 0 <= random_state < 2**32:
        raise ValueError("random_state must be an integer from 0 to 2**32 - 1.")
    data_path = Path(data_path) if data_path is not None else BASE_DIR / "data/real_final_dataset.csv"
    models_dir = Path(models_dir) if models_dir is not None else BASE_DIR / "models"
    data = load_dataset(data_path)
    development, holdout, folds = temporal_partitions(data, holdout_years, validation_years)
    fold_years = [int(development.iloc[validation]["Year"].iloc[0]) for _, validation in folds]
    pipeline = build_pipeline()
    search = GridSearchCV(
        pipeline, candidate_grid(random_state, trees), cv=folds,
        scoring="neg_mean_absolute_error", n_jobs=n_jobs, refit=False, error_score="raise",
    )
    search.fit(development[FEATURES], development["Yield"])
    results = search.cv_results_
    selected_index = select_candidate(results, len(folds))
    selected_parameters = results["params"][selected_index]
    selected = clone(pipeline).set_params(**selected_parameters)
    selected.fit(development[FEATURES], development["Yield"])
    predictions = selected.predict(holdout[FEATURES])
    baseline_predictions = crop_mean_predictions(development, holdout)
    baseline_cv = [
        mean_absolute_error(development.iloc[validation]["Yield"],
                            crop_mean_predictions(development.iloc[earlier], development.iloc[validation]))
        for earlier, validation in folds
    ]
    report = {
        "report_version": 2,
        "model": type(selected.named_steps["regressor"]).__name__,
        "sklearn_version": sklearn.__version__,
        "features": FEATURES,
        "records": len(data),
        "development_samples": len(development), "holdout_samples": len(holdout),
        "development_years": sorted(int(year) for year in development["Year"].unique()),
        "holdout_years": sorted(int(year) for year in holdout["Year"].unique()),
        "training_years": sorted(int(year) for year in data["Year"].unique()),
        "validation_years": fold_years,
        "final_refit_on_all_records": True,
        "selection": "Lowest temporal validation MAE, preferring predeclared simpler candidates within one standard error.",
        "settings": {"random_state": random_state, "trees": trees, "n_jobs": n_jobs,
                     "holdout_years": holdout_years, "validation_years": validation_years},
        "cv_mae": round(float(-results["mean_test_score"][selected_index]), 4),
        "baseline_cv_mae": round(float(np.mean(baseline_cv)), 4),
        "cv_folds": fold_comparisons(selected, development, folds),
        "train": scores(development["Yield"], selected.predict(development[FEATURES])),
        "holdout": scores(holdout["Yield"], predictions),
        "baseline_holdout": scores(holdout["Yield"], baseline_predictions),
        "per_crop": grouped_scores(holdout, predictions, baseline_predictions, "Crop"),
        "per_district": grouped_scores(holdout, predictions, baseline_predictions, "District"),
        "per_year": grouped_scores(holdout, predictions, baseline_predictions, "Year"),
        "holdout_unseen_categories": {
            column: sorted(set(holdout[column]) - set(development[column])) for column in CATEGORICAL
        },
        "candidates": [
            {"model": type(parameters["regressor"]).__name__,
             "parameters": {key.removeprefix("regressor__"): value
                            for key, value in parameters.items() if key != "regressor"},
             "cv_mae": round(float(-results["mean_test_score"][index]), 4),
             "selected": index == selected_index}
            for index, parameters in enumerate(results["params"])
        ],
        "limitations": [
            "Metrics evaluate the model fitted on development years, before final refitting on all records.",
            "Overall R2 includes differences between crops; inspect per-crop errors and the crop-mean baseline.",
            "Dataset provenance, rainfall accumulation period, and yield units are undocumented.",
            "These scores do not establish real-world farm accuracy or a calibrated prediction interval.",
            "Small subgroup samples have uncertain metrics; R2 is null for constant or single observations.",
            "Weather comparisons are observational scenarios, not causal agronomic recommendations.",
        ],
    }
    selected.fit(data[FEATURES], data["Yield"])
    report = publish_bundle(selected, report, data_path, models_dir)
    print(json.dumps({key: report[key] for key in
                      ["model", "cv_mae", "baseline_cv_mae", "train", "holdout", "baseline_holdout"]}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Validated source CSV; defaults to the bundled dataset.")
    parser.add_argument("--output", type=Path, help="Release directory; defaults to backend/models.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--holdout-years", type=int, default=2)
    parser.add_argument("--validation-years", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    try:
        train(args.data, args.output, random_state=args.seed, trees=args.trees,
              holdout_years=args.holdout_years, validation_years=args.validation_years, n_jobs=args.jobs)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Training failed: {exc}\n")


if __name__ == "__main__":
    main()
