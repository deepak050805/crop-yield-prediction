"""Train with year-separated validation and publish an honest holdout report."""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURES = ["Temperature", "Rainfall", "Humidity", "Crop", "District"]
NUMERIC = FEATURES[:3]


def scores(actual, predicted):
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 4),
        "rmse": round(float(root_mean_squared_error(actual, predicted)), 4),
        "r2": round(float(r2_score(actual, predicted)), 4),
    }


def train():
    data_path = BASE_DIR / "data" / "real_final_dataset.csv"
    data = pd.read_csv(data_path).drop_duplicates()
    for column in [*NUMERIC, "Year", "Yield"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=[*FEATURES, "Year", "Yield"])
    data = data[data["Yield"].ge(0)].sort_values("Year").reset_index(drop=True)
    years = sorted(data["Year"].unique())
    if len(years) < 6:
        raise ValueError("At least six distinct years are required for temporal evaluation.")
    development = data[data["Year"] < years[-2]].reset_index(drop=True)
    holdout = data[data["Year"] >= years[-2]].reset_index(drop=True)
    folds = []
    fold_years = sorted(development["Year"].unique())[-3:]
    for year in fold_years:
        folds.append((np.flatnonzero(development["Year"].lt(year)),
                      np.flatnonzero(development["Year"].eq(year))))

    preprocessing = ColumnTransformer([
        ("weather", StandardScaler(), NUMERIC),
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         ["Crop", "District"]),
    ])
    pipeline = Pipeline([("prepare", preprocessing), ("regressor", Ridge())])
    search = GridSearchCV(
        pipeline,
        [
            {"regressor": [Ridge()], "regressor__alpha": [1.0, 10.0, 100.0]},
            {"regressor": [RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1)],
             "regressor__max_depth": [4, 8], "regressor__min_samples_leaf": [5, 15]},
            {"regressor": [ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=1)],
             "regressor__max_depth": [4, 8], "regressor__min_samples_leaf": [5, 15]},
        ],
        cv=folds, scoring="neg_mean_absolute_error", n_jobs=1, refit=False, error_score="raise",
    )
    search.fit(development[FEATURES], development["Yield"])
    results = search.cv_results_
    best_index = int(np.argmax(results["mean_test_score"]))
    # Prefer simpler candidates within one standard error of the lowest MAE.
    threshold = (results["mean_test_score"][best_index]
                 - results["std_test_score"][best_index] / np.sqrt(len(folds)))
    selected_index = next(i for i, value in enumerate(results["mean_test_score"]) if value >= threshold)
    selected = clone(pipeline).set_params(**results["params"][selected_index])
    selected.fit(development[FEATURES], development["Yield"])
    predictions = selected.predict(holdout[FEATURES])
    crop_means = development.groupby("Crop")["Yield"].mean()
    baseline_predictions = holdout["Crop"].map(crop_means).fillna(development["Yield"].mean())
    baseline_cv = []
    for train_indices, val_indices in folds:
        earlier, later = development.iloc[train_indices], development.iloc[val_indices]
        mean = earlier.groupby("Crop")["Yield"].mean()
        baseline_cv.append(mean_absolute_error(later["Yield"], later["Crop"].map(mean).fillna(earlier["Yield"].mean())))

    report = {
        "model": type(selected.named_steps["regressor"]).__name__,
        "sklearn_version": sklearn.__version__,
        "features": FEATURES,
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "records": len(data),
        "development_years": sorted(int(y) for y in development["Year"].unique()),
        "holdout_years": sorted(int(y) for y in holdout["Year"].unique()),
        "validation_years": [int(y) for y in fold_years],
        "selection": "Lowest temporal validation MAE, preferring simpler candidates within one standard error.",
        "cv_mae": round(float(-results["mean_test_score"][selected_index]), 4),
        "baseline_cv_mae": round(float(np.mean(baseline_cv)), 4),
        "train": scores(development["Yield"], selected.predict(development[FEATURES])),
        "holdout": scores(holdout["Yield"], predictions),
        "baseline_holdout": scores(holdout["Yield"], baseline_predictions),
        "per_crop": {
            crop: scores(holdout.loc[mask, "Yield"], predictions[mask])
            for crop in sorted(holdout["Crop"].unique())
            for mask in [holdout["Crop"].eq(crop).to_numpy()]
        },
        "candidates": [
            {"model": type(parameters["regressor"]).__name__,
             "parameters": {key.removeprefix("regressor__"): value
                            for key, value in parameters.items() if key != "regressor"},
             "cv_mae": round(float(-results["mean_test_score"][i]), 4)}
            for i, parameters in enumerate(results["params"])
        ],
        "limitations": [
            "Metrics evaluate the model fitted on development years, before final refitting on all records.",
            "Overall R2 includes differences between crops; inspect per-crop errors and the crop-mean baseline.",
            "Dataset provenance is undocumented. These scores do not establish real-world farm accuracy.",
            "Weather comparisons are observational scenarios, not causal agronomic recommendations.",
        ],
    }
    selected.fit(data[FEATURES], data["Yield"])
    models = BASE_DIR / "models"
    models.mkdir(exist_ok=True)
    with (models / "yield_model.pkl").open("wb") as stream:
        pickle.dump(selected, stream)
    (models / "model_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ["model", "cv_mae", "baseline_cv_mae", "train", "holdout", "baseline_holdout"]}, indent=2))


if __name__ == "__main__":
    train()
