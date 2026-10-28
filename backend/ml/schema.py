"""The ordered inference contract shared by serving and training.

The bounds are broad plausibility limits for scenario requests, not observed
training ranges or a claim that conditions inside them are reliable estimates.
Rainfall is the dataset's reported period total (mm); its accumulation period
is unknown. Temperature is degrees Celsius and humidity is percent.
"""

import math

import pandas as pd

SCHEMA_VERSION = 1
FEATURES = ["Temperature", "Rainfall", "Humidity", "Crop", "District"]
NUMERIC = FEATURES[:3]
CATEGORICAL = FEATURES[3:]
NUMERIC_BOUNDS = {
    "Temperature": (-20.0, 60.0),
    "Rainfall": (0.0, 2000.0),
    "Humidity": (0.0, 100.0),
}
FEATURE_UNITS = {"Temperature": "C", "Rainfall": "mm", "Humidity": "%"}


def validate_numeric(feature, value):
    """Parse a finite weather value and enforce the documented plausibility bound."""
    if feature not in NUMERIC_BOUNDS:
        raise ValueError(f"Unknown numeric feature: {feature}")
    if isinstance(value, bool):
        raise ValueError(f"{feature} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{feature} must be a finite number.") from exc
    lower, upper = NUMERIC_BOUNDS[feature]
    if not math.isfinite(number) or not lower <= number <= upper:
        raise ValueError(f"{feature} must be between {lower:g} and {upper:g}.")
    return number


def prepare_input(temp, rainfall, humidity, crop, district):
    """Construct a prediction row with the exact training feature order."""
    return pd.DataFrame([[temp, rainfall, humidity, crop, district]], columns=FEATURES)
