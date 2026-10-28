"""The ordered inference contract shared by serving and training.

The bounds are broad plausibility limits for scenario requests, not observed
training ranges or a claim that conditions inside them are reliable estimates.
Rainfall is the dataset's reported period total (mm); its accumulation period
is unknown. Temperature is degrees Celsius and humidity is percent.
"""

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


def prepare_input(temp, rainfall, humidity, crop, district):
    """Construct a prediction row with the exact training feature order."""
    return pd.DataFrame([[temp, rainfall, humidity, crop, district]], columns=FEATURES)
