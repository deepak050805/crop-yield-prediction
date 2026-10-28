"""Fail-closed validation for the local historical observation table."""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .schema import CATEGORICAL, FEATURES, NUMERIC, NUMERIC_BOUNDS
except ImportError:  # Direct training script execution.
    from schema import CATEGORICAL, FEATURES, NUMERIC, NUMERIC_BOUNDS

DATA_COLUMNS = ["District", "Crop", "Year", *NUMERIC, "Yield"]


def validate_dataset(frame):
    """Return a validated copy; never silently delete or repair observations."""
    if frame.empty:
        raise ValueError("Dataset must contain observations.")
    if frame.columns.duplicated().any() or set(frame.columns) != set(DATA_COLUMNS):
        raise ValueError(f"Dataset columns must be exactly {DATA_COLUMNS}.")
    data = frame.loc[:, DATA_COLUMNS].copy()
    if data.isna().any().any():
        raise ValueError("Dataset contains missing values.")
    for column in CATEGORICAL:
        if not data[column].map(
            lambda value: isinstance(value, str)
            and bool(value.strip())
            and value == value.strip()
            and all(character.isprintable() for character in value)
        ).all():
            raise ValueError(f"{column} must contain nonempty, trimmed text labels.")
    for column in [*NUMERIC, "Year", "Yield"]:
        if data[column].map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise ValueError(f"{column} must contain numbers, not booleans.")
        try:
            data[column] = pd.to_numeric(data[column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{column} contains invalid numeric values.") from exc
        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain finite numbers.")
    for column, (lower, upper) in NUMERIC_BOUNDS.items():
        if not data[column].between(lower, upper).all():
            raise ValueError(f"{column} is outside documented bounds {lower:g}..{upper:g}.")
    if not data["Year"].between(1, 9999).all() or data["Year"].mod(1).any():
        raise ValueError("Year must be an integer calendar year from 1 to 9999.")
    if data["Yield"].lt(0).any():
        raise ValueError("Yield must be nonnegative; its source unit is undocumented.")
    data["Year"] = data["Year"].astype(int)
    return data.sort_values("Year", kind="stable").reset_index(drop=True)


def load_dataset(path):
    """Read the UTF-8 CSV and validate every record before use."""
    return validate_dataset(pd.read_csv(Path(path), encoding="utf-8", skip_blank_lines=False))
