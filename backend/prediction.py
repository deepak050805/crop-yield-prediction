"""Prediction and bounded weather scenarios, independent of HTTP handling."""

import math

import numpy as np
import pandas as pd

from backend.ml.schema import prepare_input


class PredictionService:
    def __init__(self, store, model):
        self.store = store
        self.model = model

    def predict(self, crop, district, rainfall, temp, humidity):
        bounds = self.store.bounds(crop)
        warnings = []
        for column, value in [("Rainfall", rainfall), ("Temperature", temp), ("Humidity", humidity)]:
            low, high = bounds[column]
            if not low <= value <= high:
                warnings.append(f"{column} is outside this crop's historical range ({low:g}-{high:g}); the estimate is less reliable.")
        options = [(rainfall, temp, humidity)]
        for direction in [1, -1]:
            values = []
            for column, value, step in [("Rainfall", rainfall, 100), ("Temperature", temp, 2), ("Humidity", humidity, 5)]:
                low, high = bounds[column]
                values.append(min(high, max(low, value + direction * step)))
            options.append(tuple(values))
        inputs = pd.concat([prepare_input(t, r, h, crop, district) for r, t, h in options], ignore_index=True)
        yields = np.asarray(self.model.predict(inputs))
        if yields.shape != (len(options),) or yields.dtype.kind not in "fiu":
            raise ValueError("Model predictions must contain one numeric yield per scenario.")
        if not np.isfinite(yields).all():
            raise ValueError("Model predictions must be finite.")
        if (yields < 0).any():
            warnings.append(
                "The model produced a negative yield estimate; estimates are less reliable."
            )
        best = int(yields.argmax())
        best_rain, best_temp, best_hum = options[best]
        predicted, best_yield = float(yields[0]), float(yields[best])
        gap = best_yield - predicted
        if not math.isfinite(gap):
            raise ValueError("The predicted yield gap must be finite.")
        return {
            "predicted": round(predicted, 3), "optimal": round(best_yield, 3),
            "gap": round(gap, 3), "best_rain": round(best_rain, 1),
            "best_temp": round(best_temp, 1), "best_hum": round(best_hum, 1),
            "crop": crop, "district": district, "warnings": warnings,
        }
