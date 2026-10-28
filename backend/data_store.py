"""In-memory historical observations, loaded once per application instance."""

from copy import deepcopy
from pathlib import Path

import pandas as pd


class DataStore:
    def __init__(self, path: Path):
        self.frame = pd.read_csv(path)
        self.crops = sorted(self.frame["Crop"].unique().tolist())
        self.districts = sorted(self.frame["District"].unique().tolist())
        self.first_year = int(self.frame["Year"].min())
        self.last_year = int(self.frame["Year"].max())
        self._records = {
            (crop, district): group.sort_values("Year").to_dict(orient="records")
            for (crop, district), group in self.frame.groupby(["Crop", "District"])
        }
        self._bounds = {
            crop: {column: (float(group[column].min()), float(group[column].max()))
                   for column in ["Rainfall", "Temperature", "Humidity"]}
            for crop, group in self.frame.groupby("Crop")
        }

    def records(self, crop: str, district: str) -> list[dict]:
        """Return independent records so callers cannot mutate shared state."""
        return deepcopy(self._records.get((crop, district), []))

    def bounds(self, crop: str) -> dict[str, tuple[float, float]]:
        return dict(self._bounds[crop])
