"""Typed request values and reusable validation, without Flask dependencies."""

from dataclasses import dataclass
from typing import Any, Collection, Mapping

from backend.ml.schema import validate_numeric


class ValidationError(ValueError):
    def __init__(self, message: str, field: str | None = None):
        super().__init__(message)
        self.field = field


@dataclass(frozen=True)
class Selection:
    crop: str
    district: str


@dataclass(frozen=True)
class PredictionInput(Selection):
    rainfall: float
    temperature: float
    humidity: float


def parse_selection(
    data: Mapping[str, Any], crops: Collection[str], districts: Collection[str]
) -> Selection:
    values = {}
    for field, choices in [("crop", crops), ("district", districts)]:
        value = data.get(field)
        if not isinstance(value, str) or value not in choices:
            raise ValidationError(f"Choose a valid {field}.", field)
        values[field] = value
    return Selection(**values)


def parse_prediction(
    data: Mapping[str, Any], crops: Collection[str], districts: Collection[str]
) -> PredictionInput:
    selection = parse_selection(data, crops, districts)
    values = {}
    for field, feature in [
        ("rainfall", "Rainfall"),
        ("temperature", "Temperature"),
        ("humidity", "Humidity"),
    ]:
        try:
            values[field] = float(validate_numeric(feature, data.get(field)))
        except ValueError as exc:
            raise ValidationError(str(exc), field) from exc
    return PredictionInput(selection.crop, selection.district, **values)
