"""Bounded, unambiguous HTTP payload decoding for the scalar API forms."""

import json

from flask import request
from werkzeug.exceptions import UnsupportedMediaType

from backend.validation import ValidationError


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("Send each field only once.", key)
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValidationError("JSON numbers must be finite.")


def read_payload(*, allow_form=False):
    content_encoding = request.headers.get("Content-Encoding", "identity").strip().lower()
    if content_encoding not in {"", "identity"}:
        raise UnsupportedMediaType("Encoded request bodies are not supported.")
    if request.is_json:
        try:
            data = json.loads(
                request.get_data(cache=False),
                object_pairs_hook=_unique_object,
                parse_constant=_invalid_constant,
            )
        except (ValueError, UnicodeError, RecursionError) as exc:
            if isinstance(exc, ValidationError):
                raise
            raise ValidationError("Send a valid JSON object.") from exc
        if not isinstance(data, dict):
            raise ValidationError("Send a JSON object.")
        return data
    if allow_form and request.mimetype in {
        "application/x-www-form-urlencoded",
        "multipart/form-data",
    }:
        if request.files:
            raise ValidationError("File uploads are not accepted.")
        data = {}
        for field, values in request.form.lists():
            if len(values) != 1:
                raise ValidationError("Send each field only once.", field)
            data[field] = values[0]
        return data
    raise UnsupportedMediaType(
        "Use application/json" + (" or an HTML form." if allow_form else ".")
    )
