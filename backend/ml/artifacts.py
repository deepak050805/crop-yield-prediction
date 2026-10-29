"""Publish and verify local, trusted model releases before deserializing pickle.

Hashes detect corruption and accidental mixing, not a malicious replacement of
both model and manifest. Never accept uploaded artifacts or unreviewed bundles.
"""

import hashlib
import json
import os
import pickle
import platform
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

try:
    from .schema import FEATURE_UNITS, FEATURES, NUMERIC_BOUNDS, SCHEMA_VERSION
except ImportError:
    from schema import FEATURE_UNITS, FEATURES, NUMERIC_BOUNDS, SCHEMA_VERSION

MANIFEST_NAME = "model_manifest.json"
TEXT_HASH_NORMALIZATION = "utf8-lf-v1"


class ArtifactValidationError(ValueError):
    """The configured artifact release cannot be safely used for inference."""


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def text_sha256(content: bytes) -> str:
    """Keep release verification stable across Git's Windows CRLF checkouts."""
    return sha256(content.replace(b"\r\n", b"\n"))


def feature_schema() -> dict[str, Any]:
    return {
        "version": SCHEMA_VERSION,
        "features": FEATURES,
        "numeric_bounds": {key: list(bounds) for key, bounds in NUMERIC_BOUNDS.items()},
        "units": FEATURE_UNITS,
    }


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def _json_bytes(document: dict[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _metadata_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"Duplicate artifact metadata key: {key}")
        document[key] = value
    return document


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Artifact metadata contains a non-finite JSON number: {value}")


def _read_metadata(content: bytes) -> Any:
    return json.loads(
        content, object_pairs_hook=_metadata_object, parse_constant=_reject_json_constant
    )


def _replace(path: Path, content: bytes) -> None:
    """Replace one file atomically; publish the integrity manifest last."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".release-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_pipeline(model: Any) -> None:
    if (
        not isinstance(model, Pipeline)
        or list(getattr(model, "feature_names_in_", [])) != FEATURES
        or getattr(model, "n_features_in_", None) != len(FEATURES)
    ):
        raise ArtifactValidationError("Model does not implement the ordered feature schema.")


def publish_bundle(
    model: Any, report: dict[str, Any], data_path: str | Path, models_dir: str | Path
) -> dict[str, Any]:
    """Publish a matched model/report/manifest release with reproducible metadata."""
    # Reject incomplete training before replacing any part of a previous release.
    _validate_pipeline(model)
    if report.get("features") != FEATURES or report.get("sklearn_version") != sklearn.__version__:
        raise ArtifactValidationError("Training report schema or runtime is incompatible.")
    models = Path(models_dir)
    models.mkdir(parents=True, exist_ok=True)
    model_bytes = pickle.dumps(model, protocol=5)
    model_hash = sha256(model_bytes)
    dataset_hash = text_sha256(Path(data_path).read_bytes())
    release_report = {
        **report,
        "model_version": model_hash[:16],
        "dataset_sha256": dataset_hash,
        "feature_schema": feature_schema(),
        "runtime_versions": runtime_versions(),
    }
    metrics_bytes = _json_bytes(release_report)
    manifest = {
        "format_version": 1,
        "model_version": model_hash[:16],
        "model_sha256": model_hash,
        "metrics_sha256": text_sha256(metrics_bytes),
        "text_hash_normalization": TEXT_HASH_NORMALIZATION,
        "dataset_sha256": dataset_hash,
        "feature_schema": feature_schema(),
        "runtime_versions": runtime_versions(),
    }
    _replace(models / "yield_model.pkl", model_bytes)
    _replace(models / "model_metrics.json", metrics_bytes)
    _replace(models / MANIFEST_NAME, _json_bytes(manifest))
    return release_report


def load_model_bundle(
    model_path: str | Path,
    metrics_path: str | Path,
    data_path: str | Path,
    manifest_path: str | Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Verify a trusted release in memory before any pickle code is executed."""
    model_path, metrics_path, data_path = map(Path, (model_path, metrics_path, data_path))
    manifest_path = Path(manifest_path) if manifest_path else model_path.with_name(MANIFEST_NAME)
    try:
        manifest = _read_metadata(manifest_path.read_bytes())
        model_bytes, metrics_bytes = model_path.read_bytes(), metrics_path.read_bytes()
        report = _read_metadata(metrics_bytes)
        if not isinstance(manifest, dict) or not isinstance(report, dict):
            raise ArtifactValidationError("Artifact metadata must contain JSON objects.")
        if (
            manifest.get("format_version") != 1
            or manifest.get("feature_schema") != feature_schema()
        ):
            raise ArtifactValidationError(
                "Artifact feature schema or manifest version is incompatible."
            )
        if manifest.get("text_hash_normalization") != TEXT_HASH_NORMALIZATION:
            raise ArtifactValidationError("Artifact text hash normalization is incompatible.")
        expected_hashes = {
            "model_sha256": sha256(model_bytes),
            "metrics_sha256": text_sha256(metrics_bytes),
            "dataset_sha256": text_sha256(data_path.read_bytes()),
        }
        if any(manifest.get(key) != value for key, value in expected_hashes.items()):
            raise ArtifactValidationError(
                "Artifact release hashes do not match the configured files."
            )
        versions = manifest.get("runtime_versions", {})
        if not isinstance(versions, dict):
            raise ArtifactValidationError("Artifact runtime versions are missing.")
        if versions.get("sklearn") != sklearn.__version__:
            raise ArtifactValidationError(
                "Artifact scikit-learn version does not match this runtime."
            )
        for package, current in {"numpy": np.__version__, "pandas": pd.__version__}.items():
            if versions.get(package) != current:
                raise ArtifactValidationError(
                    f"Artifact {package} version does not match this runtime."
                )
        if (
            str(versions.get("python", "")).split(".")[:2]
            != platform.python_version().split(".")[:2]
        ):
            raise ArtifactValidationError(
                "Artifact Python major/minor version does not match this runtime."
            )
        if (
            report.get("features") != FEATURES
            or report.get("feature_schema") != feature_schema()
            or report.get("dataset_sha256") != expected_hashes["dataset_sha256"]
            or report.get("sklearn_version") != sklearn.__version__
            or report.get("runtime_versions") != versions
            or report.get("model_version") != expected_hashes["model_sha256"][:16]
            or manifest.get("model_version") != report.get("model_version")
        ):
            raise ArtifactValidationError(
                "Report metadata does not describe this artifact release."
            )
        model = pickle.loads(model_bytes)
        _validate_pipeline(model)
        return model, report
    except ArtifactValidationError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        AttributeError,
        pickle.UnpicklingError,
        EOFError,
    ) as exc:
        raise ArtifactValidationError(
            "Artifact release is missing, unreadable, or invalid."
        ) from exc
