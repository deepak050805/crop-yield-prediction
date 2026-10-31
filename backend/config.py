"""Validated configuration shared by local startup and the app factory."""

import os
from pathlib import Path
from typing import Any, Mapping

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"


def _boolean(env: Mapping[str, str], key: str) -> bool:
    value = env.get(key, "false").strip().lower()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError(f"{key} must be true, false, 1, or 0.")
    return value in {"true", "1"}


def load_config(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if environ is None else environ
    try:
        port = int(env.get("PORT", "10000"))
    except ValueError as exc:
        raise ValueError("PORT must be an integer from 1 to 65535.") from exc
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be an integer from 1 to 65535.")
    log_level = env.get("FIELDWISE_LOG_LEVEL", "INFO").upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError("FIELDWISE_LOG_LEVEL must be a standard logging level.")
    trusted_hosts = None
    if "FIELDWISE_TRUSTED_HOSTS" in env:
        trusted_hosts = [host.strip() for host in env["FIELDWISE_TRUSTED_HOSTS"].split(",")]
        if any(
            not host.strip(".")
            or "/" in host
            or "*" in host
            or any(character.isspace() for character in host)
            for host in trusted_hosts
        ):
            raise ValueError("FIELDWISE_TRUSTED_HOSTS must contain comma-separated host names.")
    return {
        "PORT": port,
        "LOG_LEVEL": log_level,
        "DATA_PATH": Path(
            env.get("FIELDWISE_DATA_PATH", str(BASE_DIR / "data" / "real_final_dataset.csv"))
        ),
        "MODEL_PATH": Path(
            env.get("FIELDWISE_MODEL_PATH", str(BASE_DIR / "models" / "yield_model.pkl"))
        ),
        "METRICS_PATH": Path(
            env.get("FIELDWISE_METRICS_PATH", str(BASE_DIR / "models" / "model_metrics.json"))
        ),
        "DEBUG": False,
        "TESTING": False,
        "HTTPS": _boolean(env, "FIELDWISE_HTTPS"),
        "TRUST_PROXY": _boolean(env, "FIELDWISE_TRUST_PROXY"),
        "TRUSTED_HOSTS": trusted_hosts,
        # Requests contain five short scalar values; bound parsing work and memory.
        "MAX_CONTENT_LENGTH": 16 * 1024,
        "MAX_FORM_MEMORY_SIZE": 16 * 1024,
        "MAX_FORM_PARTS": 16,
    }
