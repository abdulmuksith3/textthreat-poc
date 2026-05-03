"""TextThreat threat-event schema construction and validation."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator, FormatChecker

from .constants import ALL_HARM_TYPES, DEFAULT_THRESHOLD, MODEL_VERSION
from .utils import now_utc_iso, read_json, repo_path


SCHEMA_PATH = repo_path("schema", "textthreat_event_schema.json")


def hash_text(text: str) -> str:
    """Return a SHA-256 hash of input text for traceability without raw text storage."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _validator(schema_path: str = str(SCHEMA_PATH)) -> Draft7Validator:
    schema = read_json(Path(schema_path))
    return Draft7Validator(schema, format_checker=FormatChecker())


def validate_event(event: dict[str, Any], schema_path: Path | None = None) -> dict[str, Any]:
    """Validate an event and return it unchanged when valid."""
    validator = _validator(str(schema_path or SCHEMA_PATH))
    errors = sorted(validator.iter_errors(event), key=lambda error: list(error.path))
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.path) or "<root>"
        raise ValueError(f"Invalid TextThreat event at {location}: {first.message}")
    return event


def build_event(
    text: str,
    scores: dict[str, float],
    source_platform: str,
    *,
    timestamp: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    confidence: float | None = None,
    model_version: str = MODEL_VERSION,
    session_id: str | None = None,
    user_hash: str | None = None,
    dp_noise_applied: bool | None = None,
    event_category: list[str] | None = None,
) -> dict[str, Any]:
    """Build a schema-valid, SIEM-ready TextThreat event from model scores."""
    clean_scores = {
        label: max(0.0, min(1.0, float(score)))
        for label, score in scores.items()
        if label in ALL_HARM_TYPES
    }
    risk_score = max(clean_scores.values(), default=0.0)
    harm_types = [label for label, score in clean_scores.items() if score >= threshold]
    if confidence is None:
        confidence = risk_score

    event: dict[str, Any] = {
        "@timestamp": timestamp or now_utc_iso(),
        "event": {
            "kind": "signal",
            "module": "textthreat",
            "category": event_category or ["digital_wellbeing", "threat_signal"],
        },
        "text_hash": hash_text(text),
        "digital_wellbeing": {
            "harm_types": harm_types,
            "risk_score": round(risk_score, 6),
            "confidence": round(max(0.0, min(1.0, float(confidence))), 6),
            "model_version": model_version,
            "scores": {label: round(score, 6) for label, score in clean_scores.items()},
        },
        "source_platform": source_platform,
    }
    if session_id:
        event["session_id"] = session_id
    if user_hash:
        event["user_hash"] = user_hash
    if dp_noise_applied is not None:
        event["digital_wellbeing"]["dp_noise_applied"] = bool(dp_noise_applied)

    return validate_event(event)
