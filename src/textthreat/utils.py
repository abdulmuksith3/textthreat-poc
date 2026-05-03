"""Small filesystem and serialization helpers used by TextThreat scripts."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
MODELS_DIR = PROJECT_ROOT / "models"


def repo_path(*parts: str) -> Path:
    """Return an absolute path under the repository root."""
    return PROJECT_ROOT.joinpath(*parts)


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_utc_iso() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds for Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write a JSON file with stable formatting."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_ndjson(path: Path, records: Iterable[dict[str, Any]]) -> Path:
    """Write newline-delimited JSON records."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return path


def read_ndjson(path: Path) -> list[dict[str, Any]]:
    """Read newline-delimited JSON records."""
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def percentile(values: list[float], q: float) -> float:
    """Compute a percentile for a non-empty list."""
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def as_float(value: Any, default: float = 0.0) -> float:
    """Convert values from pandas/JSON safely into finite floats."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(result) or np.isinf(result):
        return default
    return result
