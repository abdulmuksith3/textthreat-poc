"""Generate schema-valid TextThreat NDJSON events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import LABELS, MODEL_VERSION, STRESS_LABEL
from .schema import build_event, validate_event
from .splunk_hec import send_event
from .utils import DATA_DIR, write_ndjson


SAMPLE_OUTPUT = DATA_DIR / "exports" / "sample_textthreat_events.ndjson"


def sample_prediction_rows(count: int = 5) -> list[dict[str, Any]]:
    """Return safe synthetic prediction rows for demos and smoke tests."""
    templates = [
        ("demo-session-1", "jigsaw", {"toxic": 0.87, "insult": 0.76, "threat": 0.21}),
        ("demo-session-1", "dreaddit", {STRESS_LABEL: 0.84, "toxic": 0.12}),
        ("demo-session-2", "jigsaw", {"toxic": 0.14, "obscene": 0.08, "insult": 0.11}),
        ("demo-session-3", "jigsaw", {"threat": 0.91, "toxic": 0.82, "identity_hate": 0.18}),
        ("demo-session-4", "dreaddit", {STRESS_LABEL: 0.66, "toxic": 0.09}),
    ]
    rows: list[dict[str, Any]] = []
    for idx in range(count):
        session_id, source, scores = templates[idx % len(templates)]
        rows.append(
            {
                "text": f"safe synthetic demo comment {idx + 1}",
                "source_platform": source,
                "session_id": session_id,
                "scores": scores,
                "model_version": MODEL_VERSION,
            }
        )
    return rows


def event_from_prediction(row: dict[str, Any]) -> dict[str, Any]:
    """Convert one prediction row into a validated TextThreat event."""
    scores = row.get("scores")
    if not isinstance(scores, dict):
        scores = {label: row.get(label, 0.0) for label in LABELS + [STRESS_LABEL]}
    event = build_event(
        str(row.get("text", "")),
        {label: float(value) for label, value in scores.items()},
        str(row.get("source_platform", "demo")),
        model_version=str(row.get("model_version", MODEL_VERSION)),
        session_id=row.get("session_id"),
        user_hash=row.get("user_hash"),
        dp_noise_applied=row.get("dp_noise_applied"),
    )
    return validate_event(event)


def events_from_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build events from prediction rows."""
    return [event_from_prediction(row) for row in rows]


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    """Load prediction rows from JSON, JSONL/NDJSON, or CSV."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("predictions", [payload])
        return list(payload)
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    frame = pd.read_csv(path)
    return frame.to_dict(orient="records")


def write_events(rows: list[dict[str, Any]], output_path: Path) -> Path:
    """Build, validate, and write events to NDJSON."""
    return write_ndjson(output_path, events_from_predictions(rows))


def write_and_optionally_stream_events(rows: list[dict[str, Any]], output_path: Path, send_splunk: bool) -> tuple[Path, list[dict[str, Any]]]:
    """Build events, write NDJSON, and optionally stream each event to Splunk HEC."""
    events = events_from_predictions(rows)
    path = write_ndjson(output_path, events)
    statuses = []
    if send_splunk:
        for event in events:
            statuses.append(send_event(event))
    return path, statuses


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export TextThreat events to NDJSON.")
    parser.add_argument("--sample", action="store_true", help="Write safe synthetic sample events.")
    parser.add_argument("--count", type=int, default=5, help="Number of sample events to generate.")
    parser.add_argument("--input", type=Path, help="Prediction file: JSON, JSONL/NDJSON, or CSV.")
    parser.add_argument("--output", type=Path, default=SAMPLE_OUTPUT, help="NDJSON output path.")
    parser.add_argument("--send-splunk", action="store_true", help="Also stream generated events to Splunk HEC using environment settings.")
    return parser


def main(argv: list[str] | None = None) -> Path:
    args = build_arg_parser().parse_args(argv)
    if args.sample:
        rows = sample_prediction_rows(args.count)
    elif args.input:
        rows = load_prediction_rows(args.input)
    else:
        raise SystemExit("Provide --sample or --input predictions file.")
    output_path, statuses = write_and_optionally_stream_events(rows, args.output, args.send_splunk)
    print(f"Wrote {len(rows)} TextThreat events to {output_path}")
    if args.send_splunk:
        sent = sum(1 for status in statuses if status.get("sent"))
        print(f"Streamed {sent}/{len(statuses)} events to Splunk HEC")
    return output_path


if __name__ == "__main__":
    main()
