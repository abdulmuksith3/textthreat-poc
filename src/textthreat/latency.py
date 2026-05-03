"""Latency profiling for the TextThreat inference-to-export pipeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .constants import MODEL_VERSION
from .export_events import sample_prediction_rows
from .schema import build_event, validate_event
from .utils import RESULTS_DIR, percentile, write_json


def sample_inference(text: str, index: int) -> dict[str, float]:
    """Deterministic lightweight inference used by sample latency profiling."""
    length_signal = min(1.0, len(text) / 120.0)
    return {
        "toxic": 0.15 + 0.1 * (index % 3) + length_signal * 0.2,
        "insult": 0.18 + 0.12 * (index % 2),
        "threat": 0.08 + 0.81 * (index % 5 == 0),
        "stress": 0.22 + 0.64 * (index % 4 == 0),
    }


def profile_sample_pipeline(
    iterations: int = 25,
    *,
    simulate_siem: bool = True,
    simulate_soar: bool = True,
) -> dict[str, Any]:
    """Profile preprocessing, inference, schema validation, and simulated integration stages."""
    stage_timings: dict[str, list[float]] = {
        "preprocessing": [],
        "model_inference": [],
        "json_serialization_schema_validation": [],
        "siem_ingestion_simulation": [],
        "soar_dispatch_simulation": [],
        "detection_to_export": [],
        "total_pipeline": [],
    }
    rows = sample_prediction_rows(max(iterations, 5))
    for index in range(iterations):
        row = rows[index % len(rows)]
        total_start = time.perf_counter()

        start = time.perf_counter()
        text = str(row["text"]).strip()
        stage_timings["preprocessing"].append(time.perf_counter() - start)

        start = time.perf_counter()
        scores = sample_inference(text, index)
        stage_timings["model_inference"].append(time.perf_counter() - start)

        start = time.perf_counter()
        event = build_event(
            text,
            scores,
            "latency_sample",
            model_version=MODEL_VERSION,
            session_id=str(row.get("session_id", "latency-session")),
        )
        json.dumps(validate_event(event))
        stage_timings["json_serialization_schema_validation"].append(time.perf_counter() - start)
        detection_export_elapsed = time.perf_counter() - total_start
        stage_timings["detection_to_export"].append(detection_export_elapsed)

        start = time.perf_counter()
        if simulate_siem:
            time.sleep(0.001)
        stage_timings["siem_ingestion_simulation"].append(time.perf_counter() - start)

        start = time.perf_counter()
        if simulate_soar and event["digital_wellbeing"]["risk_score"] >= 0.8:
            time.sleep(0.001)
        stage_timings["soar_dispatch_simulation"].append(time.perf_counter() - start)
        stage_timings["total_pipeline"].append(time.perf_counter() - total_start)

    stages = {}
    for name, values in stage_timings.items():
        stages[name] = {
            "median_seconds": percentile(values, 50),
            "p95_seconds": percentile(values, 95),
            "samples": len(values),
        }
    return {
        "demo": True,
        "mode": "sample",
        "iterations": int(iterations),
        "model_version": MODEL_VERSION,
        "stages": stages,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile TextThreat latency.")
    parser.add_argument("--sample", action="store_true", help="Run sample latency profile.")
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "latency_metrics.json")
    parser.add_argument("--no-siem-simulation", dest="simulate_siem", action="store_false")
    parser.add_argument("--no-soar-simulation", dest="simulate_soar", action="store_false")
    parser.set_defaults(simulate_siem=True, simulate_soar=True)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    if not args.sample:
        print("No model profiling configuration supplied; using --sample latency profile.")
    result = profile_sample_pipeline(args.iterations, simulate_siem=args.simulate_siem, simulate_soar=args.simulate_soar)
    write_json(args.output, result)
    print(f"Wrote latency metrics to {args.output}")
    return result


if __name__ == "__main__":
    main()
