"""Synthetic co-occurrence evaluation for toxicity and stress session windows."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import HIGH_RISK_THRESHOLD, RANDOM_SEED, STRESS_LABEL
from .utils import DATA_DIR, RESULTS_DIR, ensure_dir, set_seed, write_json


SYNTHETIC_SESSIONS_PATH = DATA_DIR / "synthetic" / "cooccurrence_sessions.csv"


def generate_synthetic_sessions(count: int = 200) -> pd.DataFrame:
    """Create synthetic session windows with known co-occurrence ground truth."""
    rng = np.random.default_rng(RANDOM_SEED)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for idx in range(count):
        session_id = f"session-{idx:04d}"
        is_positive = idx < count // 2
        start = base + timedelta(minutes=idx * 40)
        if is_positive:
            rows.append(
                {
                    "session_id": session_id,
                    "@timestamp": (start + timedelta(minutes=2)).isoformat(),
                    "harm_type": "toxic",
                    "risk_score": float(rng.uniform(0.82, 0.98)),
                    "ground_truth_cooccurrence": 1,
                }
            )
            rows.append(
                {
                    "session_id": session_id,
                    "@timestamp": (start + timedelta(minutes=18)).isoformat(),
                    "harm_type": STRESS_LABEL,
                    "risk_score": float(rng.uniform(0.82, 0.98)),
                    "ground_truth_cooccurrence": 1,
                }
            )
        else:
            stress_after_window = idx % 4 == 0
            rows.append(
                {
                    "session_id": session_id,
                    "@timestamp": (start + timedelta(minutes=4)).isoformat(),
                    "harm_type": "toxic" if idx % 2 == 0 else STRESS_LABEL,
                    "risk_score": float(rng.uniform(0.82, 0.98)),
                    "ground_truth_cooccurrence": 0,
                }
            )
            if stress_after_window:
                rows.append(
                    {
                        "session_id": session_id,
                        "@timestamp": (start + timedelta(minutes=50)).isoformat(),
                        "harm_type": STRESS_LABEL,
                        "risk_score": float(rng.uniform(0.82, 0.98)),
                        "ground_truth_cooccurrence": 0,
                    }
                )
    return pd.DataFrame(rows)


def detect_session_cooccurrence(
    events: pd.DataFrame,
    *,
    window_minutes: int = 30,
    threshold: float = HIGH_RISK_THRESHOLD,
) -> dict[str, int]:
    """Detect sessions where toxicity and stress appear within the same time window."""
    frame = events.copy()
    frame["@timestamp"] = pd.to_datetime(frame["@timestamp"], utc=True)
    frame = frame[frame["risk_score"].astype(float) >= threshold]
    detections: dict[str, int] = {}
    for session_id, group in frame.groupby("session_id"):
        toxic_times = group[group["harm_type"].isin(["toxic", "threat", "insult", "obscene"])]["@timestamp"]
        stress_times = group[group["harm_type"] == STRESS_LABEL]["@timestamp"]
        detected = 0
        for tox_time in toxic_times:
            if any(abs((tox_time - stress_time).total_seconds()) <= window_minutes * 60 for stress_time in stress_times):
                detected = 1
                break
        detections[str(session_id)] = detected
    return detections


def evaluate_cooccurrence(events: pd.DataFrame, window_minutes: int = 30, threshold: float = HIGH_RISK_THRESHOLD) -> dict[str, Any]:
    """Evaluate synthetic session-window co-occurrence detection."""
    detections = detect_session_cooccurrence(events, window_minutes=window_minutes, threshold=threshold)
    truth = events.groupby("session_id")["ground_truth_cooccurrence"].max().astype(int).to_dict()
    tp = fp = tn = fn = 0
    for session_id, truth_value in truth.items():
        predicted = int(detections.get(str(session_id), 0))
        if truth_value == 1 and predicted == 1:
            tp += 1
        elif truth_value == 0 and predicted == 1:
            fp += 1
        elif truth_value == 0 and predicted == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    return {
        "evaluation_type": "synthetic_session_windows",
        "demo": False,
        "window_minutes": int(window_minutes),
        "threshold": float(threshold),
        "session_count": int(len(truth)),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": float(precision),
        "recall": float(recall),
        "false_positive_rate": float(false_positive_rate),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TextThreat co-occurrence evaluation.")
    parser.add_argument("--sample", action="store_true", help="Generate synthetic session windows.")
    parser.add_argument("--input", type=Path, help="CSV with session_id, @timestamp, harm_type, risk_score, ground_truth_cooccurrence.")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "cooccurrence_results.json")
    parser.add_argument("--synthetic-output", type=Path, default=SYNTHETIC_SESSIONS_PATH)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--window-minutes", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=HIGH_RISK_THRESHOLD)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    set_seed(RANDOM_SEED)
    if args.sample:
        frame = generate_synthetic_sessions(args.count)
        ensure_dir(args.synthetic_output.parent)
        frame.to_csv(args.synthetic_output, index=False)
        demo = True
    elif args.input:
        frame = pd.read_csv(args.input)
        demo = False
    else:
        raise SystemExit("Provide --sample or --input session-window CSV.")
    result = evaluate_cooccurrence(frame, window_minutes=args.window_minutes, threshold=args.threshold)
    result["demo"] = demo
    write_json(args.output, result)
    print(f"Wrote co-occurrence results to {args.output}")
    return result


if __name__ == "__main__":
    main()
