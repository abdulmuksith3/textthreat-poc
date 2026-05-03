"""Reusable classification metrics for thesis evidence artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .calibration import multilabel_ece
from .constants import DEFAULT_THRESHOLD, LABELS, RANDOM_SEED
from .utils import RESULTS_DIR, set_seed, write_json


def _safe_macro_auc(func, y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    values: list[float] = []
    for idx in range(y_true.shape[1]):
        column = y_true[:, idx]
        if len(np.unique(column)) < 2:
            continue
        try:
            values.append(float(func(column, y_prob[:, idx])))
        except ValueError:
            continue
    if not values:
        return None
    return float(np.mean(values))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    n_bins: int = 15,
) -> dict[str, Any]:
    """Compute multi-label metrics used by the thesis result tables."""
    labels = labels or LABELS
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have matching shapes.")

    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    per_label_precision = {label: float(value) for label, value in zip(labels, precision)}
    per_label_recall = {label: float(value) for label, value in zip(labels, recall)}
    per_label_f1 = {label: float(value) for label, value in zip(labels, f1)}

    metrics = {
        "threshold": float(threshold),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "per_label_precision": per_label_precision,
        "per_label_recall": per_label_recall,
        "per_label_f1": per_label_f1,
        "macro_roc_auc": _safe_macro_auc(roc_auc_score, y_true, y_prob),
        "macro_pr_auc": _safe_macro_auc(average_precision_score, y_true, y_prob),
        "exact_match_ratio": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "ece": multilabel_ece(y_true, y_prob, labels=labels, n_bins=n_bins),
    }
    return metrics


def sample_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic demo arrays with the same six-label shape as Jigsaw."""
    set_seed(RANDOM_SEED)
    y_true = np.array(
        [
            [1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
            [1, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )
    y_prob = np.array(
        [
            [0.91, 0.08, 0.77, 0.14, 0.83, 0.11],
            [0.18, 0.04, 0.13, 0.07, 0.19, 0.05],
            [0.82, 0.66, 0.71, 0.31, 0.68, 0.28],
            [0.22, 0.06, 0.14, 0.88, 0.21, 0.09],
            [0.76, 0.23, 0.42, 0.81, 0.73, 0.62],
            [0.11, 0.02, 0.07, 0.04, 0.09, 0.03],
        ],
        dtype=float,
    )
    return y_true, y_prob


def load_metric_arrays(truth_path: Path, scores_path: Path, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load truth and score matrices from CSV files using label column names."""
    truth = pd.read_csv(truth_path)[labels].astype(int).to_numpy()
    scores = pd.read_csv(scores_path)[labels].astype(float).to_numpy()
    return truth, scores


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute TextThreat classification metrics.")
    parser.add_argument("--sample", action="store_true", help="Write deterministic demo metrics.")
    parser.add_argument("--truth", type=Path, help="CSV with ground-truth label columns.")
    parser.add_argument("--scores", type=Path, help="CSV with probability score columns.")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "classification_metrics.json")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--bins", type=int, default=15)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    if args.sample:
        y_true, y_prob = sample_arrays()
        demo = True
    elif args.truth and args.scores:
        y_true, y_prob = load_metric_arrays(args.truth, args.scores, LABELS)
        demo = False
    else:
        raise SystemExit("Provide --sample or both --truth and --scores CSV files.")

    metrics = compute_classification_metrics(y_true, y_prob, LABELS, args.threshold, args.bins)
    metrics.update({"demo": demo, "labels": LABELS, "sample_count": int(y_true.shape[0])})
    write_json(args.output, metrics)
    print(f"Wrote classification metrics to {args.output}")
    return metrics


if __name__ == "__main__":
    main()
