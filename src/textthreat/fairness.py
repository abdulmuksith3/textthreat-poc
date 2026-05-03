"""Fairlearn-based demographic fairness audit for TextThreat outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import IDENTITY_COLUMNS, RANDOM_SEED
from .utils import RESULTS_DIR, set_seed, write_json


def _difference(values: dict[str, float]) -> float:
    if not values:
        return 0.0
    return float(max(values.values()) - min(values.values()))


def simple_group_metrics(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray) -> dict[str, Any]:
    """Compute selection-rate and TPR/FPR differences for binary sensitive groups."""
    metrics: dict[str, Any] = {"groups": {}}
    for group in [0, 1]:
        mask = sensitive == group
        if not mask.any():
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        selection_rate = float(np.mean(yp))
        positives = yt == 1
        negatives = yt == 0
        tpr = float(np.mean(yp[positives] == 1)) if positives.any() else 0.0
        fpr = float(np.mean(yp[negatives] == 1)) if negatives.any() else 0.0
        metrics["groups"][str(group)] = {
            "count": int(mask.sum()),
            "selection_rate": selection_rate,
            "true_positive_rate": tpr,
            "false_positive_rate": fpr,
        }
    selection = {group: payload["selection_rate"] for group, payload in metrics["groups"].items()}
    tpr = {group: payload["true_positive_rate"] for group, payload in metrics["groups"].items()}
    fpr = {group: payload["false_positive_rate"] for group, payload in metrics["groups"].items()}
    metrics["demographic_parity_difference"] = _difference(selection)
    metrics["equalized_odds_difference"] = max(_difference(tpr), _difference(fpr))
    return metrics


def metricframe_group_metrics(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray) -> dict[str, Any]:
    """Use Fairlearn MetricFrame when available; otherwise use the local implementation."""
    try:
        from fairlearn.metrics import MetricFrame, false_positive_rate, selection_rate, true_positive_rate
    except ImportError:
        return simple_group_metrics(y_true, y_pred, sensitive)

    frame = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "true_positive_rate": true_positive_rate,
            "false_positive_rate": false_positive_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )
    by_group = frame.by_group.fillna(0.0)
    groups = {
        str(group): {metric: float(value) for metric, value in row.items()}
        for group, row in by_group.iterrows()
    }
    selection = {group: payload["selection_rate"] for group, payload in groups.items()}
    tpr = {group: payload["true_positive_rate"] for group, payload in groups.items()}
    fpr = {group: payload["false_positive_rate"] for group, payload in groups.items()}
    return {
        "groups": groups,
        "demographic_parity_difference": _difference(selection),
        "equalized_odds_difference": max(_difference(tpr), _difference(fpr)),
    }


def sample_fairness_frame() -> pd.DataFrame:
    """Generate deterministic demo predictions with Jigsaw-style identity annotations."""
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for idx in range(80):
        y_true = int(idx % 3 == 0 or idx % 11 == 0)
        score = 0.72 if y_true else 0.22
        score += float(rng.normal(0.0, 0.08))
        row = {
            "y_true": y_true,
            "y_score": max(0.0, min(1.0, score)),
        }
        for column in IDENTITY_COLUMNS:
            row[column] = 1.0 if rng.random() < 0.18 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def audit_fairness(frame: pd.DataFrame, threshold: float = 0.5) -> dict[str, Any]:
    """Audit fairness over available identity columns."""
    missing = [column for column in ["y_true", "y_score"] if column not in frame.columns]
    if missing:
        raise ValueError(f"Fairness audit requires columns {missing}.")
    identity_columns = [column for column in IDENTITY_COLUMNS if column in frame.columns]
    if not identity_columns:
        return {
            "demo": True,
            "reason": "No supported Jigsaw identity annotation columns were found.",
            "supported_identity_columns": IDENTITY_COLUMNS,
            "subgroups": {},
        }

    y_true = frame["y_true"].astype(int).to_numpy()
    y_pred = (frame["y_score"].astype(float).to_numpy() >= threshold).astype(int)
    subgroups: dict[str, Any] = {}
    max_dpd = 0.0
    max_eod = 0.0
    for column in identity_columns:
        sensitive = (frame[column].astype(float).to_numpy() >= 0.5).astype(int)
        result = metricframe_group_metrics(y_true, y_pred, sensitive)
        subgroups[column] = result
        max_dpd = max(max_dpd, float(result["demographic_parity_difference"]))
        max_eod = max(max_eod, float(result["equalized_odds_difference"]))
    return {
        "demo": False,
        "threshold": float(threshold),
        "identity_columns": identity_columns,
        "maximum_demographic_parity_difference": max_dpd,
        "maximum_equalized_odds_difference": max_eod,
        "subgroups": subgroups,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TextThreat Fairlearn fairness audit.")
    parser.add_argument("--sample", action="store_true", help="Write deterministic demo fairness output.")
    parser.add_argument("--input", type=Path, help="CSV with y_true, y_score, and identity columns.")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "fairness_results.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    set_seed(RANDOM_SEED)
    if args.sample:
        frame = sample_fairness_frame()
        result = audit_fairness(frame, threshold=args.threshold)
        result["demo"] = True
        result["note"] = "Demo fairness audit; use Jigsaw unintended-bias identity annotations for real audit."
    elif args.input:
        frame = pd.read_csv(args.input)
        result = audit_fairness(frame, threshold=args.threshold)
    else:
        result = {
            "demo": True,
            "reason": "No fairness input supplied. Provide --input with y_true, y_score, and identity columns.",
            "supported_identity_columns": IDENTITY_COLUMNS,
        }
    write_json(args.output, result)
    print(f"Wrote fairness results to {args.output}")
    return result


if __name__ == "__main__":
    main()
