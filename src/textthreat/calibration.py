"""Expected Calibration Error utilities."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np


def binary_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> dict[str, Any]:
    """Compute binary Expected Calibration Error with uniform confidence bins."""
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError("y_true and y_prob must have the same length.")

    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, float | int]] = []
    ece = 0.0
    total = len(y_prob)
    for idx in range(n_bins):
        lower = boundaries[idx]
        upper = boundaries[idx + 1]
        if idx == n_bins - 1:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)
        count = int(mask.sum())
        if count == 0:
            bins.append({"lower": float(lower), "upper": float(upper), "count": 0, "accuracy": 0.0, "confidence": 0.0})
            continue
        accuracy = float(y_true[mask].mean())
        confidence = float(y_prob[mask].mean())
        weight = count / max(total, 1)
        ece += weight * abs(accuracy - confidence)
        bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "accuracy": accuracy,
                "confidence": confidence,
            }
        )
    return {"ece": float(ece), "n_bins": int(n_bins), "bins": bins}


def multilabel_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str] | None = None,
    n_bins: int = 15,
) -> dict[str, Any]:
    """Compute per-label and average ECE for multi-label probabilities."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have matching shapes.")
    if y_true.ndim != 2:
        raise ValueError("multi-label ECE expects 2D arrays.")

    labels = labels or [f"label_{idx}" for idx in range(y_true.shape[1])]
    per_label: dict[str, float] = {}
    detailed_bins: dict[str, Any] = {}
    for idx, label in enumerate(labels):
        result = binary_ece(y_true[:, idx], y_prob[:, idx], n_bins=n_bins)
        per_label[label] = float(result["ece"])
        detailed_bins[label] = result["bins"]
    return {
        "ece": float(np.mean(list(per_label.values()))) if per_label else 0.0,
        "n_bins": int(n_bins),
        "per_label_ece": per_label,
        "bins": detailed_bins,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute a small demo calibration result.")
    parser.add_argument("--bins", type=int, default=15)
    args = parser.parse_args()
    y_true = np.array([0, 0, 1, 1, 1])
    y_prob = np.array([0.05, 0.25, 0.62, 0.78, 0.91])
    print(binary_ece(y_true, y_prob, n_bins=args.bins))


if __name__ == "__main__":
    main()
