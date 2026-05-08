"""Output-level privacy-preserving perturbation experiment for risk scores."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .constants import LABELS, RANDOM_SEED
from .evaluate import compute_classification_metrics, sample_arrays
from .utils import RESULTS_DIR, set_seed, write_json


EPSILON_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]
DEFAULT_SENSITIVITY = 0.056


def gaussian_sigma(epsilon: float, delta: float = 1e-5, sensitivity: float = DEFAULT_SENSITIVITY) -> float:
    """Return Gaussian mechanism sigma for output-level score perturbation."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    return float(np.sqrt(2.0 * np.log(1.25 / delta)) * sensitivity / epsilon)


def perturb_scores(
    scores: np.ndarray,
    epsilon: float,
    *,
    delta: float = 1e-5,
    sensitivity: float = DEFAULT_SENSITIVITY,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """Add Gaussian noise to score outputs and clip to [0, 1]."""
    rng = rng or np.random.default_rng(RANDOM_SEED)
    sigma = gaussian_sigma(epsilon, delta=delta, sensitivity=sensitivity)
    noisy = scores + rng.normal(loc=0.0, scale=sigma, size=scores.shape)
    return np.clip(noisy, 0.0, 1.0), sigma


def run_dp_experiment(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    epsilons: list[float] | None = None,
    delta: float = 1e-5,
    sensitivity: float = DEFAULT_SENSITIVITY,
) -> dict[str, Any]:
    """Evaluate utility after output-level Gaussian perturbation."""
    epsilons = epsilons or EPSILON_VALUES
    baseline = compute_classification_metrics(y_true, y_prob, LABELS)
    baseline_f1 = baseline["macro_f1"]
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for epsilon in epsilons:
        noisy, sigma = perturb_scores(y_prob, epsilon, delta=delta, sensitivity=sensitivity, rng=rng)
        metrics = compute_classification_metrics(y_true, noisy, LABELS)
        rows.append(
            {
                "epsilon": float(epsilon),
                "sigma": float(sigma),
                "macro_f1": float(metrics["macro_f1"]),
                "f1_degradation": float(baseline_f1 - metrics["macro_f1"]),
                "relative_f1_degradation": float((baseline_f1 - metrics["macro_f1"]) / baseline_f1) if baseline_f1 else None,
            }
        )
    return {
        "experiment_type": "output-level privacy-preserving perturbation experiment",
        "demo": False,
        "delta": float(delta),
        "sensitivity": float(sensitivity),
        "mechanism": "Gaussian output perturbation",
        "mechanism_formula": "sigma = sqrt(2 * ln(1.25 / delta)) * sensitivity / epsilon",
        "perturbation_target": "label probability outputs with downstream risk score clipping to [0, 1]",
        "opacus_available": opacus_available(),
        "baseline_macro_f1": float(baseline_f1),
        "results": rows,
    }


def opacus_available() -> bool:
    """Return whether Opacus is installed in the current environment."""
    try:
        import opacus  # noqa: F401
    except ImportError:
        return False
    return True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TextThreat output-level DP perturbation experiment.")
    parser.add_argument("--sample", action="store_true", help="Use deterministic demo arrays.")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "dp_results.json")
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--sensitivity", type=float, default=DEFAULT_SENSITIVITY)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    set_seed(RANDOM_SEED)
    if not args.sample:
        print("No prediction file supplied; using sample mode for the output-level DP experiment.")
    y_true, y_prob = sample_arrays()
    result = run_dp_experiment(y_true, y_prob, delta=args.delta, sensitivity=args.sensitivity)
    result["demo"] = bool(args.sample or True)
    write_json(args.output, result)
    print(f"Wrote DP perturbation results to {args.output}")
    return result


if __name__ == "__main__":
    main()
