"""SHAP explainability support for TextThreat model outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .constants import LABELS
from .inference import TextThreatAnalyzer
from .utils import RESULTS_DIR, write_json


SAMPLE_TEXTS = [
    "You are disgusting and nobody wants you here.",
    "I am overwhelmed and scared and I cannot sleep.",
    "I will kill you if you come back.",
]


def summarize_tokens(values: Any, limit: int = 12) -> list[dict[str, Any]]:
    """Extract a compact token attribution summary from SHAP values."""
    rows: list[dict[str, Any]] = []
    data = getattr(values, "data", [])
    shap_values = getattr(values, "values", [])
    for text_index, tokens in enumerate(data):
        token_scores = np.asarray(shap_values[text_index]).reshape(-1)
        ranked = sorted(
            zip([str(token) for token in tokens], token_scores),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )
        rows.append(
            {
                "text_index": text_index,
                "top_tokens": [
                    {"token": token, "attribution": float(score)}
                    for token, score in ranked[:limit]
                    if token.strip()
                ],
            }
        )
    return rows


def run_shap_explainability(texts: list[str], output: Path) -> dict[str, Any]:
    """Run a compact SHAP text attribution experiment for the toxic score."""
    try:
        import shap
    except ImportError as exc:
        raise SystemExit("Install shap to run explainability: pip install shap") from exc

    analyzer = TextThreatAnalyzer()

    def predict_toxic(batch: list[str]) -> np.ndarray:
        scores = [analyzer.analyze(text)["scores"].get("toxic", 0.0) for text in batch]
        return np.asarray(scores, dtype=float)

    masker = shap.maskers.Text(r"\W+")
    explainer = shap.Explainer(predict_toxic, masker)
    values = explainer(texts)
    result = {
        "demo": False,
        "method": "SHAP text attribution",
        "target_output": "toxic probability",
        "sample_count": len(texts),
        "summaries": summarize_tokens(values),
    }
    write_json(output, result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TextThreat SHAP explainability.")
    parser.add_argument("--sample", action="store_true", help="Use bundled sample comments.")
    parser.add_argument("--text", action="append", help="Text to explain. Can be supplied multiple times.")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "shap_explainability_summary.json")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    texts = args.text or SAMPLE_TEXTS
    result = run_shap_explainability(texts, args.output)
    result["demo"] = bool(args.sample or not args.text)
    write_json(args.output, result)
    print(f"Wrote SHAP explainability summary to {args.output}")
    return result


if __name__ == "__main__":
    main()
