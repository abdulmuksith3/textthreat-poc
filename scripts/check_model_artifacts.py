"""Check whether downloaded TextThreat model artifacts are usable for inference."""

from __future__ import annotations

import argparse
from pathlib import Path


def has_any_weight_file(path: Path) -> bool:
    """Return true when a model directory contains a loadable weight file."""
    return any(
        (path / filename).exists()
        for filename in [
            "adapter_model.safetensors",
            "adapter_model.bin",
            "model.safetensors",
            "pytorch_model.bin",
        ]
    )


def check_distilbert(path: Path) -> list[str]:
    """Return missing/invalid artifact messages for a DistilBERT/LoRA directory."""
    messages: list[str] = []
    if not path.exists():
        return [f"Missing directory: {path}"]
    if not (path / "config.json").exists():
        messages.append(f"Missing config.json in {path}")
    if not (path / "tokenizer.json").exists() and not (path / "vocab.txt").exists():
        messages.append(f"Missing tokenizer files in {path}")
    if (path / "adapter_config.json").exists():
        if not (path / "adapter_model.safetensors").exists() and not (path / "adapter_model.bin").exists():
            messages.append(
                f"Missing LoRA adapter weights in {path}: expected adapter_model.safetensors or adapter_model.bin"
            )
    elif not has_any_weight_file(path):
        messages.append(f"Missing model weights in {path}: expected model.safetensors or pytorch_model.bin")
    return messages


def check_joblib_pair(path: Path, expected: list[str]) -> list[str]:
    """Return missing file messages for classical ML model directories."""
    messages: list[str] = []
    if not path.exists():
        return [f"Missing directory: {path}"]
    for filename in expected:
        if not (path / filename).exists():
            messages.append(f"Missing {filename} in {path}")
    return messages


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check TextThreat model artifact completeness.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repo root or extracted artifact root.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    root = args.root
    checks = {
        "distilbert_jigsaw": check_distilbert(root / "models" / "distilbert_jigsaw"),
        "svm_tfidf": check_joblib_pair(
            root / "models" / "svm_tfidf",
            ["tfidf_vectorizer.joblib", "svm_calibrated_ovr.joblib"],
        ),
        "stress_dreaddit": check_joblib_pair(
            root / "models" / "stress_dreaddit",
            ["tfidf_stress.joblib", "logreg_stress.joblib"],
        ),
    }
    failed = False
    for name, messages in checks.items():
        if messages:
            failed = True
            print(f"[FAIL] {name}")
            for message in messages:
                print(f"  - {message}")
        else:
            print(f"[OK] {name}")
    if failed:
        raise SystemExit("Model artifact check failed.")
    print("All TextThreat model artifacts look usable.")


if __name__ == "__main__":
    main()
