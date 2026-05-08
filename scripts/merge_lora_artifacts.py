"""Merge TextThreat LoRA adapters into full Hugging Face model folders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.textthreat.constants import LABELS, STRESS_LABEL


def has_adapter_weights(path: Path) -> bool:
    """Return true when a directory contains PEFT adapter weights."""
    return (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()


def has_full_weights(path: Path) -> bool:
    """Return true when a directory contains full Hugging Face model weights."""
    return (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()


def remove_adapter_files(path: Path) -> None:
    """Move adapter-only files aside after the full merged model has been saved."""
    for filename in ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]:
        artifact = path / filename
        if artifact.exists():
            backup_dir = path / "_adapter_backup"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup = backup_dir / filename
            if backup.exists():
                artifact.unlink()
            else:
                artifact.replace(backup)


def merge_lora_model(
    adapter_dir: Path,
    *,
    base_model_name: str,
    num_labels: int,
    problem_type: str,
    label_names: list[str],
) -> None:
    """Merge one PEFT LoRA adapter folder into a self-contained model folder."""
    if not adapter_dir.exists():
        print(f"Skipping missing directory: {adapter_dir}")
        return
    if has_full_weights(adapter_dir):
        remove_adapter_files(adapter_dir)
        print(f"Already full model: {adapter_dir}")
        return
    if not has_adapter_weights(adapter_dir):
        raise FileNotFoundError(f"No LoRA adapter weights found in {adapter_dir}")

    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    id2label = {idx: label for idx, label in enumerate(label_names)}
    label2id = {label: idx for idx, label in enumerate(label_names)}
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels,
        problem_type=problem_type,
        id2label=id2label,
        label2id=label2id,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged_model = model.merge_and_unload()
    merged_model.config.problem_type = problem_type
    merged_model.config.id2label = id2label
    merged_model.config.label2id = label2id

    merged_model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    remove_adapter_files(adapter_dir)
    print(f"Saved merged full model to: {adapter_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Merge TextThreat LoRA adapters into full model folders.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository or artifact root.")
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Merge Jigsaw and Dreaddit adapters when present."""
    args = build_arg_parser().parse_args(argv)
    root = args.root
    merge_lora_model(
        root / "models" / "distilbert_jigsaw",
        base_model_name=args.base_model,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        label_names=LABELS,
    )
    merge_lora_model(
        root / "models" / "distilbert_dreaddit",
        base_model_name=args.base_model,
        num_labels=2,
        problem_type="single_label_classification",
        label_names=["not_stress", STRESS_LABEL],
    )


if __name__ == "__main__":
    main()
