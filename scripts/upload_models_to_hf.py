"""Upload TextThreat trained models to Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_JIGSAW_REPO = "textthreat-distilbert-jigsaw"
DEFAULT_DREADDIT_REPO = "textthreat-distilbert-dreaddit"


def require_token() -> str:
    """Return the configured Hugging Face token or raise a helpful error."""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise SystemExit(
            "Missing HF_TOKEN. Create a Hugging Face write token at "
            "https://huggingface.co/settings/tokens and set HF_TOKEN before uploading."
        )
    return token


def validate_model_dir(path: Path) -> None:
    """Validate that a model folder has the files needed for hosted inference."""
    missing = []
    if not (path / "config.json").exists():
        missing.append("config.json")
    if not (path / "model.safetensors").exists() and not (path / "pytorch_model.bin").exists():
        missing.append("model.safetensors or pytorch_model.bin")
    if not (path / "tokenizer.json").exists() and not (path / "vocab.txt").exists():
        missing.append("tokenizer.json or vocab.txt")
    if missing:
        raise SystemExit(f"{path} is incomplete. Missing: {', '.join(missing)}")


def repo_id(namespace: str, name: str) -> str:
    """Build a Hugging Face repo id."""
    return f"{namespace.strip('/')}/{name.strip('/')}"


def upload_model(api: HfApi, local_dir: Path, target_repo: str, *, private: bool) -> None:
    """Create/update a Hugging Face model repo from a local model directory."""
    validate_model_dir(local_dir)
    api.create_repo(repo_id=target_repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=target_repo,
        repo_type="model",
        folder_path=str(local_dir),
        ignore_patterns=["hf_outputs/*", "_adapter_backup/*", "checkpoint-*/*", "training_args.bin"],
        commit_message="Upload TextThreat trained model artifact",
    )
    print(f"Uploaded {local_dir} -> https://huggingface.co/{target_repo}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Upload TextThreat models to Hugging Face Hub.")
    parser.add_argument("--namespace", default=os.getenv("HF_NAMESPACE"), help="Hugging Face username or org.")
    parser.add_argument("--jigsaw-repo", default=DEFAULT_JIGSAW_REPO)
    parser.add_argument("--dreaddit-repo", default=DEFAULT_DREADDIT_REPO)
    parser.add_argument("--jigsaw-dir", type=Path, default=Path("models/distilbert_jigsaw"))
    parser.add_argument("--dreaddit-dir", type=Path, default=Path("models/distilbert_dreaddit"))
    parser.add_argument("--private", action="store_true", help="Create private model repos.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Upload both TextThreat DistilBERT model artifacts."""
    args = build_arg_parser().parse_args(argv)
    token = require_token()
    api = HfApi(token=token)
    if not args.namespace:
        args.namespace = api.whoami()["name"]

    jigsaw_id = repo_id(args.namespace, args.jigsaw_repo)
    dreaddit_id = repo_id(args.namespace, args.dreaddit_repo)
    upload_model(api, args.jigsaw_dir, jigsaw_id, private=args.private)
    upload_model(api, args.dreaddit_dir, dreaddit_id, private=args.private)

    print("\nSet these for the hosted/local demo:")
    print(f"TEXTTHREAT_TOXICITY_MODEL_ID={jigsaw_id}")
    print(f"TEXTTHREAT_STRESS_MODEL_ID={dreaddit_id}")


if __name__ == "__main__":
    main()
