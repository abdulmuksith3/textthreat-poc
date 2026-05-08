"""Train DistilBERT classifiers for Jigsaw toxicity and Dreaddit stress."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from .constants import LABELS, RANDOM_SEED, STRESS_LABEL
from .data import DREADDIT_TEST_PATH, DREADDIT_TRAIN_PATH, JIGSAW_TRAIN_PATH, load_dreaddit, load_jigsaw
from .evaluate import compute_classification_metrics
from .utils import MODELS_DIR, RESULTS_DIR, ensure_dir, set_seed, write_json


def optional_mlflow_log(run_name: str, params: dict[str, Any], metrics: dict[str, Any]) -> None:
    """Log an MLflow run when mlflow is installed and configured."""
    try:
        import mlflow
    except ImportError:
        return
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value is not None:
                mlflow.log_metric(key, float(value))


def apply_lora(model, task_type: str):
    """Apply LoRA adapters to DistilBERT attention projections when PEFT is available."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("LoRA requested but peft is not installed. Install requirements.txt or pass --no-lora.") from exc

    peft_task = TaskType.SEQ_CLS if task_type == "sequence_classification" else TaskType.SEQ_CLS
    config = LoraConfig(
        task_type=peft_task,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],
    )
    return get_peft_model(model, config)


def build_trainer(Trainer, *, model, args, train_dataset, eval_dataset, tokenizer, data_collator, compute_metrics):
    """Create a Transformers Trainer across tokenizer API changes."""
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    return Trainer(**trainer_kwargs)


def remove_stale_adapter_files(output_dir: Path) -> None:
    """Remove adapter-only files so merged model folders load as full HF models."""
    for filename in ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]:
        path = output_dir / filename
        if path.exists():
            path.unlink()


def save_full_model_artifact(model, tokenizer, output_dir: Path, training_args) -> None:
    """Save a self-contained Hugging Face model, merging LoRA adapters when present."""
    ensure_dir(output_dir)
    model_to_save = model
    if hasattr(model_to_save, "merge_and_unload"):
        model_to_save = model_to_save.merge_and_unload()
    remove_stale_adapter_files(output_dir)
    model_to_save.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))
    torch.save(training_args, output_dir / "training_args.bin")


def train_jigsaw(args: argparse.Namespace) -> dict[str, Any]:
    """Fine-tune DistilBERT for six-label Jigsaw classification."""
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

    frame = load_jigsaw(args.data)
    if args.sample_size:
        frame = frame.sample(n=min(args.sample_size, len(frame)), random_state=RANDOM_SEED)
    dataset = Dataset.from_pandas(frame[["comment_text", *LABELS]].reset_index(drop=True))

    def add_labels(example):
        example["labels"] = [float(example[label]) for label in LABELS]
        return example

    dataset = dataset.map(add_labels)
    split = dataset.train_test_split(test_size=0.2, seed=RANDOM_SEED)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch["comment_text"], truncation=True, max_length=args.max_length)

    train_ds = split["train"].map(tokenize, batched=True)
    val_ds = split["test"].map(tokenize, batched=True)
    keep = ["input_ids", "attention_mask", "labels"]
    train_ds = train_ds.remove_columns([column for column in train_ds.column_names if column not in keep])
    val_ds = val_ds.remove_columns([column for column in val_ds.column_names if column not in keep])
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        id2label={idx: label for idx, label in enumerate(LABELS)},
        label2id={label: idx for idx, label in enumerate(LABELS)},
    )
    if args.use_lora:
        model = apply_lora(model, "sequence_classification")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1.0 / (1.0 + np.exp(-logits))
        result = compute_classification_metrics(labels.astype(int), probs, LABELS)
        return {
            "macro_f1": result["macro_f1"],
            "micro_f1": result["micro_f1"],
            "macro_roc_auc": result["macro_roc_auc"] or 0.0,
            "macro_pr_auc": result["macro_pr_auc"] or 0.0,
            "ece": result["ece"]["ece"],
        }

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "hf_outputs"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=50,
        save_strategy="epoch",
        do_eval=True,
        report_to=[],
        seed=RANDOM_SEED,
    )
    trainer = build_trainer(
        Trainer,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    eval_metrics = trainer.evaluate()
    save_full_model_artifact(trainer.model, tokenizer, args.output_dir, training_args)

    metrics = {"demo": False, "task": "jigsaw", "model_name": args.model_name, "use_lora": args.use_lora, **eval_metrics}
    write_json(args.results, metrics)
    optional_mlflow_log("distilbert_jigsaw", vars(args), {k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    return metrics


def train_dreaddit(args: argparse.Namespace) -> dict[str, Any]:
    """Fine-tune DistilBERT for binary Dreaddit stress classification."""
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

    train_frame, test_frame, text_column, label_column = load_dreaddit(args.data, args.test_data)
    if args.sample_size:
        train_frame = train_frame.sample(n=min(args.sample_size, len(train_frame)), random_state=RANDOM_SEED)
    train_ds = Dataset.from_pandas(train_frame[[text_column, label_column]].reset_index(drop=True))
    if test_frame is not None:
        val_ds = Dataset.from_pandas(test_frame[[text_column, label_column]].reset_index(drop=True))
    else:
        split = train_ds.train_test_split(test_size=0.2, seed=RANDOM_SEED)
        train_ds = split["train"]
        val_ds = split["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        tokenized = tokenizer(batch[text_column], truncation=True, max_length=args.max_length)
        tokenized["labels"] = [int(value) for value in batch[label_column]]
        return tokenized

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    keep = ["input_ids", "attention_mask", "labels"]
    train_ds = train_ds.remove_columns([column for column in train_ds.column_names if column not in keep])
    val_ds = val_ds.remove_columns([column for column in val_ds.column_names if column not in keep])
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "not_stress", 1: STRESS_LABEL},
        label2id={"not_stress": 0, STRESS_LABEL: 1},
    )
    if args.use_lora:
        model = apply_lora(model, "sequence_classification")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        stress_prob = probs[:, 1]
        pred = (stress_prob >= 0.5).astype(int)
        metrics = {"f1": float(f1_score(labels, pred, zero_division=0))}
        if len(np.unique(labels)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(labels, stress_prob))
            metrics["pr_auc"] = float(average_precision_score(labels, stress_prob))
        return metrics

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "hf_outputs"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=50,
        save_strategy="epoch",
        do_eval=True,
        report_to=[],
        seed=RANDOM_SEED,
    )
    trainer = build_trainer(
        Trainer,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    eval_metrics = trainer.evaluate()
    save_full_model_artifact(trainer.model, tokenizer, args.output_dir, training_args)

    metrics = {"demo": False, "task": "dreaddit", "model_name": args.model_name, "use_lora": args.use_lora, **eval_metrics}
    write_json(args.results, metrics)
    optional_mlflow_log("distilbert_dreaddit", vars(args), {k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train TextThreat DistilBERT classifiers.")
    parser.add_argument("--task", choices=["jigsaw", "dreaddit"], default="jigsaw")
    parser.add_argument("--data", type=Path, default=JIGSAW_TRAIN_PATH)
    parser.add_argument("--test-data", type=Path, default=DREADDIT_TEST_PATH)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--no-lora", dest="use_lora", action="store_false", help="Disable PEFT LoRA adapters.")
    parser.set_defaults(use_lora=True)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    set_seed(RANDOM_SEED)
    args = build_arg_parser().parse_args(argv)
    if args.task == "jigsaw":
        args.output_dir = args.output_dir or MODELS_DIR / "distilbert_jigsaw"
        args.results = args.results or RESULTS_DIR / "distilbert_metrics.json"
        args.data = args.data or JIGSAW_TRAIN_PATH
        metrics = train_jigsaw(args)
    else:
        args.output_dir = args.output_dir or MODELS_DIR / "distilbert_dreaddit"
        args.results = args.results or RESULTS_DIR / "dreaddit_metrics.json"
        if args.data == JIGSAW_TRAIN_PATH:
            args.data = DREADDIT_TRAIN_PATH
        metrics = train_dreaddit(args)
    print(f"Wrote DistilBERT metrics to {args.results}")
    return metrics


if __name__ == "__main__":
    main()
