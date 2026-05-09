"""Train DistilBERT classifiers for Jigsaw toxicity and Dreaddit stress."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from .constants import LABELS, RANDOM_SEED, STRESS_LABEL
from .data import (
    DREADDIT_TEST_PATH,
    DREADDIT_TRAIN_PATH,
    JIGSAW_TRAIN_PATH,
    TEXT_NORMALIZED_COLUMN,
    add_text_preprocessing,
    binary_train_test_split,
    load_dreaddit,
    load_jigsaw,
    multilabel_train_test_split,
)
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


def build_weighted_trainer(
    Trainer,
    *,
    class_weights: torch.Tensor | None = None,
    sample_weights: np.ndarray | None = None,
    multilabel: bool = False,
):
    """Create a Trainer subclass with thesis imbalance handling."""

    class TextThreatWeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            if multilabel:
                labels = labels.float()
                weight = class_weights.to(logits.device) if class_weights is not None else None
                loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
                loss = loss_function(logits, labels)
            else:
                labels = labels.long()
                weight = class_weights.to(logits.device) if class_weights is not None else None
                loss_function = torch.nn.CrossEntropyLoss(weight=weight)
                loss = loss_function(logits, labels)
            return (loss, outputs) if return_outputs else loss

        def get_train_dataloader(self):
            if sample_weights is None:
                return super().get_train_dataloader()
            from torch.utils.data import DataLoader, WeightedRandomSampler

            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
            dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
            return self.accelerator.prepare(dataloader) if hasattr(self, "accelerator") else dataloader

    return TextThreatWeightedTrainer


def multilabel_class_weights(labels: np.ndarray) -> tuple[torch.Tensor, np.ndarray, dict[str, Any]]:
    """Compute BCE positive weights and row weights for multi-label oversampling."""
    labels = labels.astype(float)
    positive_counts = labels.sum(axis=0)
    negative_counts = labels.shape[0] - positive_counts
    pos_weight = np.divide(
        negative_counts,
        positive_counts,
        out=np.ones_like(negative_counts, dtype=float),
        where=positive_counts > 0,
    )
    pos_weight = np.clip(pos_weight, 1.0, 50.0)
    positive_per_row = labels.sum(axis=1)
    row_weight = np.where(
        positive_per_row > 0,
        (labels * pos_weight).sum(axis=1) / np.maximum(positive_per_row, 1.0),
        1.0,
    )
    metadata = {
        "positive_counts": {label: int(count) for label, count in zip(LABELS, positive_counts)},
        "pos_weight": {label: float(weight) for label, weight in zip(LABELS, pos_weight)},
        "sample_weight_min": float(row_weight.min()),
        "sample_weight_max": float(row_weight.max()),
    }
    return torch.tensor(pos_weight, dtype=torch.float), row_weight.astype(float), metadata


def binary_class_weights(labels: np.ndarray) -> tuple[torch.Tensor, np.ndarray, dict[str, Any]]:
    """Compute class and row weights for the Dreaddit binary classifier."""
    labels = labels.astype(int)
    counts = np.bincount(labels, minlength=2)
    total = max(int(counts.sum()), 1)
    weights = np.divide(
        total,
        2.0 * counts,
        out=np.ones(2, dtype=float),
        where=counts > 0,
    )
    row_weight = weights[labels]
    metadata = {
        "class_counts": {"not_stress": int(counts[0]), STRESS_LABEL: int(counts[1])},
        "class_weight": {"not_stress": float(weights[0]), STRESS_LABEL: float(weights[1])},
        "sample_weight_min": float(row_weight.min()),
        "sample_weight_max": float(row_weight.max()),
    }
    return torch.tensor(weights, dtype=torch.float), row_weight.astype(float), metadata


def fit_multilabel_platt_calibrators(logits: np.ndarray, labels: np.ndarray, output_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one Platt logistic calibrator per Jigsaw label and save it."""
    ensure_dir(output_dir)
    calibrated = np.zeros_like(labels, dtype=float)
    calibrators: dict[str, Any] = {}
    for index, label in enumerate(LABELS):
        y_true = labels[:, index].astype(int)
        if len(np.unique(y_true)) < 2:
            probability = float(y_true.mean())
            calibrated[:, index] = probability
            calibrators[label] = {"type": "constant", "probability": probability}
            continue
        calibrator = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        calibrator.fit(logits[:, index].reshape(-1, 1), y_true)
        calibrated[:, index] = calibrator.predict_proba(logits[:, index].reshape(-1, 1))[:, 1]
        calibrators[label] = calibrator
    joblib.dump(calibrators, output_dir / "platt_calibrators.joblib")
    return calibrated, {"method": "platt_logistic_regression_per_label", "path": str(output_dir / "platt_calibrators.joblib")}


def fit_binary_platt_calibrator(logits: np.ndarray, labels: np.ndarray, output_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a Platt calibrator for the Dreaddit stress probability."""
    ensure_dir(output_dir)
    labels = labels.astype(int)
    if len(np.unique(labels)) < 2:
        probability = float(labels.mean())
        stress_probability = np.full(len(labels), probability)
        calibrator: Any = {"type": "constant", "probability": probability}
    else:
        margin = (logits[:, 1] - logits[:, 0]).reshape(-1, 1)
        calibrator = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        calibrator.fit(margin, labels)
        stress_probability = calibrator.predict_proba(margin)[:, 1]
    joblib.dump(calibrator, output_dir / "platt_stress_calibrator.joblib")
    probabilities = np.vstack([1.0 - stress_probability, stress_probability]).T
    return probabilities, {"method": "platt_logistic_regression_binary_margin", "path": str(output_dir / "platt_stress_calibrator.joblib")}


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
    frame = add_text_preprocessing(frame, "comment_text")
    train_frame, val_frame, split_method = multilabel_train_test_split(frame, LABELS)

    train_labels = train_frame[LABELS].to_numpy(dtype=int)
    class_weights, sample_weights, imbalance_metadata = multilabel_class_weights(train_labels)
    if not args.use_class_weights:
        class_weights = None
    if not args.use_weighted_sampler:
        sample_weights = None
    trainer_class = build_weighted_trainer(
        Trainer,
        class_weights=class_weights,
        sample_weights=sample_weights,
        multilabel=True,
    )
    train_dataset = Dataset.from_pandas(train_frame[[TEXT_NORMALIZED_COLUMN, *LABELS]].reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_frame[[TEXT_NORMALIZED_COLUMN, *LABELS]].reset_index(drop=True))

    def add_labels(example):
        example["labels"] = [float(example[label]) for label in LABELS]
        return example

    train_dataset = train_dataset.map(add_labels)
    val_dataset = val_dataset.map(add_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch[TEXT_NORMALIZED_COLUMN], truncation=True, max_length=args.max_length)

    train_ds = train_dataset.map(tokenize, batched=True)
    val_ds = val_dataset.map(tokenize, batched=True)
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
        trainer_class,
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
    prediction_output = trainer.predict(val_ds)
    final_logits = np.asarray(prediction_output.predictions)
    final_labels = np.asarray(prediction_output.label_ids).astype(int)
    if args.calibrate:
        final_probs, calibration_metadata = fit_multilabel_platt_calibrators(final_logits, final_labels, args.output_dir)
    else:
        final_probs = 1.0 / (1.0 + np.exp(-final_logits))
        calibration_metadata = {"method": "uncalibrated_sigmoid"}
    final_metrics = compute_classification_metrics(final_labels, final_probs, LABELS)
    save_full_model_artifact(trainer.model, tokenizer, args.output_dir, training_args)

    metrics = {
        "demo": False,
        "task": "jigsaw",
        "model_name": args.model_name,
        "use_lora": args.use_lora,
        "max_length": int(args.max_length),
        "preprocessing": "lowercase_url_user_number_normalization",
        "split_method": split_method,
        "class_weighted_bce": bool(args.use_class_weights),
        "minority_weighted_sampler": bool(args.use_weighted_sampler),
        "imbalance_metadata": imbalance_metadata,
        "calibration": calibration_metadata,
        "validation_metrics_after_calibration": final_metrics,
        **eval_metrics,
    }
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
    train_frame = add_text_preprocessing(train_frame, text_column)
    if test_frame is not None:
        test_frame = add_text_preprocessing(test_frame, text_column)
    split_method = "provided_dreaddit_test_file"
    if test_frame is not None:
        model_train_frame = train_frame
        train_ds = Dataset.from_pandas(model_train_frame[[TEXT_NORMALIZED_COLUMN, label_column]].reset_index(drop=True))
        val_ds = Dataset.from_pandas(test_frame[[TEXT_NORMALIZED_COLUMN, label_column]].reset_index(drop=True))
    else:
        model_train_frame, split_val, split_method = binary_train_test_split(train_frame, label_column)
        train_ds = Dataset.from_pandas(model_train_frame[[TEXT_NORMALIZED_COLUMN, label_column]].reset_index(drop=True))
        val_ds = Dataset.from_pandas(split_val[[TEXT_NORMALIZED_COLUMN, label_column]].reset_index(drop=True))
    class_weights, sample_weights, imbalance_metadata = binary_class_weights(model_train_frame[label_column].to_numpy(dtype=int))
    if not args.use_class_weights:
        class_weights = None
    if not args.use_weighted_sampler:
        sample_weights = None
    trainer_class = build_weighted_trainer(
        Trainer,
        class_weights=class_weights,
        sample_weights=sample_weights,
        multilabel=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        tokenized = tokenizer(batch[TEXT_NORMALIZED_COLUMN], truncation=True, max_length=args.max_length)
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
        trainer_class,
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
    prediction_output = trainer.predict(val_ds)
    final_logits = np.asarray(prediction_output.predictions)
    final_labels = np.asarray(prediction_output.label_ids).astype(int)
    if args.calibrate:
        final_probs, calibration_metadata = fit_binary_platt_calibrator(final_logits, final_labels, args.output_dir)
    else:
        exp = np.exp(final_logits - final_logits.max(axis=1, keepdims=True))
        final_probs = exp / exp.sum(axis=1, keepdims=True)
        calibration_metadata = {"method": "uncalibrated_softmax"}
    stress_prob = final_probs[:, 1]
    final_pred = (stress_prob >= 0.5).astype(int)
    final_metrics: dict[str, Any] = {"macro_f1": float(f1_score(final_labels, final_pred, average="macro", zero_division=0))}
    if len(np.unique(final_labels)) > 1:
        final_metrics["roc_auc"] = float(roc_auc_score(final_labels, stress_prob))
        final_metrics["pr_auc"] = float(average_precision_score(final_labels, stress_prob))
    save_full_model_artifact(trainer.model, tokenizer, args.output_dir, training_args)

    metrics = {
        "demo": False,
        "task": "dreaddit",
        "model_name": args.model_name,
        "use_lora": args.use_lora,
        "max_length": int(args.max_length),
        "preprocessing": "lowercase_url_user_number_normalization",
        "split_method": split_method,
        "class_weighted_ce": bool(args.use_class_weights),
        "minority_weighted_sampler": bool(args.use_weighted_sampler),
        "imbalance_metadata": imbalance_metadata,
        "calibration": calibration_metadata,
        "validation_metrics_after_calibration": final_metrics,
        **eval_metrics,
    }
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
    parser.add_argument("--max-length", type=int, default=231)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--no-lora", dest="use_lora", action="store_false", help="Disable PEFT LoRA adapters.")
    parser.add_argument("--no-class-weights", dest="use_class_weights", action="store_false", help="Disable class-weighted loss.")
    parser.add_argument("--no-weighted-sampler", dest="use_weighted_sampler", action="store_false", help="Disable minority weighted sampling.")
    parser.add_argument("--no-calibration", dest="calibrate", action="store_false", help="Disable post-hoc Platt calibration.")
    parser.set_defaults(use_lora=True, use_class_weights=True, use_weighted_sampler=True, calibrate=True)
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
