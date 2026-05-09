"""Train the SVM + TF-IDF baseline for Jigsaw multi-label classification."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from .constants import LABELS, RANDOM_SEED
from .data import (
    JIGSAW_TRAIN_PATH,
    TEXT_NORMALIZED_COLUMN,
    add_text_preprocessing,
    load_jigsaw,
    multilabel_train_test_split,
)
from .evaluate import compute_classification_metrics
from .utils import MODELS_DIR, RESULTS_DIR, ensure_dir, write_json


def calibrated_linear_svc(cv: int) -> CalibratedClassifierCV:
    """Create a version-tolerant calibrated LinearSVC."""
    base = LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_SEED)
    try:
        return CalibratedClassifierCV(estimator=base, cv=cv, method="sigmoid")
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base, cv=cv, method="sigmoid")


def train_one_label_classifier(
    x_train_tfidf: Any,
    y_train: np.ndarray,
    calibration_cv: int,
    *,
    use_smote: bool,
) -> tuple[Any, dict[str, Any]]:
    """Train one calibrated binary classifier, applying SMOTE only to TF-IDF vectors."""
    y_train = np.asarray(y_train, dtype=int)
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    metadata: dict[str, Any] = {
        "positive_count": positives,
        "negative_count": negatives,
        "smote_applied": False,
        "classifier": "CalibratedClassifierCV(LinearSVC)",
    }

    if len(np.unique(y_train)) < 2:
        classifier = DummyClassifier(strategy="constant", constant=int(y_train[0]))
        classifier.fit(x_train_tfidf, y_train)
        metadata["classifier"] = "DummyClassifier(constant)"
        return classifier, metadata

    x_fit = x_train_tfidf
    y_fit = y_train
    minority_count = min(positives, negatives)
    if use_smote and minority_count >= 2:
        k_neighbors = min(5, minority_count - 1)
        sampler = SMOTE(random_state=RANDOM_SEED, k_neighbors=k_neighbors)
        x_fit, y_fit = sampler.fit_resample(x_train_tfidf, y_train)
        metadata.update(
            {
                "smote_applied": True,
                "smote_k_neighbors": int(k_neighbors),
                "resampled_positive_count": int(np.asarray(y_fit).sum()),
                "resampled_negative_count": int(len(y_fit) - np.asarray(y_fit).sum()),
            }
        )

    class_counts = np.bincount(np.asarray(y_fit, dtype=int), minlength=2)
    smallest_class = int(class_counts[class_counts > 0].min())
    if smallest_class >= 2:
        effective_cv = min(calibration_cv, smallest_class)
        classifier = calibrated_linear_svc(max(2, effective_cv))
        classifier.fit(x_fit, y_fit)
        metadata["calibration_cv"] = int(max(2, effective_cv))
        return classifier, metadata

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    classifier.fit(x_fit, y_fit)
    metadata["classifier"] = "LogisticRegression(fallback_for_small_sample)"
    return classifier, metadata


def train_svm(
    data_path: Path,
    output_dir: Path,
    results_path: Path,
    *,
    sample_size: int | None = None,
    calibration_cv: int = 5,
    use_smote: bool = True,
) -> dict[str, Any]:
    """Train and evaluate the thesis SVM baseline."""
    frame = load_jigsaw(data_path)
    if sample_size:
        frame = frame.sample(n=min(sample_size, len(frame)), random_state=RANDOM_SEED)
    frame = add_text_preprocessing(frame, "comment_text")

    train_frame, val_frame, split_method = multilabel_train_test_split(frame, LABELS)
    x_train = train_frame[TEXT_NORMALIZED_COLUMN].astype(str)
    x_val = val_frame[TEXT_NORMALIZED_COLUMN].astype(str)
    y_train = train_frame[LABELS].to_numpy(dtype=int)
    y_val = val_frame[LABELS].to_numpy(dtype=int)

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
        min_df=2,
    )
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_val_tfidf = vectorizer.transform(x_val)

    classifiers = {}
    label_metadata = {}
    probability_columns = []
    for label_index, label in enumerate(LABELS):
        classifier, metadata = train_one_label_classifier(
            x_train_tfidf,
            y_train[:, label_index],
            calibration_cv,
            use_smote=use_smote,
        )
        classifiers[label] = classifier
        label_metadata[label] = metadata
        label_probabilities = classifier.predict_proba(x_val_tfidf)
        if label_probabilities.shape[1] == 1:
            only_class = int(getattr(classifier, "classes_", [0])[0])
            positive_probability = np.full(len(x_val), 1.0 if only_class == 1 else 0.0)
        else:
            positive_probability = label_probabilities[:, 1]
        probability_columns.append(positive_probability)
    probabilities = np.vstack(probability_columns).T

    metrics = compute_classification_metrics(y_val, np.asarray(probabilities), LABELS)
    metrics.update(
        {
            "demo": False,
            "model": "svm_tfidf",
            "feature_extractor": "tfidf_unigrams_bigrams",
            "max_features": 50000,
            "calibration": "CalibratedClassifierCV sigmoid",
            "preprocessing": "lowercase_url_user_number_normalization",
            "split_method": split_method,
            "smote_scope": "tfidf_feature_vectors_per_label",
            "smote_enabled": bool(use_smote),
            "label_training_metadata": label_metadata,
            "train_size": int(len(x_train)),
            "validation_size": int(len(x_val)),
        }
    )

    ensure_dir(output_dir)
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.joblib")
    joblib.dump(
        {
            "labels": LABELS,
            "classifiers": classifiers,
            "preprocessing": "lowercase_url_user_number_normalization",
            "split_method": split_method,
            "smote_enabled": bool(use_smote),
        },
        output_dir / "svm_calibrated_ovr.joblib",
    )
    write_json(results_path, metrics)
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train TextThreat SVM + TF-IDF baseline.")
    parser.add_argument("--data", type=Path, default=JIGSAW_TRAIN_PATH)
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR / "svm_tfidf")
    parser.add_argument("--results", type=Path, default=RESULTS_DIR / "svm_metrics.json")
    parser.add_argument("--sample-size", type=int, help="Optional subset size for fast experiments.")
    parser.add_argument("--calibration-cv", type=int, default=5)
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE on TF-IDF feature vectors.")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    metrics = train_svm(
        args.data,
        args.output_dir,
        args.results,
        sample_size=args.sample_size,
        calibration_cv=args.calibration_cv,
        use_smote=not args.no_smote,
    )
    print(f"Wrote SVM metrics to {args.results}")
    return metrics


if __name__ == "__main__":
    main()
