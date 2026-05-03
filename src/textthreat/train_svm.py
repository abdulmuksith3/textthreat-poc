"""Train the SVM + TF-IDF baseline for Jigsaw multi-label classification."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from .constants import LABELS, RANDOM_SEED
from .data import JIGSAW_TRAIN_PATH, load_jigsaw
from .evaluate import compute_classification_metrics
from .utils import MODELS_DIR, RESULTS_DIR, ensure_dir, write_json


def calibrated_linear_svc(cv: int) -> CalibratedClassifierCV:
    """Create a version-tolerant calibrated LinearSVC."""
    base = LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_SEED)
    try:
        return CalibratedClassifierCV(estimator=base, cv=cv, method="sigmoid")
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base, cv=cv, method="sigmoid")


def train_svm(
    data_path: Path,
    output_dir: Path,
    results_path: Path,
    *,
    sample_size: int | None = None,
    calibration_cv: int = 5,
) -> dict[str, Any]:
    """Train and evaluate the thesis SVM baseline."""
    frame = load_jigsaw(data_path)
    if sample_size:
        frame = frame.sample(n=min(sample_size, len(frame)), random_state=RANDOM_SEED)

    x_train, x_val, y_train, y_val = train_test_split(
        frame["comment_text"].astype(str),
        frame[LABELS].to_numpy(dtype=int),
        test_size=0.2,
        random_state=RANDOM_SEED,
    )

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
        min_df=2,
    )
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_val_tfidf = vectorizer.transform(x_val)

    classifier = OneVsRestClassifier(calibrated_linear_svc(calibration_cv), n_jobs=None)
    classifier.fit(x_train_tfidf, y_train)
    probabilities = classifier.predict_proba(x_val_tfidf)

    metrics = compute_classification_metrics(y_val, np.asarray(probabilities), LABELS)
    metrics.update(
        {
            "demo": False,
            "model": "svm_tfidf",
            "feature_extractor": "tfidf_unigrams_bigrams",
            "max_features": 50000,
            "calibration": "CalibratedClassifierCV sigmoid",
            "train_size": int(len(x_train)),
            "validation_size": int(len(x_val)),
        }
    )

    ensure_dir(output_dir)
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.joblib")
    joblib.dump(classifier, output_dir / "svm_calibrated_ovr.joblib")
    write_json(results_path, metrics)
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train TextThreat SVM + TF-IDF baseline.")
    parser.add_argument("--data", type=Path, default=JIGSAW_TRAIN_PATH)
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR / "svm_tfidf")
    parser.add_argument("--results", type=Path, default=RESULTS_DIR / "svm_metrics.json")
    parser.add_argument("--sample-size", type=int, help="Optional subset size for fast experiments.")
    parser.add_argument("--calibration-cv", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    metrics = train_svm(args.data, args.output_dir, args.results, sample_size=args.sample_size, calibration_cv=args.calibration_cv)
    print(f"Wrote SVM metrics to {args.results}")
    return metrics


if __name__ == "__main__":
    main()
