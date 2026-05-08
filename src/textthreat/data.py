"""Dataset loading and preprocessing helpers for Jigsaw and Dreaddit."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .constants import LABELS, RANDOM_SEED
from .utils import DATA_DIR


JIGSAW_TRAIN_PATH = DATA_DIR / "jigsaw" / "train.csv"
DREADDIT_TRAIN_PATH = DATA_DIR / "dreaddit" / "dreaddit-train.csv"
DREADDIT_TEST_PATH = DATA_DIR / "dreaddit" / "dreaddit-test.csv"
TEXT_NORMALIZED_COLUMN = "text_normalized"

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
USER_RE = re.compile(r"(?<!\w)@\w+")
NUMBER_RE = re.compile(r"\b\d+(?:[\.,]\d+)*\b")
SPACE_RE = re.compile(r"\s+")


def require_file(path: Path, description: str) -> Path:
    """Return a path or raise a helpful dataset setup error."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}. Place the dataset file there locally. "
            "Raw Kaggle/Dreaddit data is intentionally not committed."
        )
    return path


def normalize_social_text(text: object) -> str:
    """Normalize social text with the thesis preprocessing policy."""
    normalized = str(text).lower()
    normalized = URL_RE.sub("[URL]", normalized)
    normalized = USER_RE.sub("[USER]", normalized)
    normalized = NUMBER_RE.sub("[NUM]", normalized)
    normalized = SPACE_RE.sub(" ", normalized).strip()
    return normalized


def add_text_preprocessing(
    frame: pd.DataFrame,
    source_column: str,
    target_column: str = TEXT_NORMALIZED_COLUMN,
) -> pd.DataFrame:
    """Return a copy with URL/user/number-normalized text."""
    prepared = frame.copy()
    prepared[target_column] = prepared[source_column].map(normalize_social_text)
    return prepared


def multilabel_train_test_split(
    frame: pd.DataFrame,
    labels: list[str] = LABELS,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Split a multi-label frame using iterative stratification when available."""
    y = frame[labels].astype(int).to_numpy()
    indices = np.arange(len(frame))
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=seed,
        )
        train_idx, test_idx = next(splitter.split(indices, y))
        method = "iterative_multilabel_stratification"
    except ImportError:
        label_count = y.sum(axis=1)
        primary_label = np.where(y.any(axis=1), y.argmax(axis=1) + 1, 0)
        stratify_key = [f"{count}:{primary}" for count, primary in zip(label_count, primary_label)]
        try:
            train_idx, test_idx = train_test_split(
                indices,
                test_size=test_size,
                random_state=seed,
                stratify=stratify_key,
            )
            method = "cardinality_primary_label_stratified_fallback"
        except ValueError:
            train_idx, test_idx = train_test_split(
                indices,
                test_size=test_size,
                random_state=seed,
            )
            method = "random_fallback"
    train = frame.iloc[train_idx].reset_index(drop=True)
    test = frame.iloc[test_idx].reset_index(drop=True)
    return train, test, method


def binary_train_test_split(
    frame: pd.DataFrame,
    label_column: str,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Split a binary-label frame using stratification where possible."""
    stratify = frame[label_column] if frame[label_column].nunique() > 1 else None
    try:
        train, test = train_test_split(
            frame,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
        method = "binary_label_stratification" if stratify is not None else "random_single_class"
    except ValueError:
        train, test = train_test_split(frame, test_size=test_size, random_state=seed)
        method = "random_fallback"
    return train.reset_index(drop=True), test.reset_index(drop=True), method


def load_jigsaw(path: Path = JIGSAW_TRAIN_PATH) -> pd.DataFrame:
    """Load Jigsaw Toxic Comment train.csv with the six thesis labels."""
    path = require_file(path, "Jigsaw train.csv")
    frame = pd.read_csv(path)
    missing = ["comment_text", *[label for label in LABELS if label not in frame.columns]]
    missing = [column for column in missing if column not in frame.columns]
    if missing:
        raise ValueError(f"Jigsaw file {path} is missing required columns: {missing}")
    frame["comment_text"] = frame["comment_text"].astype(str)
    frame[LABELS] = frame[LABELS].astype(int)
    return frame


def first_matching_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    """Find a likely column name using exact and fuzzy matching."""
    lower_to_original = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    for column in columns:
        lowered = column.lower()
        if any(candidate.lower() in lowered for candidate in candidates):
            return column
    return None


def dreaddit_columns(frame: pd.DataFrame) -> tuple[str, str]:
    """Resolve flexible Dreaddit text and binary label column names."""
    text_column = first_matching_column(frame.columns, ["text", "sentence", "comment", "body"])
    label_column = first_matching_column(frame.columns, ["label", "stress", "is_stress", "target"])
    if not text_column or not label_column:
        raise ValueError(
            "Could not resolve Dreaddit text/label columns. Expected text/sentence-like "
            "and label/stress/is_stress-like columns."
        )
    return text_column, label_column


def load_dreaddit(
    train_path: Path = DREADDIT_TRAIN_PATH,
    test_path: Path | None = DREADDIT_TEST_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame | None, str, str]:
    """Load Dreaddit train/test CSVs and resolve flexible columns."""
    train_path = require_file(train_path, "Dreaddit train CSV")
    train = pd.read_csv(train_path)
    text_column, label_column = dreaddit_columns(train)
    train[text_column] = train[text_column].astype(str)
    train[label_column] = train[label_column].astype(int)

    test = None
    if test_path and test_path.exists():
        test = pd.read_csv(test_path)
        test_text, test_label = dreaddit_columns(test)
        if test_text != text_column:
            test = test.rename(columns={test_text: text_column})
        if test_label != label_column:
            test = test.rename(columns={test_label: label_column})
        test[text_column] = test[text_column].astype(str)
        test[label_column] = test[label_column].astype(int)
    return train, test, text_column, label_column
