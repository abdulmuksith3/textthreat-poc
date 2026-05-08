"""Dataset loading helpers for Jigsaw and Dreaddit."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .constants import LABELS
from .utils import DATA_DIR


JIGSAW_TRAIN_PATH = DATA_DIR / "jigsaw" / "train.csv"
DREADDIT_TRAIN_PATH = DATA_DIR / "dreaddit" / "dreaddit-train.csv"
DREADDIT_TEST_PATH = DATA_DIR / "dreaddit" / "dreaddit-test.csv"


def require_file(path: Path, description: str) -> Path:
    """Return a path or raise a helpful dataset setup error."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}. Place the dataset file there locally. "
            "Raw Kaggle/Dreaddit data is intentionally not committed."
        )
    return path


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
