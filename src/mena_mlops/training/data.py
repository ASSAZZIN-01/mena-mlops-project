"""Load the DVC-produced canonical datasets for model training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = {"id", "text", "label", "source"}


def load_split(data_dir: Path, split: str) -> pd.DataFrame:
    path = data_dir / f"{split}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing DVC data split: {path}")
    frame = pd.read_parquet(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing canonical columns: {sorted(missing)}")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError(f"{path} contains null canonical values")
    return frame[["id", "text", "label", "source"]].reset_index(drop=True)


def build_label_maps(labels: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("labels must contain at least two unique values")
    label_to_id = {label: index for index, label in enumerate(labels)}
    return label_to_id, {index: label for label, index in label_to_id.items()}


def encode_labels(frame: pd.DataFrame, label_to_id: dict[str, int]) -> list[int]:
    unknown = sorted(set(frame["label"]) - set(label_to_id))
    if unknown:
        raise ValueError(f"Unknown labels found in data: {unknown}")
    return [label_to_id[label] for label in frame["label"]]


def split_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(frame),
        "labels": frame["label"].value_counts().to_dict(),
        "sources": frame["source"].value_counts().to_dict(),
    }
