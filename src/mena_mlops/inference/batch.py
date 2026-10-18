"""Batch scoring for CSV, JSONL, and Parquet review datasets."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import pandas as pd


class TextPredictor(Protocol):
    model_version: str

    def predict(self, text: str) -> tuple[str, dict[str, float]]: ...


DEFAULT_BATCH_SIZE = 32


def read_batch_input(path: Path) -> pd.DataFrame:
    """Read a batch input and require the canonical ``text`` column."""

    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        frame = pd.read_json(path, lines=True)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("batch input must use .csv, .parquet, .jsonl, or .ndjson")
    if "text" not in frame.columns:
        raise ValueError("batch input must contain a 'text' column")
    if frame["text"].isna().any() or (frame["text"].astype(str).str.len() == 0).any():
        raise ValueError("batch input contains an empty or null text value")
    return frame


def predict_batch(
    records: Iterable[dict[str, object]],
    predictor: TextPredictor,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict[str, object]]:
    """Score records while preserving their fields and adding prediction metadata."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    scored_at = datetime.now(UTC).isoformat()
    results: list[dict[str, object]] = []
    input_records = list(records)
    for start in range(0, len(input_records), batch_size):
        chunk = input_records[start : start + batch_size]
        texts = [str(record["text"]) for record in chunk]
        batch_predictor = getattr(predictor, "predict_batch", None)
        predictions = (
            batch_predictor(texts)
            if batch_predictor is not None
            else [predictor.predict(text) for text in texts]
        )
        for record, (label, probabilities) in zip(
            chunk, predictions, strict=True
        ):
            result = dict(record)
            result.update(
                {
                    "prediction": label,
                    "probabilities": json.dumps(
                        probabilities,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "model_version": predictor.model_version,
                    "predicted_at": scored_at,
                }
            )
            results.append(result)
    return results


def write_batch_output(records: list[dict[str, object]], path: Path) -> None:
    """Write scored records as JSONL or Parquet based on the output suffix."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(records)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        frame.to_json(path, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError("batch output must use .parquet, .jsonl, or .ndjson")
