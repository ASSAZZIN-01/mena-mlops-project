"""Download, normalize, clean, split, and validate project datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold

OUTPUT_COLUMNS = ["id", "text", "label", "source"]
NAMESPACE = uuid.UUID("4f5b0ab2-7ec1-4c74-99d8-cfda7fe4f58b")


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return config


def stable_id(source: str, row_number: int, text: str) -> str:
    key = f"{source}:{row_number}:{text}"
    return str(uuid.uuid5(NAMESPACE, key))


def normalize_text(value: Any, cleaning: dict[str, Any]) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"[\u0000-\u001f\u007f]", " ", text)
    if cleaning.get("remove_tatweel", True):
        text = text.replace("\u0640", "")
    if cleaning.get("remove_diacritics", False):
        text = "".join(
            char
            for char in text
            if not unicodedata.combining(char)
        )
    return re.sub(r"\s+", " ", text).strip()


def map_label(value: Any, source: dict[str, Any], labels: dict[str, Any]) -> str | None:
    if value is None or pd.isna(value):
        return None
    kind = source.get("labels", {}).get("kind")
    raw = str(value).strip().lower()
    if kind == "rating":
        return labels["rating"].get(raw)
    if kind == "binary":
        return labels["binary"].get(raw)
    if kind == "signed":
        return labels["signed"].get(raw)
    return labels["mappings"].get(raw)


def normalize_frame(
    frame: pd.DataFrame,
    source: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = source["columns"]
    missing = [name for name in columns.values() if name not in frame.columns]
    if missing:
        raise ValueError(f"{source['name']} is missing columns: {missing}")

    cleaning = config["cleaning"]
    text = frame[columns["text"]].map(lambda value: normalize_text(value, cleaning))
    labels = frame[columns["label"]].map(
        lambda value: map_label(value, source, config["labels"])
    )
    result = pd.DataFrame({"text": text, "label": labels})
    result.insert(0, "source", source["name"])
    result.insert(
        0,
        "id",
        [
            stable_id(source["name"], index, value)
            for index, value in zip(frame.index, text, strict=True)
        ],
    )
    return result[OUTPUT_COLUMNS]


def read_tabular(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported tabular file: {path}")


def download_huggingface(source: dict[str, Any], destination: Path) -> Path:
    from datasets import load_dataset

    token = os.getenv("HF_API_TOKEN") or os.getenv("HF_TOKEN")
    dataset = load_dataset(
        source["dataset"],
        split=source.get("split", "train"),
        token=token,
    )
    output = destination / f"{source['name']}.parquet"
    dataset.to_pandas().to_parquet(output, index=False)
    return output


def download_kaggle(source: dict[str, Any], destination: Path) -> Path:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    source_dir = destination / source["name"]
    source_dir.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(
        source["dataset"], path=str(source_dir), unzip=True
    )
    files = sorted(
        path
        for path in source_dir.rglob("*")
        if path.suffix.lower() in {".csv", ".json", ".jsonl", ".parquet"}
    )
    if len(files) != 1:
        raise ValueError(
            f"Expected one tabular file for {source['name']}, found {files}"
        )
    return files[0]


def download_sources(config: dict[str, Any]) -> list[tuple[dict[str, Any], Path]]:
    load_dotenv()
    destination = Path(config["paths"]["raw"])
    destination.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for source in config["download"]["sources"]:
        if source["kind"] == "huggingface":
            path = download_huggingface(source, destination)
        elif source["kind"] == "kaggle":
            path = download_kaggle(source, destination)
        else:
            raise ValueError(f"Unsupported source kind: {source['kind']}")
        downloaded.append((source, path))
    return downloaded


def find_downloaded_sources(
    config: dict[str, Any],
) -> list[tuple[dict[str, Any], Path]]:
    raw = Path(config["paths"]["raw"])
    found = []
    supported = {".csv", ".json", ".jsonl", ".parquet"}
    for source in config["download"]["sources"]:
        root = raw / source["name"] if source["kind"] == "kaggle" else raw
        candidates = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in supported
            and (source["name"] in path.stem or source["kind"] == "kaggle")
        )
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected one downloaded file for {source['name']}, "
                f"found {candidates}"
            )
        found.append((source, candidates[0]))
    return found


def normalize_sources(
    config: dict[str, Any],
    downloaded: list[tuple[dict[str, Any], Path]],
) -> pd.DataFrame:
    frames = [
        normalize_frame(read_tabular(path), source, config)
        for source, path in downloaded
    ]
    combined = pd.concat(frames, ignore_index=True)
    min_length = int(config["cleaning"]["min_text_length"])
    combined = combined[
        combined["text"].str.len().ge(min_length)
        & combined["label"].isin(config["labels"]["output"])
    ].copy()
    return combined.reset_index(drop=True)


def deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    labels_per_text = frame.groupby("text", dropna=False)["label"].nunique()
    conflicting_texts = labels_per_text[labels_per_text > 1].index
    conflict_mask = frame["text"].isin(conflicting_texts)
    conflict_rows = int(conflict_mask.sum())
    without_conflicts = frame.loc[~conflict_mask]
    duplicate_rows = int(without_conflicts.duplicated("text", keep="first").sum())
    cleaned = without_conflicts.drop_duplicates(
        "text", keep="first"
    ).reset_index(drop=True)
    return cleaned, {
        "input_rows": len(frame),
        "conflicting_rows_removed": conflict_rows,
        "conflicting_texts_removed": len(conflicting_texts),
        "duplicate_rows_removed": duplicate_rows,
        "output_rows": len(cleaned),
    }


def split_data(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    split = config["split"]
    key = frame["label"] + "::" + frame["source"]
    outer = StratifiedKFold(
        n_splits=int(split["outer_folds"]),
        shuffle=True,
        random_state=int(config["seed"]),
    )
    fold_numbers = pd.Series(index=frame.index, dtype="int64")
    for fold, (_, selected) in enumerate(outer.split(frame, key)):
        fold_numbers.iloc[selected] = fold

    if split["debug"]:
        selected = frame[fold_numbers == int(split["debug_outer_fold"])].copy()
        inner_key = selected["label"] + "::" + selected["source"]
        inner = StratifiedKFold(
            n_splits=int(split["debug_inner_folds"]),
            shuffle=True,
            random_state=int(config["seed"]),
        )
        inner_folds = pd.Series(index=selected.index, dtype="int64")
        for fold, (_, indexes) in enumerate(inner.split(selected, inner_key)):
            inner_folds.iloc[indexes] = fold
        train = selected[inner_folds.isin(split["debug_train_folds"])]
        validation = selected[inner_folds.isin(split["debug_validation_folds"])]
        test = selected[inner_folds.isin(split["debug_test_folds"])]
    else:
        validation = frame[fold_numbers == int(split["full_validation_fold"])]
        test = frame[fold_numbers == int(split["full_test_fold"])]
        train = frame[~fold_numbers.isin(
            [int(split["full_validation_fold"]), int(split["full_test_fold"])]
        )]
    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def process_pipeline(config_path: Path) -> None:
    config = load_config(config_path)
    downloaded = find_downloaded_sources(config)
    normalized = normalize_sources(config, downloaded)
    cleaned, duplicate_report = deduplicate(normalized)
    interim_path = Path(config["paths"]["interim"]) / "cleaned.parquet"
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(interim_path, index=False)

    splits = split_data(cleaned, config)
    processed = Path(config["paths"]["processed"])
    processed.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        data.to_parquet(processed / f"{name}.parquet", index=False)

    report = {
        "rows": {name: len(data) for name, data in splits.items()},
        "labels": {
            name: data["label"].value_counts().to_dict()
            for name, data in splits.items()
        },
        "sources": {
            name: data["source"].value_counts().to_dict()
            for name, data in splits.items()
        },
        "duplicate_report": duplicate_report,
        "debug": bool(config["split"]["debug"]),
        "sha256": hashlib.sha256(
            cleaned.to_csv(index=False).encode("utf-8")
        ).hexdigest(),
    }
    write_json(Path(config["paths"]["reports"]) / "split_report.json", report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument(
        "--stage", choices=["download", "process", "all"], default="all"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage in {"download", "all"}:
        download_sources(config)
    if args.stage in {"process", "all"}:
        process_pipeline(args.config)


if __name__ == "__main__":
    main()
