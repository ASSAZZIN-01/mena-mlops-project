"""Train and evaluate a six-layer student from the canonical AraBERT teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from mena_mlops.training.data import (
    build_label_maps,
    encode_labels,
    load_split,
    split_summary,
)
from mena_mlops.training.evaluation import evaluate_predictions

LABEL_NAMES = ["negative", "neutral", "positive"]


class ReviewDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(value[index])
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(config: dict[str, Any]) -> torch.device:
    if bool(config.get("require_gpu", True)) and not torch.cuda.is_available():
        raise RuntimeError(
            "Distillation requires a CUDA-capable execution environment; "
            f"torch={torch.__version__}, cuda_build={torch.version.cuda}"
        )
    requested = str(config.get("device", "cuda"))
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def build_student(
    teacher_path: Path,
    tokenizer: Any,
    *,
    student_layers: int,
    num_labels: int,
) -> Any:
    teacher_config = AutoConfig.from_pretrained(teacher_path)
    if int(teacher_config.num_hidden_layers) <= student_layers:
        raise ValueError(
            "student_layers must be smaller than the teacher layer count"
        )
    teacher_config.num_hidden_layers = student_layers
    teacher_config.num_labels = num_labels
    teacher_config.id2label = {index: label for index, label in enumerate(LABEL_NAMES)}
    teacher_config.label2id = {label: index for index, label in enumerate(LABEL_NAMES)}
    student = AutoModelForSequenceClassification.from_config(teacher_config)
    student.resize_token_embeddings(len(tokenizer))
    return student


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def evaluate(
    model: Any,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    predictions: list[int] = []
    labels: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            output = model(**batch)
            total_loss += float(output.loss.detach().cpu())
            predictions.extend(output.logits.argmax(dim=-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
    metrics = evaluate_predictions(predictions, labels, LABEL_NAMES)
    return total_loss / max(1, len(loader)), metrics


def train_epoch(
    student: Any,
    teacher: Any,
    loader: DataLoader,
    optimizer: Any,
    device: torch.device,
    *,
    alpha: float,
    temperature: float,
    accumulation_steps: int,
) -> float:
    student.train()
    teacher.eval()
    optimizer.zero_grad()
    total_loss = 0.0
    for step, batch in enumerate(loader):
        batch = move_batch(batch, device)
        labels = batch["labels"]
        model_inputs = {key: value for key, value in batch.items() if key != "labels"}
        with torch.no_grad():
            teacher_logits = teacher(**model_inputs).logits
        student_logits = student(**batch).logits
        supervised = torch.nn.functional.cross_entropy(student_logits, labels)
        distillation = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student_logits / temperature, dim=-1),
            torch.nn.functional.softmax(teacher_logits / temperature, dim=-1),
            reduction="batchmean",
        ) * temperature**2
        loss = (alpha * supervised + (1.0 - alpha) * distillation)
        (loss / accumulation_steps).backward()
        if (step + 1) % accumulation_steps == 0 or step + 1 == len(loader):
            optimizer.step()
            optimizer.zero_grad()
        total_loss += float(loss.detach().cpu())
    return total_loss / max(1, len(loader))


def log_metrics(prefix: str, metrics: dict[str, Any]) -> None:
    mlflow.log_metrics(
        {
            f"{prefix}_accuracy": metrics["accuracy"],
            f"{prefix}_macro_f1": metrics["macro_f1"],
            f"{prefix}_weighted_f1": metrics["weighted_f1"],
            f"{prefix}_neutral_f1": metrics["per_class"]["neutral"]["f1"],
        }
    )
    for label, values in metrics["per_class"].items():
        mlflow.log_metric(f"{prefix}_{label}_f1", values["f1"])


def run(config_path: Path) -> str:
    load_dotenv()
    config = load_config(config_path)
    set_seed(int(config["seed"]))
    device = select_device(config)
    data_dir = Path(config["paths"]["data_dir"])
    teacher_path = Path(config["teacher_model"])
    frames = {
        split: load_split(data_dir, split)
        for split in ("train", "validation", "test")
    }
    label_to_id, _ = build_label_maps(LABEL_NAMES)
    tokenizer = AutoTokenizer.from_pretrained(teacher_path)
    teacher = AutoModelForSequenceClassification.from_pretrained(
        teacher_path,
        num_labels=len(LABEL_NAMES),
    ).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student = build_student(
        teacher_path,
        tokenizer,
        student_layers=int(config["student_layers"]),
        num_labels=len(LABEL_NAMES),
    ).to(device)
    datasets = {
        split: ReviewDataset(
            frames[split]["text"].astype(str).tolist(),
            encode_labels(frames[split], label_to_id),
            tokenizer,
            int(config["max_length"]),
        )
        for split in frames
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=int(config["batch_size"]),
            shuffle=True,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=int(config["eval_batch_size"]),
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=int(config["eval_batch_size"]),
        ),
    }
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = str(config["mlflow"]["experiment_name"])
    mlflow.set_experiment(experiment_name)
    output_dir = Path(config["paths"]["output_dir"])
    report_dir = Path(config["paths"]["report_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run(run_name=str(config["mlflow"]["run_name"])) as active_run:
        mlflow.log_params(
            {
                "teacher_model": str(teacher_path),
                "teacher_layers": int(
                    AutoConfig.from_pretrained(teacher_path).num_hidden_layers
                ),
                "student_layers": int(config["student_layers"]),
                "alpha": float(config["alpha"]),
                "temperature": float(config["temperature"]),
                "learning_rate": float(config["learning_rate"]),
                "batch_size": int(config["batch_size"]),
                "epochs": int(config["epochs"]),
                "device": str(device),
                "cuda_device": (
                    torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
                ),
                "train_dataset_sha256": digest(data_dir / "train.parquet"),
                "validation_dataset_sha256": digest(data_dir / "validation.parquet"),
                "test_dataset_sha256": digest(data_dir / "test.parquet"),
            }
        )
        mlflow.log_dict(config, "distillation_config.json")
        mlflow.log_dict(
            {split: split_summary(frame) for split, frame in frames.items()},
            "data_summary.json",
        )
        best_f1 = -1.0
        stale_epochs = 0
        for epoch in range(int(config["epochs"])):
            train_loss = train_epoch(
                student,
                teacher,
                loaders["train"],
                optimizer,
                device,
                alpha=float(config["alpha"]),
                temperature=float(config["temperature"]),
                accumulation_steps=int(config["gradient_accumulation_steps"]),
            )
            validation_loss, validation = evaluate(
                student, loaders["validation"], device
            )
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "validation_macro_f1": validation["macro_f1"],
                },
                step=epoch,
            )
            if validation["macro_f1"] > best_f1:
                best_f1 = validation["macro_f1"]
                stale_epochs = 0
                student.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
            else:
                stale_epochs += 1
            if stale_epochs >= int(config["early_stopping_patience"]):
                break
        student = AutoModelForSequenceClassification.from_pretrained(
            output_dir
        ).to(device)
        reports: dict[str, Any] = {}
        for split in ("validation", "test"):
            _, metrics = evaluate(student, loaders[split], device)
            reports[split] = metrics
            log_metrics(split, metrics)
        report_path = report_dir / "evaluation.json"
        report_path.write_text(
            json.dumps(reports, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        mlflow.log_artifact(str(report_path), artifact_path="reports")
        mlflow.log_artifacts(str(output_dir), artifact_path="student-model")
        if device.type == "cuda":
            mlflow.log_metric(
                "peak_cuda_memory_mb",
                torch.cuda.max_memory_allocated(device) / (1024 * 1024),
            )
        return active_run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/distillation.yaml")
    )
    args = parser.parse_args()
    run_id = run(args.config)
    print(json.dumps({"mlflow_run_id": run_id}))


if __name__ == "__main__":
    main()
