"""Fine-tune AraBERT and track training plus evaluation in one MLflow run."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from mena_mlops.training.data import (
    build_label_maps,
    encode_labels,
    load_split,
    split_summary,
)
from mena_mlops.training.evaluation import evaluate_by_source, evaluate_predictions


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


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(config: dict[str, Any]) -> torch.device:
    requested = config["training"]["device"]
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def run_epoch(
    model: Any,
    loader: DataLoader,
    device: torch.device,
    optimizer: Any | None,
    accumulation_steps: int,
) -> tuple[float, list[int], list[int]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    predictions: list[int] = []
    labels: list[int] = []
    if training:
        optimizer.zero_grad()
    for step, batch in enumerate(loader):
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.set_grad_enabled(training):
            output = model(**batch)
            loss = output.loss
            if training:
                (loss / accumulation_steps).backward()
                if (step + 1) % accumulation_steps == 0 or step + 1 == len(loader):
                    optimizer.step()
                    optimizer.zero_grad()
        total_loss += float(loss.detach().cpu())
        predictions.extend(output.logits.argmax(dim=-1).detach().cpu().tolist())
        labels.extend(batch["labels"].detach().cpu().tolist())
    return total_loss / max(1, len(loader)), predictions, labels


def log_evaluation(prefix: str, metrics: dict[str, Any]) -> None:
    mlflow.log_metrics(
        {
            f"{prefix}_accuracy": metrics["accuracy"],
            f"{prefix}_macro_f1": metrics["macro_f1"],
            f"{prefix}_weighted_f1": metrics["weighted_f1"],
        }
    )
    for label, values in metrics["per_class"].items():
        mlflow.log_metrics(
            {
                f"{prefix}_{label}_precision": values["precision"],
                f"{prefix}_{label}_recall": values["recall"],
                f"{prefix}_{label}_f1": values["f1"],
            }
        )


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log_dataset_inputs(
    frames: dict[str, Any], data_dir: Path
) -> dict[str, str]:
    digests = {}
    for split, frame in frames.items():
        path = data_dir / f"{split}.parquet"
        digest = file_digest(path)
        dataset = mlflow.data.from_pandas(
            frame,
            source=str(path),
            name=f"mena-mlops-{split}",
            digest=digest[:32],
            targets="label",
        )
        mlflow.log_input(
            dataset,
            context=f"{split}_dataset",
            tags={"pipeline": "dvc", "split": split},
        )
        digests[split] = digest
    return digests


def run_training(config_path: Path) -> str:
    load_dotenv()
    config = load_yaml(config_path)
    data_config = load_yaml(Path(config["paths"]["data_config"]))
    set_seed(int(config["seed"]))
    device = select_device(config)
    label_names = list(data_config["labels"]["output"])
    label_to_id, id_to_label = build_label_maps(label_names)
    data_dir = Path(config["paths"]["data_dir"])
    frames = {
        split: load_split(data_dir, split) for split in ["train", "validation", "test"]
    }

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"]["name"],
        num_labels=len(label_names),
        id2label=id_to_label,
        label2id=label_to_id,
    ).to(device)
    train_config = config["training"]
    datasets = {
        split: ReviewDataset(
            frame["text"].tolist(),
            encode_labels(frame, label_to_id),
            tokenizer,
            int(train_config["max_length"]),
        )
        for split, frame in frames.items()
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=int(train_config["batch_size"]),
            shuffle=True,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=int(train_config["eval_batch_size"]),
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=int(train_config["eval_batch_size"]),
        ),
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    mlflow.set_tracking_uri(str(config["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(str(config["mlflow"]["experiment_name"]))

    with mlflow.start_run(run_name=str(config["mlflow"]["run_name"])) as run:
        mlflow.log_params(
            {
                "model_name": config["model"]["name"],
                "max_length": train_config["max_length"],
                "batch_size": train_config["batch_size"],
                "eval_batch_size": train_config["eval_batch_size"],
                "gradient_accumulation_steps": train_config[
                    "gradient_accumulation_steps"
                ],
                "epochs": train_config["epochs"],
                "learning_rate": train_config["learning_rate"],
                "weight_decay": train_config["weight_decay"],
                "seed": config["seed"],
                "device": str(device),
                "cuda_available": torch.cuda.is_available(),
                "data_mode": data_config["split"]["debug"],
            }
        )
        mlflow.log_dict(config, "training_config.json")
        mlflow.log_dict(data_config["labels"], "label_mapping.json")
        mlflow.log_dict(
            {split: split_summary(frame) for split, frame in frames.items()},
            "data_summary.json",
        )
        dataset_digests = log_dataset_inputs(frames, data_dir)
        mlflow.log_params(
            {
                f"{split}_dataset_digest": digest
                for split, digest in dataset_digests.items()
            }
        )
        best_validation_f1 = -1.0
        output_dir = Path(config["paths"]["output_dir"])
        for epoch in range(int(train_config["epochs"])):
            train_loss, _, _ = run_epoch(
                model,
                loaders["train"],
                device,
                optimizer,
                int(train_config["gradient_accumulation_steps"]),
            )
            validation_loss, validation_predictions, validation_labels = run_epoch(
                model, loaders["validation"], device, None, 1
            )
            validation = evaluate_predictions(
                validation_labels, validation_predictions, label_names
            )
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "validation_macro_f1": validation["macro_f1"],
                },
                step=epoch,
            )
            if validation["macro_f1"] > best_validation_f1:
                best_validation_f1 = validation["macro_f1"]
                output_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

        model = AutoModelForSequenceClassification.from_pretrained(
            output_dir
        ).to(device)
        reports = {}
        for split in ["validation", "test"]:
            _, predictions, labels = run_epoch(
                model, loaders[split], device, None, 1
            )
            metrics = evaluate_predictions(labels, predictions, label_names)
            reports[split] = metrics
            log_evaluation(split, metrics)
            if split == "test":
                reports["test_by_source"] = evaluate_by_source(
                    frames[split], labels, predictions, label_names
                )
                for source, source_metrics in reports["test_by_source"].items():
                    mlflow.log_metric(
                        f"test_{source}_macro_f1", source_metrics["macro_f1"]
                    )

        report_dir = Path(config["paths"]["report_dir"])
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "evaluation.json"
        report_path.write_text(
            json.dumps(reports, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        mlflow.log_artifact(str(report_path), artifact_path="reports")
        mlflow.log_artifacts(str(output_dir), artifact_path="model")
        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            name="registered_model",
            registered_model_name=str(config["mlflow"]["registered_model_name"]),
            task="text-classification",
            pip_requirements=[
                "torch==2.5.1",
                "transformers==5.18.0",
                "safetensors>=0.4",
            ],
        )
        return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    args = parser.parse_args()
    run_id = run_training(args.config)
    print(f"MLflow run: {run_id}")


if __name__ == "__main__":
    main()
