"""Vertex AI training entry point for the feedback-gated candidate job."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import torch
import yaml
from google.cloud import storage

from mena_mlops.model_card import build_model_card, write_model_card
from mena_mlops.promotion import evaluate_and_gate_candidate
from mena_mlops.training.train import load_yaml, run_training


def _download(uri: str, destination: Path) -> None:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path:
        raise ValueError(f"expected a GCS URI, got {uri!r}")
    client = storage.Client()
    bucket = client.bucket(parsed.netloc)
    bucket.blob(parsed.path.lstrip("/")).download_to_filename(destination)


def _upload_directory(directory: Path, prefix: str) -> None:
    parsed = urlparse(prefix)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"expected a GCS prefix, got {prefix!r}")
    bucket = storage.Client().bucket(parsed.netloc)
    for path in directory.iterdir():
        if path.is_file():
            bucket.blob(
                f"{parsed.path.lstrip('/')}/{path.name}"
            ).upload_from_filename(path)


def main() -> None:
    input_prefix = os.environ["VERTEX_INPUT_PREFIX"].rstrip("/")
    with tempfile.TemporaryDirectory(prefix="vertex-retraining-") as temporary:
        root = Path(temporary)
        data_dir = root / "data"
        data_dir.mkdir()
        for split in ("train", "validation", "test", "feedback"):
            _download(f"{input_prefix}/{split}.parquet", data_dir / f"{split}.parquet")
        lineage_path = root / "lineage.json"
        lineage_uri = os.environ.get("VERTEX_LINEAGE_URI")
        if lineage_uri:
            _download(lineage_uri, lineage_path)
            lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        else:
            lineage = {
                "training_image": os.environ.get(
                    "VERTEX_TRAINING_IMAGE_URI", "unknown"
                )
            }

        train = pd.read_parquet(data_dir / "train.parquet")
        feedback = pd.read_parquet(data_dir / "feedback.parquet")
        feedback["id"] = feedback["id"].astype(str)
        feedback = feedback[train.columns]
        pd.concat([train, feedback], ignore_index=True).to_parquet(
            data_dir / "train.parquet",
            index=False,
        )

        config = load_yaml(
            Path(os.environ.get("TRAINING_CONFIG", "configs/training.yaml"))
        )
        config["paths"] = dict(config["paths"])
        config["paths"]["data_dir"] = str(data_dir)
        config["paths"]["output_dir"] = str(root / "model")
        config["paths"]["report_dir"] = str(root / "reports")
        config["training"] = dict(config["training"])
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Vertex GPU is unavailable in the training container: "
                f"torch={torch.__version__}, cuda_build={torch.version.cuda}, "
                f"device_count={torch.cuda.device_count()}"
            )
        print(
            "Vertex GPU: "
            f"{torch.cuda.get_device_name(0)} "
            f"(torch={torch.__version__}, cuda_build={torch.version.cuda})",
            flush=True,
        )
        config["training"]["device"] = "cuda"
        config_path = root / "training.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        run_id = run_training(config_path)
        promotion_config = load_yaml(Path("configs/promotion.yaml"))
        quality_config = load_yaml(Path("configs/quality.yaml"))
        gate = promotion_config["quality_gate"]
        optimization = quality_config["optimization"]
        decision = evaluate_and_gate_candidate(
            root / "model",
            data_dir / "test.parquet",
            root / "variants",
            root / "reports" / "promotion_decision.json",
            stable_macro_f1=float(
                quality_config["model_quality"]["baseline_test_macro_f1"]
            ),
            minimum_macro_f1=float(gate["minimum_macro_f1"]),
            minimum_improvement=float(gate["minimum_improvement"]),
            minimum_class_f1={
                str(label): float(value)
                for label, value in gate["minimum_class_f1"].items()
            },
            maximum_macro_f1_drop_percent=float(
                optimization["maximum_macro_f1_drop_percent"]
            ),
            max_length=int(config["training"]["max_length"]),
        )
        card = build_model_card(
            model_id=os.environ.get("VERTEX_MODEL_ID", input_prefix.rsplit("/", 1)[-1]),
            training_job_id=os.environ.get("VERTEX_TRAINING_JOB_ID"),
            mlflow_run_id=run_id,
            config=config,
            data_config=load_yaml(
                Path(os.environ.get("DATA_CONFIG", "configs/data.yaml"))
            ),
            data_dir=data_dir,
            model_dir=root / "model",
            release_dir=root / "variants",
            evaluation=json.loads(
                (root / "reports" / "evaluation.json").read_text(encoding="utf-8")
            ),
            promotion_decision=decision,
            lineage=lineage,
        )
        write_model_card(card, root / "model-card")
        release_prefix = f"{input_prefix}/release"
        _upload_directory(root / "variants", release_prefix)
        _upload_directory(root / "reports", release_prefix)
        _upload_directory(root / "model-card", release_prefix)
        print(
            f"MLflow candidate run: {run_id}; "
            f"promotion status: {decision['status']}; "
            f"selected variant: {decision['selected_variant']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
