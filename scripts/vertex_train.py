"""Vertex AI training entry point for the feedback-gated candidate job."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import torch
import yaml
from google.cloud import storage

from mena_mlops.training.train import load_yaml, run_training


def _download(uri: str, destination: Path) -> None:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path:
        raise ValueError(f"expected a GCS URI, got {uri!r}")
    client = storage.Client()
    bucket = client.bucket(parsed.netloc)
    bucket.blob(parsed.path.lstrip("/")).download_to_filename(destination)


def main() -> None:
    input_prefix = os.environ["VERTEX_INPUT_PREFIX"].rstrip("/")
    with tempfile.TemporaryDirectory(prefix="vertex-retraining-") as temporary:
        root = Path(temporary)
        data_dir = root / "data"
        data_dir.mkdir()
        for split in ("train", "validation", "test", "feedback"):
            _download(f"{input_prefix}/{split}.parquet", data_dir / f"{split}.parquet")

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
        print(f"MLflow candidate run: {run_id}")


if __name__ == "__main__":
    main()
