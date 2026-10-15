"""Vertex entry point for configurable manual initial-model training."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import yaml
from google.cloud import storage

from mena_mlops.training.train import load_yaml, run_training


def _download(uri: str, destination: Path) -> None:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path:
        raise ValueError(f"expected a GCS URI, got {uri!r}")
    storage.Client().bucket(parsed.netloc).blob(
        parsed.path.lstrip("/")
    ).download_to_filename(destination)


def _upload_tree(directory: Path, prefix: str) -> None:
    parsed = urlparse(prefix)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"expected a GCS prefix, got {prefix!r}")
    bucket = storage.Client().bucket(parsed.netloc)
    for path in directory.rglob("*"):
        if path.is_file():
            relative = path.relative_to(directory).as_posix()
            bucket.blob(f"{parsed.path.lstrip('/')}/{relative}").upload_from_filename(
                path
            )


def main() -> None:
    input_prefix = os.environ["VERTEX_INPUT_PREFIX"].rstrip("/")
    output_prefix = os.environ["VERTEX_OUTPUT_PREFIX"]
    with tempfile.TemporaryDirectory(prefix="vertex-manual-training-") as temporary:
        root = Path(temporary)
        data_dir = root / "data"
        data_dir.mkdir()
        for split in ("train", "validation", "test"):
            _download(
                f"{input_prefix}/{split}.parquet",
                data_dir / f"{split}.parquet",
            )

        training_config = root / "training.yaml"
        data_config = root / "data.yaml"
        _download(os.environ["VERTEX_TRAINING_CONFIG_URI"], training_config)
        _download(os.environ["VERTEX_DATA_CONFIG_URI"], data_config)

        config = load_yaml(training_config)
        config["paths"] = dict(config["paths"])
        config["paths"]["data_config"] = str(data_config)
        config["paths"]["data_dir"] = str(data_dir)
        config["paths"]["output_dir"] = str(root / "model")
        config["paths"]["report_dir"] = str(root / "reports")
        config["training"] = dict(config["training"])
        config["training"]["device"] = "cuda"
        training_config.write_text(
            yaml.safe_dump(config),
            encoding="utf-8",
        )

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable in the Vertex container: "
                f"torch={torch.__version__}, cuda_build={torch.version.cuda}"
            )
        print(
            "Vertex manual-training GPU: "
            f"{torch.cuda.get_device_name(0)} "
            f"(torch={torch.__version__}, cuda_build={torch.version.cuda})",
            flush=True,
        )
        run_id = run_training(training_config)
        _upload_tree(root / "model", f"{output_prefix}/model")
        _upload_tree(root / "reports", f"{output_prefix}/reports")
        print(f"Manual Vertex MLflow run: {run_id}", flush=True)
        print(f"Manual Vertex artifacts: {output_prefix}", flush=True)


if __name__ == "__main__":
    main()
