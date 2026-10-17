"""Vertex entry point for the shared-image distillation experiment."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import yaml
from google.cloud import storage

from mena_mlops.observability import emit_event
from mena_mlops.training.train import load_yaml
from distill_student import run


def _download(uri: str, destination: Path) -> None:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path:
        raise ValueError(f"expected a GCS URI, got {uri!r}")
    storage.Client().bucket(parsed.netloc).blob(
        parsed.path.lstrip("/")
    ).download_to_filename(destination)


def _download_tree(prefix: str, destination: Path) -> None:
    parsed = urlparse(prefix)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"expected a GCS prefix, got {prefix!r}")
    bucket = storage.Client().bucket(parsed.netloc)
    for blob in bucket.list_blobs(prefix=parsed.path.lstrip("/")):
        relative = Path(blob.name).relative_to(parsed.path.lstrip("/"))
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(target)


def _upload_tree(directory: Path, prefix: str) -> None:
    parsed = urlparse(prefix)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"expected a GCS prefix, got {prefix!r}")
    bucket = storage.Client().bucket(parsed.netloc)
    for path in directory.rglob("*"):
        if path.is_file():
            relative = path.relative_to(directory).as_posix()
            bucket.blob(
                f"{parsed.path.lstrip('/')}/{relative}"
            ).upload_from_filename(path)


def main() -> None:
    input_prefix = os.environ["VERTEX_INPUT_PREFIX"].rstrip("/")
    output_prefix = os.environ["VERTEX_OUTPUT_PREFIX"]
    with tempfile.TemporaryDirectory(prefix="vertex-distillation-") as temporary:
        root = Path(temporary)
        data_dir = root / "data"
        data_dir.mkdir()
        for split in ("train", "validation", "test"):
            _download(f"{input_prefix}/{split}.parquet", data_dir / f"{split}.parquet")
        teacher_dir = root / "teacher"
        _download_tree(f"{input_prefix}/teacher", teacher_dir)
        config_path = root / "distillation.yaml"
        _download(os.environ["VERTEX_DISTILLATION_CONFIG_URI"], config_path)
        config = load_yaml(config_path)
        config["teacher_model"] = str(teacher_dir)
        config["paths"] = dict(config["paths"])
        config["paths"]["data_dir"] = str(data_dir)
        config["paths"]["output_dir"] = str(root / "student")
        config["paths"]["report_dir"] = str(root / "reports")
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable in the Vertex container: "
                f"torch={torch.__version__}, cuda_build={torch.version.cuda}"
            )
        emit_event(
            "vertex_distillation_started",
            "distillation",
            status="running",
            accelerator=torch.cuda.get_device_name(0),
            torch_version=torch.__version__,
            cuda_build=torch.version.cuda,
        )
        run_id = run(config_path)
        _upload_tree(root / "student", f"{output_prefix}/student")
        _upload_tree(root / "reports", f"{output_prefix}/reports")
        lineage_uri = os.environ.get("VERTEX_LINEAGE_URI")
        if lineage_uri:
            lineage_path = root / "lineage.json"
            _download(lineage_uri, lineage_path)
            _upload_tree(root / "student", f"{output_prefix}/model-card-inputs")
            (root / "reports" / "lineage.json").write_text(
                json.dumps(json.loads(lineage_path.read_text()), indent=2) + "\n",
                encoding="utf-8",
            )
            _upload_tree(root / "reports", f"{output_prefix}/reports")
        emit_event(
            "vertex_distillation_completed",
            "distillation",
            status="succeeded",
            mlflow_run_id=run_id,
            output_prefix=output_prefix,
        )


if __name__ == "__main__":
    main()
