"""Submit a configurable manual training experiment to Vertex AI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import google.auth
from google.auth.transport.requests import AuthorizedSession
from google.cloud import storage

from mena_mlops.observability import emit_event

TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def _upload(path: Path, uri: str) -> None:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path:
        raise ValueError(f"expected a GCS URI, got {uri!r}")
    storage.Client().bucket(parsed.netloc).blob(
        parsed.path.lstrip("/")
    ).upload_from_filename(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _submit_and_wait(spec: dict[str, object], project: str, region: str) -> str:
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    session = AuthorizedSession(credentials)
    endpoint = (
        f"https://{region}-aiplatform.googleapis.com/v1/"
        f"projects/{project}/locations/{region}/customJobs"
    )
    response = session.post(endpoint, json=spec, timeout=60)
    response.raise_for_status()
    job_name = response.json()["name"]
    emit_event(
        "vertex_job_submitted",
        "manual_training",
        status="submitted",
        job_name=job_name,
    )
    job_endpoint = f"https://{region}-aiplatform.googleapis.com/v1/{job_name}"
    while True:
        job = session.get(job_endpoint, timeout=60)
        job.raise_for_status()
        state = str(job.json().get("state", ""))
        emit_event(
            "vertex_job_state",
            "manual_training",
            status=state.removeprefix("JOB_STATE_").lower(),
            job_name=job_name,
        )
        if state in TERMINAL_STATES:
            if state != "JOB_STATE_SUCCEEDED":
                raise RuntimeError(f"Vertex job finished unsuccessfully: {state}")
            return job_name
        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training.yaml"),
        help="Training configuration to run unchanged on Vertex.",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Data configuration recorded with the experiment.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--job-prefix",
        default="manual-training",
        help="Display-name prefix for the Vertex job.",
    )
    args = parser.parse_args()

    project = os.environ["VERTEX_PROJECT_ID"]
    region = os.getenv("VERTEX_REGION", "us-central1")
    image_uri = os.environ["VERTEX_TRAINING_IMAGE_URI"]
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    staging_bucket = os.environ["VERTEX_STAGING_BUCKET"].rstrip("/")
    job_id = datetime.now(UTC).strftime(f"{args.job_prefix}-%Y%m%d-%H%M%S")
    prefix = f"{staging_bucket}/{job_id}"

    _upload(args.config, f"{prefix}/training.yaml")
    _upload(args.data_config, f"{prefix}/data.yaml")
    lineage = {
        "git_commit": _git_commit(),
        "dvc_commit": _git_commit(),
        "dvc_lock_sha256": _sha256(Path("dvc.lock")),
        "dvc_yaml_sha256": _sha256(Path("dvc.yaml")),
        "training_config_sha256": _sha256(args.config),
        "data_config_sha256": _sha256(args.data_config),
        "training_image": image_uri,
        "reproduction_command": (
            f"python scripts/submit_vertex_training.py --config {args.config} "
            f"--data-config {args.data_config} --data-dir {args.data_dir}"
        ),
    }
    lineage_path = args.config.parent / ".vertex-lineage.json"
    lineage_path.write_text(json.dumps(lineage, indent=2) + "\n", encoding="utf-8")
    _upload(lineage_path, f"{prefix}/lineage.json")
    lineage_path.unlink()
    for split in ("train", "validation", "test"):
        _upload(
            args.data_dir / f"{split}.parquet",
            f"{prefix}/{split}.parquet",
        )

    env = [
        {"name": "VERTEX_INPUT_PREFIX", "value": prefix},
        {"name": "VERTEX_OUTPUT_PREFIX", "value": f"{prefix}/output"},
        {"name": "VERTEX_TRAINING_CONFIG_URI", "value": f"{prefix}/training.yaml"},
        {"name": "VERTEX_DATA_CONFIG_URI", "value": f"{prefix}/data.yaml"},
        {"name": "VERTEX_LINEAGE_URI", "value": f"{prefix}/lineage.json"},
        {"name": "VERTEX_MODEL_ID", "value": job_id},
        {"name": "VERTEX_TRAINING_JOB_ID", "value": job_id},
    ]
    for name, value in (
        ("MLFLOW_TRACKING_URI", tracking_uri),
        ("MLFLOW_ARTIFACT_LOCATION", os.getenv("MLFLOW_ARTIFACT_LOCATION")),
    ):
        if value:
            env.append({"name": name, "value": value})

    job_spec = {
        "displayName": job_id,
        "jobSpec": {
            "workerPoolSpecs": [
                {
                    "machineSpec": {
                        "machineType": os.getenv(
                            "VERTEX_MACHINE_TYPE",
                            "n1-standard-4",
                        ),
                        "acceleratorType": os.getenv(
                            "VERTEX_ACCELERATOR_TYPE",
                            "NVIDIA_TESLA_T4",
                        ),
                        "acceleratorCount": 1,
                    },
                    "replicaCount": 1,
                    "containerSpec": {
                        "imageUri": image_uri,
                        "command": ["python", "scripts/vertex_manual_train.py"],
                        "env": env,
                    },
                }
            ]
        },
    }
    emit_event(
        "vertex_job_submission_started",
        "manual_training",
        status="started",
        model_id=job_id,
        config=str(args.config),
        accelerator=os.getenv("VERTEX_ACCELERATOR_TYPE", "NVIDIA_TESLA_T4"),
    )
    emit_event(
        "vertex_job_completed",
        "manual_training",
        status="succeeded",
        job_name=_submit_and_wait(job_spec, project, region),
        model_id=job_id,
    )


if __name__ == "__main__":
    main()
