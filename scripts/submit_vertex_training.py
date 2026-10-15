"""Submit a configurable manual training experiment to Vertex AI."""

from __future__ import annotations

import argparse
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import google.auth
from google.auth.transport.requests import AuthorizedSession
from google.cloud import storage

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
    print(f"Submitted Vertex manual training job: {job_name}", flush=True)
    job_endpoint = f"https://{region}-aiplatform.googleapis.com/v1/{job_name}"
    while True:
        job = session.get(job_endpoint, timeout=60)
        job.raise_for_status()
        state = str(job.json().get("state", ""))
        print(f"Vertex job state: {state}", flush=True)
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
    print(
        f"Submitting {job_id} with config {args.config} to "
        f"{os.getenv('VERTEX_ACCELERATOR_TYPE', 'NVIDIA_TESLA_T4')}",
        flush=True,
    )
    print(_submit_and_wait(job_spec, project, region), flush=True)


if __name__ == "__main__":
    main()
