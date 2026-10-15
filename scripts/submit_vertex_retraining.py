"""Submit and monitor a Vertex AI Custom Job."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import google.auth
import yaml
from google.auth.transport.requests import AuthorizedSession
from google.cloud import storage

from mena_mlops.feedback import FeedbackStore
from mena_mlops.retraining import (
    drift_fingerprint,
    drift_triggered,
    feedback_thresholds_reached,
)

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
    (
        storage.Client()
        .bucket(parsed.netloc)
        .blob(parsed.path.lstrip("/"))
        .upload_from_filename(path)
    )


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
    if response.status_code == 429:
        raise RuntimeError(
            "Vertex AI rejected the Custom Job submission with HTTP 429: "
            f"{response.text[:1000]}"
        )
    response.raise_for_status()
    job_name = response.json()["name"]
    print(f"Submitted Vertex job: {job_name}", flush=True)
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
        "--database",
        type=Path,
        default=Path("data/feedback/reviewed.db"),
    )
    parser.add_argument(
        "--feedback-config",
        type=Path,
        default=Path("configs/feedback.yaml"),
    )
    parser.add_argument(
        "--drift-summary",
        type=Path,
        default=None,
        help="Evidently/PSI summary used as an independent retraining trigger.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--scheduled",
        action="store_true",
        help="Submit the scheduled full-data training job without trigger gating.",
    )
    parser.add_argument(
        "--job-prefix",
        default="feedback-retraining",
        help="Display-name prefix for the Vertex job.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.feedback_config.read_text(encoding="utf-8"))
    thresholds = {
        str(label): int(value)
        for label, value in config["retraining_thresholds"].items()
    }
    store = FeedbackStore(args.database)
    checkpoint = store.retraining_checkpoint()
    feedback_upper_bound = store.latest_reviewed_id()
    counts = store.counts_since(checkpoint.last_feedback_id)
    feedback_ready = feedback_thresholds_reached(counts, thresholds)
    drift_summary = (
        json.loads(args.drift_summary.read_text(encoding="utf-8"))
        if args.drift_summary is not None
        else None
    )
    current_drift_fingerprint = drift_fingerprint(drift_summary)
    drift_ready = (
        drift_triggered(drift_summary)
        and current_drift_fingerprint != checkpoint.last_drift_fingerprint
    )
    if not args.scheduled and not (feedback_ready or drift_ready):
        print(
            json.dumps(
                {
                    "status": "skipped",
                    "counts": counts,
                    "total_counts": store.counts(),
                    "thresholds": thresholds,
                    "feedback_triggered": feedback_ready,
                    "drift_triggered": drift_ready,
                }
            )
        )
        return

    project = os.environ["VERTEX_PROJECT_ID"]
    region = os.getenv("VERTEX_REGION", "us-central1")
    image_uri = os.environ["VERTEX_TRAINING_IMAGE_URI"]
    staging_bucket = os.environ["VERTEX_STAGING_BUCKET"].rstrip("/")
    job_id = datetime.now(UTC).strftime(
        f"{args.job_prefix}-%Y%m%d-%H%M%S"
    )
    prefix = f"{staging_bucket}/{job_id}"
    with tempfile_feedback(
        args.database,
        after_id=checkpoint.last_feedback_id,
        through_id=feedback_upper_bound,
    ) as feedback_path:
        for split in ("train", "validation", "test"):
            _upload(args.data_dir / f"{split}.parquet", f"{prefix}/{split}.parquet")
        _upload(feedback_path, f"{prefix}/feedback.parquet")

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
                        "command": ["python", "scripts/vertex_train.py"],
                        "env": [
                            {"name": "VERTEX_INPUT_PREFIX", "value": prefix},
                            {
                                "name": "MLFLOW_TRACKING_URI",
                                "value": os.environ["MLFLOW_TRACKING_URI"],
                            },
                            {
                                "name": "MLFLOW_ARTIFACT_LOCATION",
                                "value": os.getenv(
                                    "MLFLOW_ARTIFACT_LOCATION",
                                    "gs://mena-mlops-project-dvc/mlflow",
                                ),
                            },
                        ],
                    },
                }
            ],
        },
    }
    print(
        json.dumps(
            {
                "status": "submitted",
                "counts": counts,
                "job_id": job_id,
                "feedback_triggered": feedback_ready,
                "drift_triggered": drift_ready,
                "scheduled": args.scheduled,
            }
        )
    )
    _submit_and_wait(job_spec, project, region)
    store.mark_retraining_succeeded(
        feedback_upper_bound,
        current_drift_fingerprint
        if drift_ready
        else checkpoint.last_drift_fingerprint,
    )


class tempfile_feedback:
    def __init__(
        self,
        database: Path,
        after_id: int,
        through_id: int,
    ) -> None:
        self.database = database
        self.after_id = after_id
        self.through_id = through_id
        self.path: Path | None = None

    def __enter__(self) -> Path:
        import tempfile

        handle = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
        handle.close()
        self.path = Path(handle.name)
        FeedbackStore(self.database).export_reviewed(
            self.path,
            after_id=self.after_id,
            through_id=self.through_id,
        )
        return self.path

    def __exit__(self, *_: object) -> None:
        if self.path is not None:
            self.path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
