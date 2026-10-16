"""Submit and monitor a Vertex AI Custom Job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import google.auth
import yaml
from google.auth.transport.requests import AuthorizedSession
from google.cloud import storage

from mena_mlops.feedback import FeedbackStore
from mena_mlops.observability import emit_event
from mena_mlops.release_registry import ReleaseRegistry
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


def _write_release_decision_uri(output_path: Path, release_prefix: str) -> str:
    """Persist the exact release decision consumed by a deployment task."""

    decision_uri = f"{release_prefix.rstrip('/')}/promotion_decision.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(decision_uri + "\n", encoding="utf-8")
    return decision_uri


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_root() -> Path:
    return Path(
        os.environ.get(
            "MENA_HOST_PROJECT_ROOT",
            os.environ.get("MENA_AUTO_DEPLOY_PROJECT_ROOT", str(Path.cwd())),
        )
    )


def _git_commit() -> str:
    repository_root = _repository_root()
    return subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={repository_root}",
            "-C",
            str(repository_root),
            "rev-parse",
            "HEAD",
        ],
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
    if response.status_code == 429:
        raise RuntimeError(
            "Vertex AI rejected the Custom Job submission with HTTP 429: "
            f"{response.text[:1000]}"
        )
    response.raise_for_status()
    job_name = response.json()["name"]
    emit_event(
        "vertex_job_submitted",
        "automatic_retraining",
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
            "automatic_retraining",
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
        "--manual",
        action="store_true",
        help="Submit an on-demand full-data Vertex job without trigger gating.",
    )
    parser.add_argument(
        "--job-prefix",
        default="feedback-retraining",
        help="Display-name prefix for the Vertex job.",
    )
    parser.add_argument(
        "--release-decision-output",
        type=Path,
        default=None,
        help=(
            "Write the approved/rejected release decision URI here after a "
            "successful Vertex job. Used by orchestrators for deployment."
        ),
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.feedback_config.read_text(encoding="utf-8"))
    thresholds = {
        str(label): int(value)
        for label, value in config["retraining_thresholds"].items()
    }
    store = FeedbackStore(args.database)
    attempt_checkpoint = store.retraining_checkpoint()
    deployment_checkpoint = store.successful_deployment_checkpoint()
    feedback_upper_bound = store.latest_reviewed_id()
    feedback_time_upper_bound = store.latest_reviewed_at()
    counts = store.counts_since(
        attempt_checkpoint.last_feedback_id,
        reviewed_after=attempt_checkpoint.last_reviewed_at,
    )
    feedback_ready = feedback_thresholds_reached(counts, thresholds)
    drift_summary = (
        json.loads(args.drift_summary.read_text(encoding="utf-8"))
        if args.drift_summary is not None
        else None
    )
    current_drift_fingerprint = drift_fingerprint(drift_summary)
    drift_ready = (
        drift_triggered(drift_summary)
        and current_drift_fingerprint != attempt_checkpoint.last_drift_fingerprint
    )
    if not args.manual and not (feedback_ready or drift_ready):
        if args.release_decision_output is not None:
            args.release_decision_output.unlink(missing_ok=True)
        emit_event(
            "retraining_skipped",
            "automatic_retraining",
            status="skipped",
            counts=counts,
            total_counts=store.counts(),
            thresholds=thresholds,
            feedback_triggered=feedback_ready,
            drift_triggered=drift_ready,
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
    registry = ReleaseRegistry(
        Path(
            os.getenv(
                "CONTROL_PLANE_DB",
                "reports/control-plane/releases.db",
            )
        )
    )
    registry.start_release(
        job_id,
        trigger="manual" if args.manual else "automatic",
        vertex_job_id=job_id,
        payload={
            "feedback_triggered": feedback_ready,
            "drift_triggered": drift_ready,
        },
    )
    with tempfile_feedback(
        args.database,
        after_id=deployment_checkpoint.last_feedback_id,
        through_id=feedback_upper_bound,
        reviewed_after=deployment_checkpoint.last_reviewed_at,
        reviewed_through=feedback_time_upper_bound,
    ) as feedback_path:
        for split in ("train", "validation", "test"):
            _upload(args.data_dir / f"{split}.parquet", f"{prefix}/{split}.parquet")
        _upload(feedback_path, f"{prefix}/feedback.parquet")
    deployed_baseline_path = Path(
        os.getenv(
            "DEPLOYED_MODEL_BASELINE_PATH",
            "reports/retraining/deployed_baseline.json",
        )
    )
    deployed_baseline_uri: str | None = None
    if deployed_baseline_path.is_file():
        deployed_baseline_uri = f"{prefix}/deployed_baseline.json"
        _upload(deployed_baseline_path, deployed_baseline_uri)
    lineage_fd, lineage_name = tempfile.mkstemp(prefix="vertex-retraining-lineage-")
    os.close(lineage_fd)
    lineage_path = Path(lineage_name)
    lineage_path.write_text(
        json.dumps(
            {
                "git_commit": _git_commit(),
                "dvc_commit": _git_commit(),
                "dvc_lock_sha256": _sha256(_repository_root() / "dvc.lock"),
                "dvc_yaml_sha256": _sha256(_repository_root() / "dvc.yaml"),
                "training_config_sha256": _sha256(
                    _repository_root() / "configs/training.yaml"
                ),
                "data_config_sha256": _sha256(
                    _repository_root() / "configs/data.yaml"
                ),
                "training_image": image_uri,
                "reproduction_command": (
                    "python scripts/submit_vertex_retraining.py --manual "
                    f"--data-dir {args.data_dir}"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _upload(lineage_path, f"{prefix}/lineage.json")
    lineage_path.unlink()

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
                                "name": "VERTEX_LINEAGE_URI",
                                "value": f"{prefix}/lineage.json",
                            },
                            {"name": "VERTEX_MODEL_ID", "value": job_id},
                            {"name": "VERTEX_TRAINING_JOB_ID", "value": job_id},
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
    if deployed_baseline_uri is not None:
        job_spec["jobSpec"]["workerPoolSpecs"][0]["containerSpec"]["env"].append(
            {
                "name": "VERTEX_DEPLOYED_BASELINE_URI",
                "value": deployed_baseline_uri,
            }
        )
    emit_event(
        "retraining_submission_started",
        "automatic_retraining",
        status="started",
        counts=counts,
        model_id=job_id,
        feedback_triggered=feedback_ready,
        drift_triggered=drift_ready,
        manual=args.manual,
    )
    try:
        _submit_and_wait(job_spec, project, region)
    except Exception as error:
        registry.record_deployment(
            job_id,
            status="training_failed",
            payload={"error": str(error)},
        )
        raise
    decision_uri = f"{prefix}/release/promotion_decision.json"
    if args.release_decision_output is not None:
        _write_release_decision_uri(
            args.release_decision_output,
            f"{prefix}/release",
        )
    parsed = urlparse(decision_uri)
    decision = json.loads(
        storage.Client()
        .bucket(parsed.netloc)
        .blob(parsed.path.lstrip("/"))
        .download_as_text()
    )
    registry.record_decision(
        job_id,
        decision_uri=decision_uri,
        decision=decision,
        payload={"vertex_job_id": job_id},
    )
    store.mark_retraining_attempt(
        feedback_upper_bound,
        current_drift_fingerprint
        if drift_ready
        else attempt_checkpoint.last_drift_fingerprint,
        reviewed_at=feedback_time_upper_bound,
    )


class tempfile_feedback:
    def __init__(
        self,
        database: Path,
        after_id: int,
        through_id: int,
        reviewed_after: str | None = None,
        reviewed_through: str | None = None,
    ) -> None:
        self.database = database
        self.after_id = after_id
        self.through_id = through_id
        self.reviewed_after = reviewed_after
        self.reviewed_through = reviewed_through
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
            reviewed_after=self.reviewed_after,
            reviewed_through=self.reviewed_through,
        )
        return self.path

    def __exit__(self, *_: object) -> None:
        if self.path is not None:
            self.path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
