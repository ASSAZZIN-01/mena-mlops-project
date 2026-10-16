"""FastAPI control plane for the MENA MLOps operations dashboard."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mena_mlops.feedback import FeedbackStore
from mena_mlops.release_registry import ReleaseRegistry

PROJECT_ROOT = Path(os.getenv("MENA_PROJECT_ROOT", Path.cwd()))
REGISTRY_PATH = Path(
    os.getenv(
        "CONTROL_PLANE_DB",
        str(PROJECT_ROOT / "reports" / "control-plane" / "releases.db"),
    )
)
STATIC_DIR = Path(
    os.getenv(
        "CONTROL_PLANE_STATIC_DIR",
        str(PROJECT_ROOT / "web" / "control-center"),
    )
)
FEEDBACK_DB = Path(
    os.getenv("FEEDBACK_DB", str(PROJECT_ROOT / "data" / "feedback" / "reviewed.db"))
)
AUDIT_DB = Path(
    os.getenv("AUDIT_DB", str(PROJECT_ROOT / "data" / "audit" / "predictions.db"))
)
FEEDBACK_CONFIG = Path(
    os.getenv("FEEDBACK_CONFIG", str(PROJECT_ROOT / "configs" / "feedback.yaml"))
)
registry = ReleaseRegistry(REGISTRY_PATH)
app = FastAPI(title="MENA MLOps control plane", version="1.0.0")


class ReviewSubmission(BaseModel):
    item_id: int
    decision: str
    corrected_label: str | None = None

if STATIC_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")


def _feedback_summary() -> dict[str, object]:
    try:
        config = yaml.safe_load(FEEDBACK_CONFIG.read_text(encoding="utf-8"))
        thresholds = {
            str(label): int(value)
            for label, value in config["retraining_thresholds"].items()
        }
        counts = FeedbackStore(FEEDBACK_DB).counts()
        return {
            "counts": counts,
            "thresholds": thresholds,
            "ready": all(counts[label] >= thresholds[label] for label in thresholds),
            "annotation": FeedbackStore(FEEDBACK_DB).annotation_summary(AUDIT_DB),
        }
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        sqlite3.Error,
        yaml.YAMLError,
    ) as error:
        return {
            "error": str(error),
            "counts": {},
            "thresholds": {},
            "annotation": {},
            "ready": False,
        }


def _health(url: str) -> dict[str, str]:
    try:
        with urlopen(url, timeout=2) as response:
            return {"status": "healthy", "detail": str(response.status)}
    except (OSError, URLError) as error:
        return {"status": "unhealthy", "detail": str(error)}


@app.get("/api/overview")
def overview() -> dict[str, object]:
    return {
        **registry.overview(),
        "feedback": _feedback_summary(),
        "services": {
            "gateway": _health(
                os.getenv("MODEL_API_URL", "http://nginx:8080") + "/health"
            ),
            "prometheus": _health(
                os.getenv("PROMETHEUS_URL", "http://prometheus:9090") + "/-/ready"
            ),
            "mlflow": _health(
                os.getenv("MLFLOW_URL", "http://mlflow:5000") + "/health"
            ),
            "airflow": _health(
                os.getenv("AIRFLOW_INTERNAL_URL", "http://airflow:8080")
                + "/health"
            ),
        },
        "links": {
            "airflow": os.getenv("AIRFLOW_URL", "http://localhost:8081"),
            "grafana": os.getenv("GRAFANA_URL", "http://localhost:3000"),
            "mlflow": os.getenv("MLFLOW_PUBLIC_URL", "http://localhost:5000"),
            "feedback": os.getenv("FEEDBACK_URL", "http://localhost:8502"),
            "prometheus": os.getenv(
                "PROMETHEUS_PUBLIC_URL",
                "http://localhost:9090",
            ),
        },
    }


@app.get("/api/releases")
def releases() -> list[dict[str, object]]:
    return registry.list_releases()


@app.get("/api/releases/{release_id}")
def release(release_id: str) -> dict[str, object]:
    result = registry.get_release(release_id)
    if result is None:
        raise HTTPException(status_code=404, detail="release not found")
    return result


@app.get("/api/feedback")
def feedback() -> dict[str, object]:
    return _feedback_summary()


@app.get("/api/reviewer/next")
def reviewer_next() -> dict[str, object]:
    store = FeedbackStore(FEEDBACK_DB)
    item = store.next_pending()
    summary = _feedback_summary()
    if item is None:
        return {"item": None, "queue_position": 0, "queue_total": 0, "summary": summary}
    pending = summary.get("annotation", {})
    class_summary = pending.get(item.predicted_label, {})
    queue_total = sum(
        int(values.get("pending", 0))
        for values in pending.values()
        if isinstance(values, dict)
    )
    return {
        "item": {
            "id": item.id,
            "text": item.text,
            "predicted_label": item.predicted_label,
            "confidence": item.confidence,
        },
        "queue_position": 1,
        "queue_total": queue_total,
        "class_summary": class_summary,
        "summary": summary,
    }


@app.post("/api/reviewer/submit")
def reviewer_submit(submission: ReviewSubmission) -> dict[str, object]:
    store = FeedbackStore(FEEDBACK_DB)
    try:
        if submission.decision == "correct":
            store.review(submission.item_id, correct=True)
        elif submission.decision == "wrong":
            store.review(
                submission.item_id,
                correct=False,
                corrected_label=submission.corrected_label,
            )
        elif submission.decision == "defer":
            store.defer(submission.item_id)
        else:
            raise ValueError("decision must be correct, wrong, or defer")
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return reviewer_next()


@app.get("/api/deployments")
def deployments() -> dict[str, object]:
    releases = registry.list_releases()
    baseline_path = PROJECT_ROOT / "reports" / "retraining" / "deployed_baseline.json"
    baseline: dict[str, object] | None = None
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    return {
        "current": baseline,
        "history": [
            release
            for release in releases
            if release["deployment_status"] != "not_started"
        ],
    }


@app.get("/api/models")
def models() -> dict[str, object]:
    baseline_path = PROJECT_ROOT / "reports" / "retraining" / "deployed_baseline.json"
    if not baseline_path.is_file():
        return {"current": None, "candidates": registry.list_releases()}
    return {
        "current": json.loads(baseline_path.read_text(encoding="utf-8")),
        "candidates": registry.list_releases(),
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/reviewer")
def reviewer() -> FileResponse:
    return FileResponse(STATIC_DIR / "reviewer.html")


@app.get("/{path:path}")
def static_file(path: str) -> FileResponse:
    candidate = (STATIC_DIR / path).resolve()
    if STATIC_DIR.resolve() not in candidate.parents or not candidate.is_file():
        raise HTTPException(status_code=404, detail="page not found")
    return FileResponse(candidate)
