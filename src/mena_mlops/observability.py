"""Structured lifecycle events shared by orchestration and deployment tools."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


def emit_event(
    event_type: str,
    component: str,
    *,
    severity: str = "INFO",
    status: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Emit one JSON event and return the serialized event payload."""

    event: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event_id": str(uuid4()),
        "severity": severity,
        "event_type": event_type,
        "component": component,
    }
    if status is not None:
        event["status"] = status
    environment_fields = {
        "MENA_MODEL_ID": "model_id",
        "MENA_RELEASE_ID": "release_id",
        "VERTEX_TRAINING_JOB_ID": "vertex_training_job_id",
        "MLFLOW_RUN_ID": "mlflow_run_id",
        "MENA_TRIGGER": "trigger",
    }
    for name, field in environment_fields.items():
        value = os.getenv(name)
        if value:
            event[field] = value
    event.update(fields)
    print(json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True)
    return event


def emit_error(
    event_type: str,
    component: str,
    error: Exception,
    **fields: Any,
) -> dict[str, Any]:
    return emit_event(
        event_type,
        component,
        severity="ERROR",
        status="failed",
        error_type=type(error).__name__,
        error_message=str(error),
        **fields,
    )
