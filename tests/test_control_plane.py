import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

import mena_mlops.control_plane as control_plane


def test_control_plane_health() -> None:
    response = TestClient(control_plane.app).get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_feedback_endpoint_exposes_annotation_summary(tmp_path: Path) -> None:
    feedback_path = tmp_path / "feedback.db"
    audit_path = tmp_path / "audit.db"
    with sqlite3.connect(audit_path) as connection:
        connection.execute("CREATE TABLE predictions (prediction TEXT NOT NULL)")
        connection.execute(
            "INSERT INTO predictions (prediction) VALUES (?)", ("neutral",)
        )
    control_plane.FEEDBACK_DB = feedback_path
    control_plane.AUDIT_DB = audit_path

    response = TestClient(control_plane.app).get("/api/feedback")

    assert response.status_code == 200
    assert response.json()["annotation"]["neutral"] == {
        "total_predictions": 1,
        "suspected": 0,
        "reviewed": 0,
        "confusing": 0,
        "pending": 0,
    }
