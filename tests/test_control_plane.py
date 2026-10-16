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


def test_reviewer_api_returns_and_submits_pending_item(tmp_path: Path) -> None:
    feedback_path = tmp_path / "feedback.db"
    audit_path = tmp_path / "audit.db"
    with sqlite3.connect(audit_path) as connection:
        connection.execute("CREATE TABLE predictions (prediction TEXT NOT NULL)")
        connection.execute(
            "INSERT INTO predictions (prediction) VALUES (?)", ("positive",)
        )
    store = control_plane.FeedbackStore(feedback_path)
    item_id = store.add_uncertain("نص للمراجعة", "positive", 0.42)
    control_plane.FEEDBACK_DB = feedback_path
    control_plane.AUDIT_DB = audit_path

    client = TestClient(control_plane.app)
    next_response = client.get("/api/reviewer/next")
    assert next_response.json()["item"]["id"] == item_id
    assert next_response.json()["queue_total"] == 1

    submit_response = client.post(
        "/api/reviewer/submit",
        json={"item_id": item_id, "decision": "correct"},
    )

    assert submit_response.status_code == 200
    assert submit_response.json()["item"] is None


def test_feedback_summary_reports_progress_since_last_retraining(
    tmp_path: Path,
) -> None:
    feedback_path = tmp_path / "feedback.db"
    audit_path = tmp_path / "audit.db"
    store = control_plane.FeedbackStore(feedback_path)
    old_item = store.add_uncertain("قديم", "negative", 0.4)
    store.review(old_item, correct=True)
    store.mark_retraining_succeeded(old_item, None)
    new_item = store.add_uncertain("جديد", "negative", 0.4)
    store.review(new_item, correct=True)
    control_plane.FEEDBACK_DB = feedback_path
    control_plane.AUDIT_DB = audit_path

    summary = control_plane._feedback_summary()

    assert summary["counts"] == {"negative": 1, "neutral": 0, "positive": 0}
    assert summary["lifetime_counts"] == {
        "negative": 2,
        "neutral": 0,
        "positive": 0,
    }
