"""Tests for durable prediction audit records."""

import json

from mena_mlops.audit import PredictionAuditStore


def test_audit_record_preserves_prediction_provenance(tmp_path) -> None:
    store = PredictionAuditStore(tmp_path / "audit.db")

    request_id = store.record(
        text="خدمة متوسطة",
        prediction="positive",
        probabilities={"negative": 0.02, "neutral": 0.08, "positive": 0.9},
        model_version="onnx-int8-v1",
        source="batch",
        input_id="row-000001",
    )

    with store._connect() as connection:
        row = connection.execute(
            "SELECT request_id, input_id, source, text, prediction, confidence, "
            "probabilities, model_version, review_status FROM predictions"
        ).fetchone()

    assert row[0] == request_id
    assert row[1:6] == ("row-000001", "batch", "خدمة متوسطة", "positive", 0.9)
    assert json.loads(row[6])["positive"] == 0.9
    assert row[7:] == ("onnx-int8-v1", "not_required")


def test_audit_record_marks_uncertain_predictions_pending(tmp_path) -> None:
    store = PredictionAuditStore(tmp_path / "audit.db")

    store.record(
        text="خدمة عادية",
        prediction="negative",
        probabilities={"negative": 0.6, "neutral": 0.3, "positive": 0.1},
        model_version="onnx-int8-v1",
        source="live",
    )

    with store._connect() as connection:
        status = connection.execute(
            "SELECT review_status FROM predictions"
        ).fetchone()[0]

    assert status == "pending"


def test_audit_record_many_persists_one_batch(tmp_path) -> None:
    store = PredictionAuditStore(tmp_path / "audit.db")

    store.record_many(
        [
            {
                "request_id": "batch-1",
                "input_id": "row-1",
                "source": "batch",
                "text": "نص أول",
                "prediction": "positive",
                "probabilities": {
                    "negative": 0.1,
                    "neutral": 0.2,
                    "positive": 0.7,
                },
                "model_version": "onnx-int8-v1",
            },
            {
                "request_id": "batch-2",
                "input_id": "row-2",
                "source": "batch",
                "text": "نص ثان",
                "prediction": "negative",
                "probabilities": {
                    "negative": 0.9,
                    "neutral": 0.05,
                    "positive": 0.05,
                },
                "model_version": "onnx-int8-v1",
            },
        ]
    )

    with store._connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM predictions"
        ).fetchone()[0] == 2
