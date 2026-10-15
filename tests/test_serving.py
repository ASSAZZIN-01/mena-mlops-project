import json

import pytest
from fastapi.testclient import TestClient

from mena_mlops.serving.app import _select_onnx_providers, create_app


class FakePredictor:
    model_version = "test-v1"

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        assert text
        return "positive", {"negative": 0.1, "neutral": 0.2, "positive": 0.7}


def test_health_reports_loaded_model() -> None:
    client = TestClient(create_app(FakePredictor()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": "test-v1"}


def test_predict_returns_versioned_probabilities() -> None:
    client = TestClient(create_app(FakePredictor()))

    response = client.post("/predict", json={"text": "منتج ممتاز"})

    assert response.status_code == 200
    assert response.json() == {
        "label": "positive",
        "probabilities": {"negative": 0.1, "neutral": 0.2, "positive": 0.7},
        "model_version": "test-v1",
    }


def test_predict_emits_correlated_observability_event(capsys) -> None:
    client = TestClient(create_app(FakePredictor()))

    response = client.post(
        "/predict",
        headers={"x-request-id": "request-1", "x-source": "live"},
        json={"text": "منتج ممتاز"},
    )

    assert response.status_code == 200
    events = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line
    ]
    completed = next(
        event for event in events if event["event_type"] == "prediction_completed"
    )
    assert completed["request_id"] == "request-1"
    assert completed["model_version"] == "test-v1"
    assert completed["input_chars"] == len("منتج ممتاز")
    assert completed["confidence"] == 0.7
    assert completed["deployment_stage"] == "stable"
    assert "mena_gateway_predictions_total" in client.get("/metrics").text


def test_predict_rejects_empty_text() -> None:
    client = TestClient(create_app(FakePredictor()))

    response = client.post("/predict", json={"text": ""})

    assert response.status_code == 422


def test_onnx_provider_defaults_to_cpu() -> None:
    assert _select_onnx_providers({"CPUExecutionProvider"}, "cpu", None) == [
        "CPUExecutionProvider"
    ]


def test_onnx_provider_uses_cuda_when_configured() -> None:
    assert _select_onnx_providers(
        {"CPUExecutionProvider", "CUDAExecutionProvider"}, "cuda", None
    ) == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_onnx_provider_fails_when_unavailable() -> None:
    with pytest.raises(RuntimeError, match="none of the requested"):
        _select_onnx_providers(
            {"CPUExecutionProvider"}, "cuda", "CUDAExecutionProvider"
        )
