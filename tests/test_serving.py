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
