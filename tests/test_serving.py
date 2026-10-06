from fastapi.testclient import TestClient

from mena_mlops.serving.app import create_app


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
