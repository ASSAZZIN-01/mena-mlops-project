"""BentoML service for the versioned Arabic sentiment model."""

from __future__ import annotations

from threading import Lock

import bentoml
from prometheus_client import Counter, Gauge, Histogram

from mena_mlops.serving.app import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    _load_predictor,
)

PREDICTIONS = Counter(
    "mena_model_predictions_total",
    "Number of predictions by argmax label.",
    ["label", "model_version"],
)
CONFIDENCE = Histogram(
    "mena_model_prediction_confidence",
    "Confidence of predictions grouped by their argmax label.",
    ["label", "model_version"],
    buckets=(0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0),
)
CONFIDENCE_MIN = Gauge(
    "mena_model_prediction_confidence_min",
    "Minimum observed probability for each class.",
    ["label", "model_version"],
)
CONFIDENCE_MAX = Gauge(
    "mena_model_prediction_confidence_max",
    "Maximum observed probability for each class.",
    ["label", "model_version"],
)
_confidence_extrema: dict[tuple[str, str], tuple[float, float]] = {}
_confidence_lock = Lock()


def record_prediction_metrics(
    label: str,
    probabilities: dict[str, float],
    model_version: str,
) -> None:
    """Record prediction mix and probability statistics for observability."""

    PREDICTIONS.labels(label, model_version).inc()
    confidence = probabilities[label]
    CONFIDENCE.labels(label, model_version).observe(confidence)
    key = (label, model_version)
    with _confidence_lock:
        current_min, current_max = _confidence_extrema.get(
            key, (confidence, confidence)
        )
        current_min = min(current_min, confidence)
        current_max = max(current_max, confidence)
        _confidence_extrema[key] = (current_min, current_max)
    CONFIDENCE_MIN.labels(label, model_version).set(current_min)
    CONFIDENCE_MAX.labels(label, model_version).set(current_max)


@bentoml.service(
    name="mena-arabert",
    traffic={"timeout": 30, "concurrency": 16},
)
class SentimentService:
    """Serve one immutable model artifact per BentoML process."""

    def __init__(self) -> None:
        self.predictor = _load_predictor()

    @bentoml.api
    def predict(self, text: str) -> PredictionResponse:
        payload = PredictionRequest(text=text)
        label, probabilities = self.predictor.predict(payload.text)
        record_prediction_metrics(label, probabilities, self.predictor.model_version)
        return PredictionResponse(
            label=label,
            probabilities=probabilities,
            model_version=self.predictor.model_version,
        )

    @bentoml.api
    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_version=self.predictor.model_version,
        )
