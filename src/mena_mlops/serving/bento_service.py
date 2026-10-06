"""BentoML service for the versioned Arabic sentiment model."""

from __future__ import annotations

import bentoml

from mena_mlops.serving.app import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    _load_predictor,
)


@bentoml.service(
    name="mena-arabert",
    traffic={"timeout": 30, "concurrency": 16},
)
class SentimentService:
    """Serve one immutable model artifact per BentoML process."""

    def __init__(self) -> None:
        self.predictor = _load_predictor()

    @bentoml.api
    def predict(self, payload: PredictionRequest) -> PredictionResponse:
        label, probabilities = self.predictor.predict(payload.text)
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
