"""BentoML service for the versioned Arabic sentiment model."""

from __future__ import annotations

import os
import time
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import uuid4

import bentoml
from prometheus_client import Counter, Gauge, Histogram

from mena_mlops.audit import PredictionAuditStore
from mena_mlops.feedback import FeedbackStore
from mena_mlops.observability import emit_error, emit_event
from mena_mlops.serving.app import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    _load_predictor,
)

PREDICTIONS = Counter(
    "mena_model_predictions_total",
    "Number of predictions by argmax label.",
    ["label", "model_version", "source", "deployment_stage"],
)
CONFIDENCE = Histogram(
    "mena_model_prediction_confidence",
    "Confidence of predictions grouped by their argmax label.",
    ["label", "model_version", "source", "deployment_stage"],
    buckets=(0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0),
)
CONFIDENCE_MIN = Gauge(
    "mena_model_prediction_confidence_min",
    "Minimum observed probability for each class.",
    ["label", "model_version", "source", "deployment_stage"],
)
CONFIDENCE_MAX = Gauge(
    "mena_model_prediction_confidence_max",
    "Maximum observed probability for each class.",
    ["label", "model_version", "source", "deployment_stage"],
)
_confidence_extrema: dict[tuple[str, str], tuple[float, float]] = {}
_confidence_lock = Lock()


def record_prediction_metrics(
    label: str,
    probabilities: dict[str, float],
    model_version: str,
    source: str = "unknown",
    deployment_stage: str = "stable",
) -> None:
    """Record prediction mix and probability statistics for observability."""

    PREDICTIONS.labels(label, model_version, source, deployment_stage).inc()
    confidence = probabilities[label]
    CONFIDENCE.labels(
        label, model_version, source, deployment_stage
    ).observe(confidence)
    key = (label, model_version)
    with _confidence_lock:
        current_min, current_max = _confidence_extrema.get(
            key, (confidence, confidence)
        )
        current_min = min(current_min, confidence)
        current_max = max(current_max, confidence)
        _confidence_extrema[key] = (current_min, current_max)
    CONFIDENCE_MIN.labels(
        label, model_version, source, deployment_stage
    ).set(current_min)
    CONFIDENCE_MAX.labels(
        label, model_version, source, deployment_stage
    ).set(current_max)


@bentoml.service(
    name="mena-arabert",
    traffic={"timeout": 30, "concurrency": 16},
)
class SentimentService:
    """Serve one immutable model artifact per BentoML process."""

    def __init__(self) -> None:
        self.predictor = _load_predictor()
        self.audit_store = PredictionAuditStore(
            Path(os.getenv("AUDIT_DB", "data/audit/predictions.db"))
        )
        self.feedback_store = FeedbackStore(
            Path(os.getenv("FEEDBACK_DB", "data/feedback/reviewed.db"))
        )

    @bentoml.api
    def predict(
        self,
        text: str,
        source: str = "live",
        input_id: str | None = None,
    ) -> PredictionResponse:
        started = time.perf_counter()
        request_id = str(uuid4())
        deployment_stage = os.getenv("DEPLOYMENT_STAGE", "stable")
        payload = PredictionRequest(text=text)
        try:
            label, probabilities = self.predictor.predict(payload.text)
            record_prediction_metrics(
                label,
                probabilities,
                self.predictor.model_version,
                source,
                deployment_stage,
            )
            confidence = probabilities[label]
            self.audit_store.record(
                text=text,
                prediction=label,
                probabilities=probabilities,
                model_version=self.predictor.model_version,
                source=source,
                input_id=input_id,
                request_id=request_id,
            )
            if 0.25 <= confidence <= 0.75:
                self.feedback_store.add_uncertain(text, label, confidence)
            response = PredictionResponse(
                label=label,
                probabilities=probabilities,
                model_version=self.predictor.model_version,
            )
            emit_event(
                "prediction_completed",
                "bentoml",
                status="success",
                request_id=request_id,
                source=source,
                input_chars=len(text),
                label=label,
                confidence=confidence,
                model_version=self.predictor.model_version,
                deployment_stage=deployment_stage,
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
            )
            return response
        except Exception as error:
            emit_error(
                "prediction_failed",
                "bentoml",
                error,
                request_id=request_id,
                source=source,
                model_version=self.predictor.model_version,
                deployment_stage=deployment_stage,
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
            )
            raise

    @bentoml.api
    def predict_batch(
        self,
        texts: list[str],
        source: str = "batch",
        input_ids: list[str] | None = None,
        feedback_policy: Literal["audit_only", "reviewable"] = "audit_only",
    ) -> list[PredictionResponse]:
        """Run a bounded vectorized batch without polluting feedback by default."""

        if not texts:
            raise ValueError("texts must contain at least one item")
        if len(texts) > 128:
            raise ValueError("a batch cannot contain more than 128 texts")
        if input_ids is not None and len(input_ids) != len(texts):
            raise ValueError("input_ids must have the same length as texts")
        if source not in {"live", "batch", "stream"}:
            raise ValueError(f"unsupported source: {source}")

        started = time.perf_counter()
        request_id = str(uuid4())
        deployment_stage = os.getenv("DEPLOYMENT_STAGE", "stable")
        predictions = self.predictor.predict_batch(texts)
        responses: list[PredictionResponse] = []
        audit_records: list[dict[str, object]] = []
        for index, (label, probabilities) in enumerate(predictions):
            record_prediction_metrics(
                label,
                probabilities,
                self.predictor.model_version,
                source,
                deployment_stage,
            )
            confidence = probabilities[label]
            audit_records.append(
                {
                    "text": texts[index],
                    "prediction": label,
                    "probabilities": probabilities,
                    "model_version": self.predictor.model_version,
                    "source": source,
                    "input_id": (
                        input_ids[index] if input_ids is not None else None
                    ),
                    "request_id": f"{request_id}-{index}",
                }
            )
            if (
                feedback_policy == "reviewable"
                and 0.25 <= confidence <= 0.75
            ):
                self.feedback_store.add_uncertain(texts[index], label, confidence)
            responses.append(
                PredictionResponse(
                    label=label,
                    probabilities=probabilities,
                    model_version=self.predictor.model_version,
                )
            )
        self.audit_store.record_many(audit_records)
        emit_event(
            "batch_prediction_completed",
            "bentoml",
            status="success",
            request_id=request_id,
            source=source,
            batch_size=len(texts),
            model_version=self.predictor.model_version,
            deployment_stage=deployment_stage,
            feedback_policy=feedback_policy,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        return responses

    @bentoml.api
    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_version=self.predictor.model_version,
        )
