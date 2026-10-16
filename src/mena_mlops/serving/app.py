"""FastAPI service for versioned Arabic sentiment predictions."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from mena_mlops.observability import emit_error, emit_event


class Predictor(Protocol):
    model_version: str

    def predict(self, text: str) -> tuple[str, dict[str, float]]: ...

    def predict_batch(
        self, texts: list[str]
    ) -> list[tuple[str, dict[str, float]]]: ...


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4_000)


class PredictionResponse(BaseModel):
    label: str
    probabilities: dict[str, float]
    model_version: str


class BatchPredictionRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=128)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_version: str


def _select_onnx_providers(
    available: set[str],
) -> list[str]:
    if "CPUExecutionProvider" not in available:
        raise RuntimeError(
            "CPUExecutionProvider is not available in ONNX Runtime: "
            f"available={sorted(available)}"
        )
    return ["CPUExecutionProvider"]


REQUESTS = Counter(
    "mena_model_requests_total",
    "Prediction requests handled by the model service.",
    ["route", "status", "model_version"],
)
LATENCY = Histogram(
    "mena_model_request_latency_seconds",
    "Prediction request latency in seconds.",
    ["route", "model_version"],
)
PREDICTIONS = Counter(
    "mena_gateway_predictions_total",
    "Prediction outcomes emitted by the gateway.",
    ["label", "model_version", "deployment_stage"],
)
CONFIDENCE = Histogram(
    "mena_gateway_prediction_confidence",
    "Prediction confidence emitted by the gateway.",
    ["label", "model_version", "deployment_stage"],
    buckets=(0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0),
)


@dataclass
class TransformerPredictor:
    """Load one immutable model directory for the lifetime of the process."""

    model_path: Path
    model_version: str
    device: str

    def __post_init__(self) -> None:
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path
        ).to(torch.device(self.device))
        self.device = torch.device(self.device)
        self.model.eval()
        raw_labels = self.model.config.id2label
        self.labels = {
            int(index): str(label).lower()
            for index, label in raw_labels.items()
        }

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=int(os.getenv("MODEL_MAX_LENGTH", "128")),
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        import torch

        with torch.inference_mode():
            probabilities = torch.softmax(self.model(**encoded).logits[0], dim=-1)
        values = {
            self.labels[index]: round(float(probability), 6)
            for index, probability in enumerate(probabilities)
        }
        label = max(values, key=values.get)
        return label, values

    def predict_batch(
        self, texts: list[str]
    ) -> list[tuple[str, dict[str, float]]]:
        import torch

        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(os.getenv("MODEL_MAX_LENGTH", "128")),
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(**encoded).logits, dim=-1)
        results = []
        for row in probabilities:
            values = {
                self.labels[index]: round(float(probability), 6)
                for index, probability in enumerate(row)
            }
            results.append((max(values, key=values.get), values))
        return results


@dataclass
class OnnxPredictor:
    """Run an ONNX deployment variant with the canonical tokenizer."""

    model_path: Path
    tokenizer_path: Path
    model_version: str

    def __post_init__(self) -> None:
        import onnxruntime

        from mena_mlops.optimization import onnx_predict

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        selected = _select_onnx_providers(
            set(onnxruntime.get_available_providers()),
        )
        self.session = onnxruntime.InferenceSession(
            str(self.model_path), providers=selected
        )
        self.providers = tuple(self.session.get_providers())
        self._onnx_predict = onnx_predict
        raw_config = self.tokenizer.init_kwargs.get("model_max_length", 128)
        self.max_length = min(int(raw_config), 128)
        self.labels = {0: "negative", 1: "neutral", 2: "positive"}

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        logits = self._onnx_predict(
            self.session,
            self.tokenizer,
            text,
            max_length=self.max_length,
        )
        values_array = np.asarray(logits, dtype=np.float64).reshape(-1)
        shifted = values_array - np.max(values_array)
        probabilities = np.exp(shifted)
        probabilities /= np.sum(probabilities)
        values = {
            self.labels[index]: round(float(probability), 6)
            for index, probability in enumerate(probabilities.tolist())
        }
        return max(values, key=values.get), values

    def predict_batch(
        self, texts: list[str]
    ) -> list[tuple[str, dict[str, float]]]:
        from mena_mlops.optimization import onnx_predict_batch

        logits_batch = onnx_predict_batch(
            self.session,
            self.tokenizer,
            texts,
            max_length=self.max_length,
        )
        results = []
        for logits in logits_batch:
            values_array = np.asarray(logits, dtype=np.float64).reshape(-1)
            shifted = values_array - np.max(values_array)
            probabilities = np.exp(shifted)
            probabilities /= np.sum(probabilities)
            values = {
                self.labels[index]: round(float(probability), 6)
                for index, probability in enumerate(probabilities.tolist())
            }
            results.append((max(values, key=values.get), values))
        return results


def _load_predictor() -> Predictor:
    if os.getenv("MODEL_BACKEND", "pytorch").lower() == "onnx":
        model_path = Path(os.environ["MODEL_ONNX_PATH"])
        tokenizer_path = Path(
            os.getenv("MODEL_TOKENIZER_PATH", str(model_path.parent))
        )
        return OnnxPredictor(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            model_version=os.getenv("MODEL_VERSION", model_path.stem),
        )
    model_path = Path(os.getenv("MODEL_PATH", "models/arabert-debug"))
    if not model_path.is_dir():
        raise RuntimeError(
            f"MODEL_PATH does not point to a model directory: {model_path}"
        )
    return TransformerPredictor(
        model_path=model_path,
        model_version=os.getenv("MODEL_VERSION", model_path.name),
        device="cpu",
    )


def create_app(predictor: Predictor | None = None) -> FastAPI:
    """Create an app, allowing tests to inject a deterministic predictor."""

    app = FastAPI(title="MENA sentiment model service", version="0.1.0")
    app.state.predictor = predictor

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        loaded = app.state.predictor
        if loaded is None:
            raise HTTPException(status_code=503, detail="model is not loaded")
        return HealthResponse(status="ok", model_version=loaded.model_version)

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
        loaded = app.state.predictor
        if loaded is None:
            raise HTTPException(status_code=503, detail="model is not loaded")
        started = time.perf_counter()
        request_id = request.headers.get("x-request-id") or str(uuid4())
        deployment_stage = os.getenv("DEPLOYMENT_STAGE", "stable")
        status = "success"
        try:
            label, probabilities = loaded.predict(payload.text)
            confidence = probabilities[label]
            PREDICTIONS.labels(
                label, loaded.model_version, deployment_stage
            ).inc()
            CONFIDENCE.labels(
                label, loaded.model_version, deployment_stage
            ).observe(confidence)
            emit_event(
                "prediction_completed",
                "gateway",
                status="success",
                request_id=request_id,
                source=request.headers.get("x-source", "live"),
                input_chars=len(payload.text),
                label=label,
                confidence=confidence,
                model_version=loaded.model_version,
                deployment_stage=deployment_stage,
            )
            return PredictionResponse(
                label=label,
                probabilities=probabilities,
                model_version=loaded.model_version,
            )
        except Exception as error:
            status = "error"
            emit_error(
                "prediction_failed",
                "gateway",
                error,
                request_id=request_id,
                model_version=loaded.model_version,
                deployment_stage=deployment_stage,
            )
            raise
        finally:
            elapsed = time.perf_counter() - started
            route = request.url.path
            REQUESTS.labels(route, status, loaded.model_version).inc()
            LATENCY.labels(route, loaded.model_version).observe(elapsed)
            if status == "success":
                emit_event(
                    "prediction_latency_recorded",
                    "gateway",
                    status="success",
                    request_id=request_id,
                    latency_ms=round(elapsed * 1000, 3),
                    model_version=loaded.model_version,
                    deployment_stage=deployment_stage,
                )

    @app.post("/predict_batch", response_model=BatchPredictionResponse)
    def predict_batch(
        payload: BatchPredictionRequest, request: Request
    ) -> BatchPredictionResponse:
        loaded = app.state.predictor
        if loaded is None:
            raise HTTPException(status_code=503, detail="model is not loaded")
        started = time.perf_counter()
        source = request.headers.get("x-source", "batch")
        try:
            predictions = loaded.predict_batch(payload.texts)
            return BatchPredictionResponse(
                predictions=[
                    PredictionResponse(
                        label=label,
                        probabilities=probabilities,
                        model_version=loaded.model_version,
                    )
                    for label, probabilities in predictions
                ]
            )
        except Exception as error:
            emit_error(
                "batch_prediction_failed",
                "gateway",
                error,
                source=source,
                model_version=loaded.model_version,
                batch_size=len(payload.texts),
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
            )
            raise

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> PlainTextResponse:
        return PlainTextResponse(
            generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    return app


app = create_app()


def load_model_on_startup() -> None:
    """Load the configured model explicitly from the server entry point."""

    if app.state.predictor is None:
        app.state.predictor = _load_predictor()
