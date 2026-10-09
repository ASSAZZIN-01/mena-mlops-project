"""FastAPI service for versioned Arabic sentiment predictions."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Predictor(Protocol):
    model_version: str

    def predict(self, text: str) -> tuple[str, dict[str, float]]: ...


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4_000)


class PredictionResponse(BaseModel):
    label: str
    probabilities: dict[str, float]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str


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


@dataclass
class TransformerPredictor:
    """Load one immutable model directory for the lifetime of the process."""

    model_path: Path
    model_version: str
    device: torch.device

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path
        ).to(self.device)
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
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(**encoded).logits[0], dim=-1)
        values = {
            self.labels[index]: round(float(probability), 6)
            for index, probability in enumerate(probabilities)
        }
        label = max(values, key=values.get)
        return label, values


@dataclass
class OnnxPredictor:
    """Run an ONNX deployment variant with the canonical tokenizer."""

    model_path: Path
    tokenizer_path: Path
    model_version: str

    def __post_init__(self) -> None:
        from onnxruntime import InferenceSession

        from mena_mlops.optimization import onnx_predict

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.session = InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
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
        probabilities = torch.softmax(torch.from_numpy(np.asarray(logits)), dim=-1)
        values = {
            self.labels[index]: round(float(probability), 6)
            for index, probability in enumerate(probabilities)
        }
        return max(values, key=values.get), values


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
    requested_device = os.getenv("MODEL_DEVICE", "auto")
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    return TransformerPredictor(
        model_path=model_path,
        model_version=os.getenv("MODEL_VERSION", model_path.name),
        device=torch.device(requested_device),
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
        try:
            label, probabilities = loaded.predict(payload.text)
            return PredictionResponse(
                label=label,
                probabilities=probabilities,
                model_version=loaded.model_version,
            )
        finally:
            elapsed = time.perf_counter() - started
            route = request.url.path
            REQUESTS.labels(route, "success", loaded.model_version).inc()
            LATENCY.labels(route, loaded.model_version).observe(elapsed)

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
