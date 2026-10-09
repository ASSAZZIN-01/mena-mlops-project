"""CPU model optimization and repeatable latency measurements."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch


def quantize_linear_layers(model: torch.nn.Module) -> torch.nn.Module:
    """Return a dynamically quantized CPU copy of a Transformer model."""

    if next(model.parameters()).device.type != "cpu":
        raise ValueError("dynamic quantization requires a CPU model")
    return torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )


def benchmark_callable(
    predict: Callable[[], object],
    *,
    iterations: int = 20,
    warmup: int = 5,
) -> dict[str, float]:
    """Measure average and p95 wall-clock latency in milliseconds."""

    if iterations < 1 or warmup < 0:
        raise ValueError("iterations must be positive and warmup cannot be negative")
    for _ in range(warmup):
        predict()
    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        predict()
        samples.append((time.perf_counter() - started) * 1000)
    samples.sort()
    return {
        "iterations": float(iterations),
        "mean_ms": sum(samples) / len(samples),
        "p95_ms": samples[min(len(samples) - 1, int(len(samples) * 0.95))],
    }


def export_onnx_model(
    model: torch.nn.Module,
    tokenizer: Any,
    output_path: Path,
    *,
    max_length: int = 128,
) -> None:
    """Export a sequence-classification model with dynamic batch axes."""

    encoded = tokenizer(
        "هذا المنتج ممتاز",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    model.eval().cpu()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    class LogitsWrapper(torch.nn.Module):
        def __init__(self, wrapped: torch.nn.Module) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
            return self.wrapped(*inputs).logits

    with torch.no_grad():
        torch.onnx.export(
            LogitsWrapper(model),
            tuple(encoded.values()),
            output_path,
            input_names=list(encoded.keys()),
            output_names=["logits"],
            dynamic_axes={
                key: {0: "batch", 1: "sequence"}
                for key in encoded
            }
            | {"logits": {0: "batch"}},
            opset_version=17,
        )


def onnx_predict(
    session: Any,
    tokenizer: Any,
    text: str,
    *,
    max_length: int = 128,
) -> np.ndarray:
    """Run one ONNX inference and return logits."""

    encoded = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=max_length,
    )
    inputs = {
        input_spec.name: encoded[input_spec.name].astype(np.int64)
        for input_spec in session.get_inputs()
        if input_spec.name in encoded
    }
    return np.asarray(session.run(["logits"], inputs)[0][0])
