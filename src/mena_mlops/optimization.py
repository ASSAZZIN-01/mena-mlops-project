"""CPU model optimization and repeatable latency measurements."""

from __future__ import annotations

import time
from collections.abc import Callable

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
