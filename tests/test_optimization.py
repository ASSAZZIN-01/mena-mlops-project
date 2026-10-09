"""Tests for optimization utilities."""

import torch

from mena_mlops.optimization import benchmark_callable, quantize_linear_layers


def test_dynamic_quantization_and_benchmark() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
    optimized = quantize_linear_layers(model)
    assert optimized(torch.ones(1, 2)).shape == (1, 2)
    result = benchmark_callable(lambda: optimized(torch.ones(1, 2)), iterations=2)
    assert result["mean_ms"] >= 0
