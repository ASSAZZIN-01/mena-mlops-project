"""Tests for optimization utilities."""

import numpy as np
import torch

from mena_mlops.optimization import (
    benchmark_callable,
    onnx_predict,
    quantize_linear_layers,
)


def test_dynamic_quantization_and_benchmark() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
    optimized = quantize_linear_layers(model)
    assert optimized(torch.ones(1, 2)).shape == (1, 2)
    result = benchmark_callable(lambda: optimized(torch.ones(1, 2)), iterations=2)
    assert result["mean_ms"] >= 0


class _Input:
    def __init__(self, name: str) -> None:
        self.name = name


class _Tokenizer:
    def __call__(self, text: str, **_: object) -> dict[str, np.ndarray]:
        del text
        return {
            "input_ids": np.array([[1, 2]], dtype=np.int64),
            "token_type_ids": np.array([[0, 0]], dtype=np.int64),
            "attention_mask": np.array([[1, 1]], dtype=np.int64),
        }


class _Session:
    def get_inputs(self) -> list[_Input]:
        return [
            _Input("input_ids"),
            _Input("token_type_ids"),
            _Input("attention_mask"),
        ]

    def run(
        self,
        outputs: list[str],
        inputs: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        assert outputs == ["logits"]
        assert set(inputs) == {"input_ids", "token_type_ids", "attention_mask"}
        return [np.array([[0.1, 0.2, 0.7]], dtype=np.float32)]


def test_onnx_predict_maps_runtime_input_names() -> None:
    logits = onnx_predict(_Session(), _Tokenizer(), "نص")
    assert np.argmax(logits) == 2
