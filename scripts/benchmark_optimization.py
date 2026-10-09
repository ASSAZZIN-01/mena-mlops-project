"""Compare the original CPU model with dynamic INT8 quantization."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from mena_mlops.optimization import benchmark_callable, quantize_linear_layers
from mena_mlops.serving.app import _load_predictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/arabert-debug")
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/optimization.json")
    )
    args = parser.parse_args()

    import os

    os.environ["MODEL_PATH"] = str(args.model_path)
    os.environ["MODEL_DEVICE"] = "cpu"
    predictor = _load_predictor()
    optimized_model = quantize_linear_layers(copy.deepcopy(predictor.model))
    encoded = predictor.tokenizer(
        "هذا المنتج ممتاز",
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    def original() -> torch.Tensor:
        with torch.inference_mode():
            return predictor.model(**encoded).logits

    def optimized() -> torch.Tensor:
        with torch.inference_mode():
            return optimized_model(**encoded).logits

    result = {
        "model_path": str(args.model_path),
        "original": benchmark_callable(original, iterations=args.iterations),
        "dynamic_int8": benchmark_callable(optimized, iterations=args.iterations),
    }
    result["speedup"] = (
        result["original"]["mean_ms"] / result["dynamic_int8"]["mean_ms"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
