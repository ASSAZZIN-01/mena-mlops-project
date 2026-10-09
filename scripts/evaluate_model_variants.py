"""Compare quality and latency for PyTorch, ONNX FP32, and ONNX INT8."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from onnxruntime import InferenceSession

from mena_mlops.optimization import benchmark_callable, onnx_predict
from mena_mlops.quality import check_optimization_quality
from mena_mlops.serving.app import _load_predictor
from mena_mlops.training.evaluation import evaluate_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("models/arabert-debug"))
    parser.add_argument(
        "--variants-dir",
        type=Path,
        default=Path("models/arabert-debug-variants"),
    )
    parser.add_argument(
        "--test-data", type=Path, default=Path("data/processed/test.parquet")
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--max-macro-f1-drop-percent", type=float, default=5.0)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/optimization-comparison.json")
    )
    args = parser.parse_args()

    os.environ["MODEL_PATH"] = str(args.model_path)
    os.environ["MODEL_DEVICE"] = "cpu"
    predictor = _load_predictor()
    frame = pd.read_parquet(args.test_data)
    label_names = ["negative", "neutral", "positive"]
    label_ids = {label: index for index, label in enumerate(label_names)}
    texts = frame["text"].astype(str).tolist()
    labels = [label_ids[str(label)] for label in frame["label"]]

    sessions = {
        "onnx-fp32": InferenceSession(
            str(args.variants_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        ),
        "onnx-int8": InferenceSession(
            str(args.variants_dir / "model-int8.onnx"),
            providers=["CPUExecutionProvider"],
        ),
    }
    results: dict[str, object] = {}

    def pytorch_logits(text: str) -> np.ndarray:
        encoded = predictor.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        with torch.inference_mode():
            return predictor.model(**encoded).logits[0].numpy()

    predictors = {
        "pytorch": pytorch_logits,
        **{
            name: lambda text, session=session: onnx_predict(
                session, predictor.tokenizer, text
            )
            for name, session in sessions.items()
        },
    }
    for name, predict in predictors.items():
        predictions = [int(np.argmax(predict(text))) for text in texts]
        metrics = evaluate_predictions(
            predictions=predictions,
            labels=labels,
            label_names=label_names,
        )
        latency = benchmark_callable(
            lambda: predict(texts[0]),
            iterations=args.iterations,
        )
        results[name] = {"quality": metrics, "latency": latency}
    baseline = {
        "macro_f1": results["pytorch"]["quality"]["macro_f1"],
        "mean_latency_ms": results["pytorch"]["latency"]["mean_ms"],
    }
    for name, result in results.items():
        if name == "pytorch":
            continue
        variant = {
            "macro_f1": result["quality"]["macro_f1"],
            "mean_latency_ms": result["latency"]["mean_ms"],
        }
        drop_percent = (
            (baseline["macro_f1"] - variant["macro_f1"])
            / baseline["macro_f1"]
            * 100
        )
        failures = check_optimization_quality(
            baseline,
            variant,
            maximum_macro_f1_drop_percent=args.max_macro_f1_drop_percent,
        )
        result["comparison"] = {
            "macro_f1_drop_percent": drop_percent,
            "speedup_percent": (
                1 - variant["mean_latency_ms"] / baseline["mean_latency_ms"]
            )
            * 100,
            "promotion_eligible": not failures,
            "quality_gate_failures": failures,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
