"""Evaluate, optimize, and gate a newly trained candidate model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from onnxruntime import InferenceSession
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from mena_mlops.deployment.quality_gate import evaluate_quality_gate
from mena_mlops.observability import emit_event
from mena_mlops.optimization import benchmark_callable, export_onnx_model, onnx_predict
from mena_mlops.quality import (
    check_optimization_quality,
    compare_deployed_model,
    select_optimization_variant,
)
from mena_mlops.training.evaluation import evaluate_predictions

LABEL_NAMES = ["negative", "neutral", "positive"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluate_predictions(
    predictions: list[int],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    labels = [LABEL_NAMES.index(str(label)) for label in frame["label"]]
    return evaluate_predictions(labels, predictions, LABEL_NAMES)


def _transformer_predictions(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    max_length: int,
) -> list[int]:
    predictions: list[int] = []
    model.eval()
    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        with torch.inference_mode():
            predictions.append(int(model(**encoded).logits.argmax(dim=-1)[0]))
    return predictions


def _onnx_predictions(
    session: InferenceSession,
    tokenizer: Any,
    texts: list[str],
    *,
    max_length: int,
) -> list[int]:
    return [
        int(
            np.argmax(
                onnx_predict(session, tokenizer, text, max_length=max_length)
            )
        )
        for text in texts
    ]


def _comparison(
    baseline: dict[str, Any],
    variant: dict[str, Any],
    *,
    maximum_macro_f1_drop_percent: float,
) -> dict[str, Any]:
    baseline_macro = float(baseline["quality"]["macro_f1"])
    variant_macro = float(variant["quality"]["macro_f1"])
    drop_percent = (baseline_macro - variant_macro) / baseline_macro * 100
    failures = check_optimization_quality(
        {
            "macro_f1": baseline_macro,
            "mean_latency_ms": baseline["latency"]["mean_ms"],
        },
        {
            "macro_f1": variant_macro,
            "mean_latency_ms": variant["latency"]["mean_ms"],
        },
        maximum_macro_f1_drop_percent=maximum_macro_f1_drop_percent,
    )
    return {
        "macro_f1_drop_percent": drop_percent,
        "speedup_percent": (
            1 - variant["latency"]["mean_ms"] / baseline["latency"]["mean_ms"]
        )
        * 100,
        "promotion_eligible": not failures,
        "quality_gate_failures": failures,
    }


def evaluate_and_gate_candidate(
    model_dir: Path,
    test_data: Path,
    output_dir: Path,
    report_path: Path,
    *,
    stable_macro_f1: float,
    minimum_macro_f1: float,
    minimum_improvement: float,
    minimum_class_f1: dict[str, float],
    maximum_macro_f1_drop_percent: float,
    deployed_baseline: dict[str, Any] | None = None,
    maximum_deployed_performance_regression_percent: float = 3.0,
    max_length: int = 128,
    iterations: int = 20,
) -> dict[str, Any]:
    """Create deployment variants and return a durable promotion decision."""

    model = AutoModelForSequenceClassification.from_pretrained(model_dir).cpu()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    frame = pd.read_parquet(test_data)
    texts = frame["text"].astype(str).tolist()

    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = output_dir / "model.onnx"
    int8_path = output_dir / "model-int8.onnx"
    export_onnx_model(model, tokenizer, fp32_path, max_length=max_length)
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    tokenizer.save_pretrained(output_dir)

    sessions = {
        "onnx-fp32": InferenceSession(
            str(fp32_path), providers=["CPUExecutionProvider"]
        ),
        "onnx-int8": InferenceSession(
            str(int8_path), providers=["CPUExecutionProvider"]
        ),
    }

    def pytorch_predict() -> object:
        _transformer_predictions(
            model, tokenizer, [texts[0]], max_length=max_length
        )
        return None

    def onnx_predict_one(session: InferenceSession) -> object:
        _onnx_predictions(session, tokenizer, [texts[0]], max_length=max_length)
        return None

    results: dict[str, Any] = {
        "pytorch": {
            "quality": _evaluate_predictions(
                _transformer_predictions(
                    model, tokenizer, texts, max_length=max_length
                ),
                frame,
            ),
            "latency": benchmark_callable(
                pytorch_predict,
                iterations=iterations,
            ),
        }
    }
    for name, session in sessions.items():
        results[name] = {
            "quality": _evaluate_predictions(
                _onnx_predictions(session, tokenizer, texts, max_length=max_length),
                frame,
            ),
            "latency": benchmark_callable(
                lambda session=session: onnx_predict_one(session),
                iterations=iterations,
            ),
        }

    baseline = results["pytorch"]
    for name in sessions:
        results[name]["comparison"] = _comparison(
            baseline,
            results[name],
            maximum_macro_f1_drop_percent=maximum_macro_f1_drop_percent,
        )
    candidates = {
        name: {
            "macro_f1": result["quality"]["macro_f1"],
            "mean_latency_ms": result["latency"]["mean_ms"],
            "p95_latency_ms": result["latency"]["p95_ms"],
        }
        for name, result in results.items()
        if name != "pytorch"
    }
    selected = select_optimization_variant(
        candidates,
        {
            "macro_f1": baseline["quality"]["macro_f1"],
            "mean_latency_ms": baseline["latency"]["mean_ms"],
        },
        maximum_macro_f1_drop_percent=maximum_macro_f1_drop_percent,
    )
    optimization_failures = (
        ["no optimized variant passed the optimization gate"]
        if selected is None
        else results[selected]["comparison"]["quality_gate_failures"]
    )
    selected_quality = (
        results[selected]["quality"] if selected is not None else baseline["quality"]
    )
    quality_accepted, quality_failures = evaluate_quality_gate(
        {"macro_f1": stable_macro_f1},
        selected_quality,
        minimum_macro_f1=minimum_macro_f1,
        minimum_improvement=minimum_improvement,
        minimum_class_f1=minimum_class_f1,
    )
    deployed_failures = (
        ["deployed model baseline is unavailable"]
        if deployed_baseline is None
        else compare_deployed_model(
            deployed_baseline,
            results[selected] if selected is not None else baseline,
            maximum_regression_percent=maximum_deployed_performance_regression_percent,
        )
    )
    stable_comparison: dict[str, Any] | None = None
    if deployed_baseline is not None and selected is not None:
        stable_quality = deployed_baseline.get("quality", {})
        candidate_result = results[selected]
        candidate_quality = candidate_result["quality"]
        stable_comparison = {
            "stable_model_id": deployed_baseline.get("model_id"),
            "stable": {
                "quality": stable_quality,
                "latency": deployed_baseline.get("latency", {}),
            },
            "candidate": {
                "variant": selected,
                "quality": candidate_quality,
                "latency": candidate_result["latency"],
            },
            "deltas": {
                "macro_f1": float(candidate_quality["macro_f1"])
                - float(stable_quality["macro_f1"]),
                "latency_mean_ms": float(candidate_result["latency"]["mean_ms"])
                - float(deployed_baseline["latency"]["mean_ms"]),
                "throughput_rps": float(
                    candidate_result["latency"]["throughput_rps"]
                )
                - float(deployed_baseline["latency"]["throughput_rps"]),
                "per_class_f1": {
                    label: float(candidate_quality["per_class"][label]["f1"])
                    - float(stable_quality["per_class"][label]["f1"])
                    for label in stable_quality.get("per_class", {})
                    if label in candidate_quality.get("per_class", {})
                },
            },
        }
    decision = {
        "status": (
            "approved"
            if (
                selected is not None
                and not optimization_failures
                and quality_accepted
                and not deployed_failures
            )
            else "rejected"
        ),
        "selected_variant": selected,
        "quality_gate": {
            "accepted": quality_accepted,
            "failures": quality_failures,
        },
        "optimization_gate": {
            "accepted": selected is not None and not optimization_failures,
            "failures": optimization_failures,
        },
        "deployed_model_gate": {
            "accepted": not deployed_failures,
            "failures": deployed_failures,
            "maximum_regression_percent": (
                maximum_deployed_performance_regression_percent
            ),
        },
        "results": results,
        "stable_macro_f1": stable_macro_f1,
        "stable_comparison": stable_comparison,
    }
    manifest = {
        "files": {
            name: {"path": name, "sha256": _sha256(output_dir / name)}
            for name in ("model.onnx", "model-int8.onnx")
        },
        "selected_variant": selected,
        "promotion_status": decision["status"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    emit_event(
        "promotion_completed",
        "promotion",
        status=decision["status"],
        selected_variant=selected,
        report_path=str(report_path),
        quality_gate=quality_accepted,
        optimization_gate=selected is not None and not optimization_failures,
    )
    return decision
