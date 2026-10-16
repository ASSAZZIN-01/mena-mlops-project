"""Tests for candidate promotion quality gates."""

from mena_mlops.deployment.quality_gate import evaluate_quality_gate
from mena_mlops.quality import compare_deployed_model


def _benchmark(macro_f1: float, neutral_f1: float, latency: float) -> dict[str, object]:
    return {
        "quality": {
            "macro_f1": macro_f1,
            "per_class": {
                "negative": {"f1": 0.70},
                "neutral": {"f1": neutral_f1},
                "positive": {"f1": 0.80},
            },
        },
        "latency": {
            "mean_ms": latency,
            "throughput_rps": 1000 / latency,
        },
    }


def test_deployed_model_gate_allows_three_percent_performance_regression() -> None:
    failures = compare_deployed_model(
        _benchmark(0.60, 0.30, 100.0),
        _benchmark(0.60, 0.30, 103.0),
    )

    assert failures == []


def test_deployed_model_gate_rejects_quality_and_performance_regression() -> None:
    failures = compare_deployed_model(
        _benchmark(0.60, 0.30, 100.0),
        _benchmark(0.59, 0.29, 104.0),
    )

    assert "candidate macro-F1 is below the deployed model" in failures
    assert "candidate neutral F1 is below the deployed model" in failures
    assert "candidate latency regresses by more than the allowed budget" in failures


def test_quality_gate_accepts_candidate_above_all_floors() -> None:
    accepted, reasons = evaluate_quality_gate(
        {"macro_f1": 0.60},
        {
            "macro_f1": 0.63,
            "per_class": {
                "negative": {"f1": 0.55},
                "neutral": {"f1": 0.35},
                "positive": {"f1": 0.56},
            },
        },
        minimum_macro_f1=0.60,
        minimum_improvement=0.01,
        minimum_class_f1={"negative": 0.50, "neutral": 0.30, "positive": 0.50},
    )

    assert accepted
    assert reasons == []


def test_quality_gate_reports_failed_conditions() -> None:
    accepted, reasons = evaluate_quality_gate(
        {"macro_f1": 0.60},
        {"macro_f1": 0.60, "per_class": {"neutral": {"f1": 0.0}}},
        minimum_macro_f1=0.60,
        minimum_improvement=0.01,
        minimum_class_f1={"neutral": 0.30},
    )

    assert not accepted
    assert len(reasons) == 2
