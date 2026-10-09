"""Tests for the CI model-quality regression gate."""

from mena_mlops.quality import (
    check_optimization_quality,
    check_quality,
    select_optimization_variant,
)


def test_baseline_metrics_pass() -> None:
    failures = check_quality(
        {
            "test": {
                "macro_f1": 0.5522739620462365,
                "per_class": {"neutral": {"f1": 0.0}},
            }
        },
        baseline_macro_f1=0.5522739620462365,
        minimum_neutral_f1=0.0,
    )
    assert failures == []


def test_macro_f1_regression_fails() -> None:
    failures = check_quality(
        {
            "test": {
                "macro_f1": 0.4,
                "per_class": {"neutral": {"f1": 0.0}},
            }
        },
        baseline_macro_f1=0.5522739620462365,
        minimum_neutral_f1=0.0,
    )
    assert len(failures) == 1


def test_optimization_allows_small_quality_drop_with_speedup() -> None:
    failures = check_optimization_quality(
        {"macro_f1": 0.5523, "mean_latency_ms": 100.0},
        {"macro_f1": 0.5462, "mean_latency_ms": 17.0},
    )
    assert failures == []


def test_optimization_rejects_drop_at_threshold() -> None:
    failures = check_optimization_quality(
        {"macro_f1": 0.55, "mean_latency_ms": 100.0},
        {"macro_f1": 0.5225, "mean_latency_ms": 17.0},
    )
    assert failures


def test_optimization_selects_fastest_quality_eligible_variant() -> None:
    selected = select_optimization_variant(
        {
            "fast-but-bad": {
                "macro_f1": 0.5,
                "mean_latency_ms": 10.0,
                "p95_latency_ms": 20.0,
            },
            "int8": {
                "macro_f1": 0.5462,
                "mean_latency_ms": 17.0,
                "p95_latency_ms": 24.0,
            },
            "fp32": {
                "macro_f1": 0.5523,
                "mean_latency_ms": 65.0,
                "p95_latency_ms": 100.0,
            },
        },
        {"macro_f1": 0.5523, "mean_latency_ms": 100.0},
    )
    assert selected == "int8"
