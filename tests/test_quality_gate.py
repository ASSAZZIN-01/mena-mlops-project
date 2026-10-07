"""Tests for candidate promotion quality gates."""

from mena_mlops.deployment.quality_gate import evaluate_quality_gate


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
