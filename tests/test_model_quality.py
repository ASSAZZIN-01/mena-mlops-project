"""Tests for the CI model-quality regression gate."""

from mena_mlops.quality import check_quality


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
