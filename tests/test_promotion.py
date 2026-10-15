from mena_mlops.promotion import _comparison


def test_optimization_comparison_rejects_excessive_quality_drop() -> None:
    comparison = _comparison(
        {
            "quality": {"macro_f1": 0.80},
            "latency": {"mean_ms": 10.0},
        },
        {
            "quality": {"macro_f1": 0.70},
            "latency": {"mean_ms": 5.0},
        },
        maximum_macro_f1_drop_percent=5.0,
    )

    assert not comparison["promotion_eligible"]
    assert "macro-F1 drop 12.50% is not below 5.00%" in (
        comparison["quality_gate_failures"]
    )
