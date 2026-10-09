"""Model-quality regression checks."""

from __future__ import annotations


def check_quality(
    metrics: dict[str, object],
    *,
    baseline_macro_f1: float,
    minimum_neutral_f1: float,
) -> list[str]:
    """Return quality-gate failures for a training evaluation report."""

    test_metrics = metrics.get("test")
    if not isinstance(test_metrics, dict):
        raise ValueError("evaluation report must contain test metrics")
    failures: list[str] = []
    macro_f1 = float(test_metrics["macro_f1"])
    if macro_f1 < baseline_macro_f1:
        failures.append(
            f"test macro-F1 {macro_f1:.6f} is below baseline "
            f"{baseline_macro_f1:.6f}"
        )
    per_class = test_metrics.get("per_class")
    if not isinstance(per_class, dict):
        raise ValueError("test metrics must contain per_class metrics")
    neutral = per_class.get("neutral")
    neutral_f1 = float(neutral["f1"]) if isinstance(neutral, dict) else 0.0
    if neutral_f1 < minimum_neutral_f1:
        failures.append(
            f"test neutral F1 {neutral_f1:.6f} is below minimum "
            f"{minimum_neutral_f1:.6f}"
        )
    return failures
