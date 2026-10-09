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


def check_optimization_quality(
    baseline: dict[str, float],
    variant: dict[str, float],
    *,
    maximum_macro_f1_drop_percent: float = 5.0,
) -> list[str]:
    """Gate an optimization variant against quality and latency baselines."""

    baseline_macro_f1 = float(baseline["macro_f1"])
    variant_macro_f1 = float(variant["macro_f1"])
    baseline_latency = float(baseline["mean_latency_ms"])
    variant_latency = float(variant["mean_latency_ms"])
    if baseline_macro_f1 <= 0:
        raise ValueError("baseline macro-F1 must be positive")
    drop_percent = (
        (baseline_macro_f1 - variant_macro_f1) / baseline_macro_f1 * 100
    )
    failures: list[str] = []
    if drop_percent >= maximum_macro_f1_drop_percent:
        failures.append(
            f"macro-F1 drop {drop_percent:.2f}% is not below "
            f"{maximum_macro_f1_drop_percent:.2f}%"
        )
    if variant_latency >= baseline_latency:
        failures.append(
            f"mean latency {variant_latency:.2f} ms is not below "
            f"baseline {baseline_latency:.2f} ms"
        )
    return failures


def select_optimization_variant(
    variants: dict[str, dict[str, float]],
    baseline: dict[str, float],
    *,
    maximum_macro_f1_drop_percent: float = 5.0,
) -> str | None:
    """Select the fastest quality-eligible variant, using p95 as a tie-breaker."""

    ranked = sorted(
        variants.items(),
        key=lambda item: (
            float(item[1]["mean_latency_ms"]),
            float(item[1]["p95_latency_ms"]),
        ),
    )
    for name, variant in ranked:
        if not check_optimization_quality(
            baseline,
            variant,
            maximum_macro_f1_drop_percent=maximum_macro_f1_drop_percent,
        ):
            return name
    return None
