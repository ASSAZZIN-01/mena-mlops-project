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


def compare_deployed_model(
    stable: dict[str, object],
    candidate: dict[str, object],
    *,
    maximum_regression_percent: float = 3.0,
) -> list[str]:
    """Reject quality regressions and material serving regressions."""

    stable_quality = stable.get("quality")
    candidate_quality = candidate.get("quality")
    if not isinstance(stable_quality, dict) or not isinstance(candidate_quality, dict):
        raise ValueError("deployed baseline and candidate need quality metrics")
    failures: list[str] = []
    if float(candidate_quality["macro_f1"]) < float(stable_quality["macro_f1"]):
        failures.append("candidate macro-F1 is below the deployed model")
    stable_classes = stable_quality.get("per_class")
    candidate_classes = candidate_quality.get("per_class")
    if not isinstance(stable_classes, dict) or not isinstance(candidate_classes, dict):
        raise ValueError("quality metrics need per_class metrics")
    for label, stable_metrics in stable_classes.items():
        candidate_metrics = candidate_classes.get(label)
        if not isinstance(stable_metrics, dict) or not isinstance(
            candidate_metrics, dict
        ):
            failures.append(f"candidate is missing deployed {label} metrics")
        elif float(candidate_metrics["f1"]) < float(stable_metrics["f1"]):
            failures.append(f"candidate {label} F1 is below the deployed model")

    stable_latency = float(stable["latency"]["mean_ms"])
    candidate_latency = float(candidate["latency"]["mean_ms"])
    stable_throughput = float(stable["latency"]["throughput_rps"])
    candidate_throughput = float(candidate["latency"]["throughput_rps"])
    if candidate_latency > stable_latency * (1 + maximum_regression_percent / 100):
        failures.append(
            "candidate latency regresses by more than the allowed budget"
        )
    if candidate_throughput < stable_throughput * (
        1 - maximum_regression_percent / 100
    ):
        failures.append(
            "candidate throughput regresses by more than the allowed budget"
        )
    return failures
