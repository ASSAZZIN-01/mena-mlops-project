"""Model-quality gates for candidate promotion."""

from __future__ import annotations


def evaluate_quality_gate(
    stable: dict[str, object],
    candidate: dict[str, object],
    *,
    minimum_macro_f1: float,
    minimum_improvement: float,
    minimum_class_f1: dict[str, float],
) -> tuple[bool, list[str]]:
    """Compare verified-label metrics and return a decision with reasons."""

    reasons: list[str] = []
    stable_macro = float(stable["macro_f1"])
    candidate_macro = float(candidate["macro_f1"])
    if candidate_macro < minimum_macro_f1:
        reasons.append("candidate macro-F1 is below the minimum")
    if candidate_macro - stable_macro < minimum_improvement:
        reasons.append("candidate macro-F1 does not improve over stable")

    candidate_classes = candidate.get("per_class")
    if not isinstance(candidate_classes, dict):
        raise ValueError("candidate metrics must contain per_class metrics")
    for label, minimum in minimum_class_f1.items():
        metrics = candidate_classes.get(label)
        if not isinstance(metrics, dict) or float(metrics.get("f1", 0)) < minimum:
            reasons.append(f"candidate {label} F1 is below the minimum")
    return not reasons, reasons
