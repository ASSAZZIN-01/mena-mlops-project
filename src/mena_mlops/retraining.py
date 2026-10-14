"""Retraining trigger decisions shared by Airflow and local workflows."""

from __future__ import annotations


def feedback_thresholds_reached(
    counts: dict[str, int], thresholds: dict[str, int]
) -> bool:
    """Return whether every configured feedback threshold has been reached."""

    return all(
        counts.get(label, 0) >= required
        for label, required in thresholds.items()
    )


def drift_triggered(summary: dict[str, object] | None) -> bool:
    """Return whether Evidently or the PSI alert requests retraining."""

    if summary is None:
        return False
    return bool(summary.get("drift_detected") or summary.get("psi_alert"))
