"""Retraining trigger decisions shared by Airflow and local workflows."""

from __future__ import annotations

import hashlib
import json


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


def drift_fingerprint(summary: dict[str, object] | None) -> str | None:
    """Create a stable identity for one persisted drift summary."""

    if not drift_triggered(summary):
        return None
    payload = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
