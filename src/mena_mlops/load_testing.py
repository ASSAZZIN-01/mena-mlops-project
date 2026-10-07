"""Shared validation helpers for load-test responses."""

from __future__ import annotations

from typing import Any


def valid_prediction_response(body: Any) -> bool:
    if not isinstance(body, dict):
        return False
    probabilities = body.get("probabilities")
    if not isinstance(probabilities, dict):
        return False
    if body.get("label") not in {"negative", "neutral", "positive"}:
        return False
    if set(probabilities) != {"negative", "neutral", "positive"}:
        return False
    if not all(isinstance(value, int | float) for value in probabilities.values()):
        return False
    return abs(sum(probabilities.values()) - 1.0) <= 0.01 and bool(
        body.get("model_version")
    )
