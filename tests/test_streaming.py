"""Tests for Redis Streams inference contracts."""

import json

import pytest

from mena_mlops.inference.streaming import (
    build_prediction_event,
    parse_review_event,
)


class FakePredictor:
    model_version = "stream-v1"

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        return "positive", {"negative": 0.1, "neutral": 0.2, "positive": 0.7}


def test_stream_event_is_validated() -> None:
    event = parse_review_event({"text": "خدمة ممتازة", "event_id": "r-1"})
    assert event.event_id == "r-1"

    with pytest.raises(ValueError, match="non-empty"):
        parse_review_event({"text": " "})


def test_prediction_event_contains_model_metadata() -> None:
    event = build_prediction_event(
        "1-0",
        parse_review_event({"text": "خدمة ممتازة"}),
        FakePredictor(),
    )

    assert event["message_id"] == "1-0"
    assert event["prediction"] == "positive"
    assert event["model_version"] == "stream-v1"
    assert json.loads(event["probabilities"])["positive"] == 0.7
