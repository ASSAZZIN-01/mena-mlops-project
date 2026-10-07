"""Tests for durable feedback and retraining thresholds."""

from pathlib import Path

import pytest

from mena_mlops.feedback import FeedbackStore


def test_feedback_review_and_thresholds(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.db")
    item_id = store.add_uncertain("خدمة ممتازة", "neutral", 0.5)

    store.review(item_id, correct=False, corrected_label="positive")

    assert store.counts() == {"negative": 0, "neutral": 0, "positive": 1}
    assert store.thresholds_reached({"positive": 1})


def test_incorrect_review_requires_corrected_label(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.db")
    item_id = store.add_uncertain("خدمة سيئة", "negative", 0.5)

    with pytest.raises(ValueError, match="corrected label"):
        store.review(item_id, correct=False)
