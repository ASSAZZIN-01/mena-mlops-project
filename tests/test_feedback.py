"""Tests for durable feedback and retraining thresholds."""

from pathlib import Path

import pandas as pd
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


def test_protected_feedback_is_excluded_from_queue_and_export(
    tmp_path: Path,
) -> None:
    store = FeedbackStore(
        tmp_path / "feedback.db",
        protected_texts={"protected example"},
    )
    item_id = store.add_uncertain(" Protected   Example ", "positive", 0.5)

    assert store.next_pending() is None
    assert store.exclusion_counts() == {"protected_split_overlap": 1}
    export_path = tmp_path / "feedback.parquet"
    assert store.export_reviewed(export_path) == 0
    assert store.get_item(item_id).review_status == "excluded"


def test_deferred_feedback_is_not_validated_or_pending(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.db")
    item_id = store.add_uncertain("نص غامض", "neutral", 0.5)

    store.defer(item_id)

    assert store.next_pending() is None
    assert store.counts() == {"negative": 0, "neutral": 0, "positive": 0}


def test_annotation_summary_combines_audit_denominators_and_queue_states(
    tmp_path: Path,
) -> None:
    audit_path = tmp_path / "audit.db"
    import sqlite3

    with sqlite3.connect(audit_path) as connection:
        connection.execute(
            "CREATE TABLE predictions (prediction TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO predictions (prediction) VALUES (?)",
            [("negative",), ("negative",), ("neutral",), ("positive",)],
        )

    store = FeedbackStore(tmp_path / "feedback.db")
    reviewed = store.add_uncertain("نص مراجع", "negative", 0.5)
    store.review(reviewed, correct=True)
    deferred = store.add_uncertain("نص غامض", "negative", 0.5)
    store.defer(deferred)
    store.add_uncertain("نص معلق", "positive", 0.5)

    assert store.annotation_summary(audit_path) == {
        "negative": {
            "total_predictions": 2,
            "suspected": 2,
            "reviewed": 1,
            "confusing": 1,
            "pending": 0,
        },
        "neutral": {
            "total_predictions": 1,
            "suspected": 0,
            "reviewed": 0,
            "confusing": 0,
            "pending": 0,
        },
        "positive": {
            "total_predictions": 1,
            "suspected": 1,
            "reviewed": 0,
            "confusing": 0,
            "pending": 1,
        },
    }


def test_annotation_summary_handles_missing_audit_database(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.db")
    assert store.annotation_summary(tmp_path / "missing.db")["negative"] == {
        "total_predictions": 0,
        "suspected": 0,
        "reviewed": 0,
        "confusing": 0,
        "pending": 0,
    }


def _review(store: FeedbackStore, label: str) -> int:
    item_id = store.add_uncertain("نص للمراجعة", "neutral", 0.5)
    store.review(item_id, correct=False, corrected_label=label)
    return item_id


def test_retraining_checkpoint_consumes_a_batch_without_deleting_history(
    tmp_path: Path,
) -> None:
    store = FeedbackStore(tmp_path / "feedback.db")
    first_id = _review(store, "negative")
    _review(store, "positive")
    upper_bound = store.latest_reviewed_id()

    assert store.counts_since(0) == {"negative": 1, "neutral": 0, "positive": 1}
    export_path = tmp_path / "batch.parquet"
    assert store.export_reviewed(export_path, after_id=0, through_id=upper_bound) == 2
    assert set(pd.read_parquet(export_path)["id"]) == {first_id, upper_bound}

    store.mark_retraining_succeeded(upper_bound, "drift-a")

    checkpoint = store.retraining_checkpoint()
    assert checkpoint.last_feedback_id == upper_bound
    assert checkpoint.last_drift_fingerprint == "drift-a"
    assert store.counts_since(checkpoint.last_feedback_id) == {
        "negative": 0,
        "neutral": 0,
        "positive": 0,
    }
    assert store.counts() == {"negative": 1, "neutral": 0, "positive": 1}


def test_failed_retraining_does_not_advance_checkpoint(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.db")
    reviewed_id = _review(store, "positive")

    checkpoint = store.retraining_checkpoint()
    assert checkpoint.last_feedback_id == 0
    assert store.counts_since(checkpoint.last_feedback_id)["positive"] == 1
    assert store.latest_reviewed_id() == reviewed_id


def test_late_review_of_older_item_is_counted_after_checkpoint(
    tmp_path: Path,
) -> None:
    store = FeedbackStore(tmp_path / "feedback.db")
    old_item = store.add_uncertain("نص قديم", "positive", 0.5)
    store.add_uncertain("نص آخر", "neutral", 0.5)
    store.review(2, correct=True)
    checkpoint_time = store.latest_reviewed_at()
    store.mark_retraining_succeeded(2, None, reviewed_at=checkpoint_time)

    store.review(old_item, correct=True)
    checkpoint = store.retraining_checkpoint()

    assert store.counts_since(
        checkpoint.last_feedback_id,
        reviewed_after=checkpoint.last_reviewed_at,
    ) == {"negative": 0, "neutral": 0, "positive": 1}


def test_attempt_and_deployment_watermarks_are_independent(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.db")
    first_id = _review(store, "negative")
    store.mark_retraining_attempt(first_id, "attempt-a", store.latest_reviewed_at())
    second_id = _review(store, "positive")

    attempt = store.retraining_checkpoint()
    deployed = store.successful_deployment_checkpoint()
    assert attempt.last_feedback_id == first_id
    assert deployed.last_feedback_id == 0
    assert store.counts_since(
        attempt.last_feedback_id,
        reviewed_after=attempt.last_reviewed_at,
    ) == {"negative": 0, "neutral": 0, "positive": 1}

    store.mark_successful_deployment(first_id, attempt.last_reviewed_at)
    deployed = store.successful_deployment_checkpoint()
    assert deployed.last_feedback_id == first_id
    assert store.counts_since(
        deployed.last_feedback_id,
        reviewed_after=deployed.last_reviewed_at,
    ) == {"negative": 0, "neutral": 0, "positive": 1}
    assert second_id > first_id


def test_revise_updates_existing_annotation_without_duplicate_feedback(
    tmp_path: Path,
) -> None:
    store = FeedbackStore(tmp_path / "feedback.db")
    item_id = store.add_uncertain("نص قابل للتصحيح", "negative", 0.5)
    store.review(item_id, correct=True)

    store.revise(item_id, correct=False, corrected_label="positive")

    item = store.get_item(item_id)
    assert item.id == item_id
    assert item.reviewed_label == "positive"
    assert item.review_status == "validated"
    assert store.next_pending() is None
