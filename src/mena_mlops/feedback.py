"""Durable human feedback storage and retraining threshold checks."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

LABELS = ("negative", "neutral", "positive")


@dataclass(frozen=True)
class FeedbackItem:
    id: int
    text: str
    predicted_label: str
    confidence: float
    reviewed_label: str | None
    reviewed_at: str | None
    review_status: str


@dataclass(frozen=True)
class RetrainingCheckpoint:
    """Last feedback included in a completed retraining attempt."""

    last_feedback_id: int
    last_drift_fingerprint: str | None
    last_reviewed_at: str | None = None


@dataclass(frozen=True)
class DeploymentCheckpoint:
    """Last feedback represented by a successfully deployed model."""

    last_feedback_id: int
    last_reviewed_at: str | None


class FeedbackStore:
    """Persist uncertain predictions and reviewed labels in SQLite."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    predicted_label TEXT NOT NULL,
                    confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
                    reviewed_label TEXT CHECK(
                        reviewed_label IS NULL OR reviewed_label IN
                        ('negative', 'neutral', 'positive')
                    ),
                    reviewed_at TEXT,
                    review_status TEXT NOT NULL DEFAULT 'pending'
                )
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(feedback)")
            }
            if "review_status" not in columns:
                connection.execute(
                    """
                    ALTER TABLE feedback
                    ADD COLUMN review_status TEXT NOT NULL DEFAULT 'pending'
                    """
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS retraining_state (
                    id INTEGER PRIMARY KEY CHECK(id = 1),
                    last_feedback_id INTEGER NOT NULL DEFAULT 0,
                    last_drift_fingerprint TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(retraining_state)")
            }
            if "last_reviewed_at" not in columns:
                connection.execute(
                    """
                    ALTER TABLE retraining_state
                    ADD COLUMN last_reviewed_at TEXT
                    """
                )
                connection.execute(
                    """
                    UPDATE retraining_state
                    SET last_reviewed_at = updated_at
                    WHERE last_reviewed_at IS NULL
                    """
                )
            for column, definition in (
                ("last_attempt_feedback_id", "INTEGER NOT NULL DEFAULT 0"),
                ("last_attempt_reviewed_at", "TEXT"),
                ("last_attempt_drift_fingerprint", "TEXT"),
                ("last_deployed_feedback_id", "INTEGER NOT NULL DEFAULT 0"),
                ("last_deployed_reviewed_at", "TEXT"),
            ):
                if column not in columns:
                    connection.execute(
                        f"ALTER TABLE retraining_state ADD COLUMN {column} {definition}"
                    )
            connection.execute(
                """
                UPDATE retraining_state
                SET last_attempt_feedback_id = last_feedback_id,
                    last_attempt_reviewed_at = last_reviewed_at,
                    last_attempt_drift_fingerprint = last_drift_fingerprint
                WHERE id = 1
                  AND last_attempt_feedback_id = 0
                  AND last_feedback_id != 0
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def add_uncertain(
        self,
        text: str,
        predicted_label: str,
        confidence: float,
    ) -> int:
        """Store one uncertain prediction and return its database ID."""

        if not text.strip():
            raise ValueError("feedback text must not be empty")
        if predicted_label not in LABELS:
            raise ValueError(f"unsupported predicted label: {predicted_label}")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO feedback (text, predicted_label, confidence)
                VALUES (?, ?, ?)
                """,
                (text, predicted_label, confidence),
            )
            return int(cursor.lastrowid)

    def next_pending(self) -> FeedbackItem | None:
        """Return the oldest unreviewed item."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, text, predicted_label, confidence,
                       reviewed_label, reviewed_at, review_status
                FROM feedback
                WHERE reviewed_label IS NULL AND review_status = 'pending'
                ORDER BY id
                LIMIT 1
                """
            ).fetchone()
        return FeedbackItem(**dict(row)) if row else None

    def pending_items(self) -> list[FeedbackItem]:
        """Return all pending items in queue order."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, text, predicted_label, confidence,
                       reviewed_label, reviewed_at, review_status
                FROM feedback
                WHERE reviewed_label IS NULL AND review_status = 'pending'
                ORDER BY id
                """
            ).fetchall()
        return [FeedbackItem(**dict(row)) for row in rows]

    def reviewed_items(self) -> list[FeedbackItem]:
        """Return completed annotations in review order."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, text, predicted_label, confidence,
                       reviewed_label, reviewed_at, review_status
                FROM feedback
                WHERE review_status != 'pending'
                ORDER BY id
                """
            ).fetchall()
        return [FeedbackItem(**dict(row)) for row in rows]

    def review(
        self,
        item_id: int,
        correct: bool,
        corrected_label: str | None = None,
    ) -> None:
        """Record a review, requiring a corrected label for incorrect predictions."""

        item = self._get(item_id)
        reviewed_label = item.predicted_label if correct else corrected_label
        if reviewed_label not in LABELS:
            raise ValueError("an incorrect review requires a valid corrected label")
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE feedback
                SET reviewed_label = ?, reviewed_at = ?, review_status = 'validated'
                WHERE id = ? AND reviewed_label IS NULL
                """,
                (reviewed_label, datetime.now(UTC).isoformat(), item_id),
            )

    def defer(self, item_id: int) -> None:
        """Defer an uncertain item without adding it to validated data."""

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE feedback
                SET reviewed_at = ?, review_status = 'deferred'
                WHERE id = ? AND reviewed_label IS NULL
                """,
                (datetime.now(UTC).isoformat(), item_id),
            )

    def revise(
        self,
        item_id: int,
        correct: bool,
        corrected_label: str | None = None,
        defer: bool = False,
    ) -> None:
        """Replace an existing annotation without creating another feedback row."""

        item = self._get(item_id)
        if defer:
            reviewed_label = None
            status = "deferred"
        else:
            reviewed_label = item.predicted_label if correct else corrected_label
            if reviewed_label not in LABELS:
                raise ValueError("an incorrect review requires a valid corrected label")
            status = "validated"
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE feedback
                SET reviewed_label = ?, reviewed_at = ?, review_status = ?
                WHERE id = ?
                """,
                (
                    reviewed_label,
                    datetime.now(UTC).isoformat(),
                    status,
                    item_id,
                ),
            )

    def get_item(self, item_id: int) -> FeedbackItem:
        """Return one feedback item by ID."""

        return self._get(item_id)

    def counts(self) -> dict[str, int]:
        """Count reviewed examples by their validated label."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT reviewed_label, COUNT(*) AS count
                FROM feedback
                WHERE reviewed_label IS NOT NULL
                GROUP BY reviewed_label
                """
            ).fetchall()
        return {
            label: next(
                (
                    int(row["count"])
                    for row in rows
                    if row["reviewed_label"] == label
                ),
                0,
            )
            for label in LABELS
        }

    def annotation_summary(self, audit_path: Path) -> dict[str, dict[str, int]]:
        """Summarize prediction denominators and annotation states by label."""

        summary = {
            label: {
                "total_predictions": 0,
                "suspected": 0,
                "reviewed": 0,
                "confusing": 0,
                "pending": 0,
            }
            for label in LABELS
        }
        if audit_path.is_file():
            with sqlite3.connect(audit_path) as audit_connection:
                rows = audit_connection.execute(
                    """
                    SELECT prediction, COUNT(*) AS count
                    FROM predictions
                    GROUP BY prediction
                    """
                ).fetchall()
        else:
            rows = []
        for label, count in rows:
            if label in summary:
                summary[str(label)]["total_predictions"] = int(count)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT predicted_label, review_status, COUNT(*) AS count
                FROM feedback
                GROUP BY predicted_label, review_status
                """
            ).fetchall()
        for label, status, count in rows:
            if label not in summary:
                continue
            summary[str(label)]["suspected"] += int(count)
            state_key = {
                "validated": "reviewed",
                "deferred": "confusing",
                "pending": "pending",
            }.get(str(status))
            if state_key is not None:
                summary[str(label)][state_key] += int(count)
        return summary

    def counts_since(
        self,
        feedback_id: int,
        reviewed_after: str | None = None,
    ) -> dict[str, int]:
        """Count reviewed examples added after a successful retraining."""

        with self._connect() as connection:
            if reviewed_after is None:
                rows = connection.execute(
                    """
                    SELECT reviewed_label, COUNT(*) AS count
                    FROM feedback
                    WHERE reviewed_label IS NOT NULL AND id > ?
                    GROUP BY reviewed_label
                    """,
                    (feedback_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT reviewed_label, COUNT(*) AS count
                    FROM feedback
                    WHERE reviewed_label IS NOT NULL AND reviewed_at > ?
                    GROUP BY reviewed_label
                    """,
                    (reviewed_after,),
                ).fetchall()
        return {
            label: next(
                (
                    int(row["count"])
                    for row in rows
                    if row["reviewed_label"] == label
                ),
                0,
            )
            for label in LABELS
        }

    def thresholds_reached(self, thresholds: dict[str, int]) -> bool:
        """Return whether every configured class threshold has been reached."""

        counts = self.counts()
        return all(
            counts.get(label, 0) >= required
            for label, required in thresholds.items()
        )

    def retraining_checkpoint(self) -> RetrainingCheckpoint:
        """Return the last completed retraining-attempt checkpoint."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT last_attempt_feedback_id AS last_feedback_id,
                       last_attempt_drift_fingerprint AS last_drift_fingerprint,
                       last_attempt_reviewed_at AS last_reviewed_at
                FROM retraining_state
                WHERE id = 1
                """
            ).fetchone()
        if row is None:
            return RetrainingCheckpoint(0, None)
        return RetrainingCheckpoint(
            last_feedback_id=int(row["last_feedback_id"]),
            last_drift_fingerprint=row["last_drift_fingerprint"],
            last_reviewed_at=row["last_reviewed_at"],
        )

    def successful_deployment_checkpoint(self) -> DeploymentCheckpoint:
        """Return the last feedback represented in production."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT last_deployed_feedback_id, last_deployed_reviewed_at
                FROM retraining_state
                WHERE id = 1
                """
            ).fetchone()
        if row is None:
            return DeploymentCheckpoint(0, None)
        return DeploymentCheckpoint(
            last_feedback_id=int(row["last_deployed_feedback_id"]),
            last_reviewed_at=row["last_deployed_reviewed_at"],
        )

    def latest_reviewed_id(self) -> int:
        """Return the highest ID of any reviewed example."""

        with self._connect() as connection:
            value = connection.execute(
                """
                SELECT COALESCE(MAX(id), 0)
                FROM feedback
                WHERE reviewed_label IS NOT NULL
                """
            ).fetchone()[0]
        return int(value)

    def latest_reviewed_at(self) -> str | None:
        """Return the review timestamp used as the retraining cursor."""

        with self._connect() as connection:
            value = connection.execute(
                """
                SELECT MAX(reviewed_at)
                FROM feedback
                WHERE reviewed_label IS NOT NULL
                """
            ).fetchone()[0]
        return str(value) if value is not None else None

    def mark_retraining_attempt(
        self,
        feedback_id: int,
        drift_fingerprint: str | None,
        reviewed_at: str | None = None,
    ) -> None:
        """Advance the attempt cursor after a completed Vertex release attempt."""

        if feedback_id < 0:
            raise ValueError("feedback checkpoint cannot be negative")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO retraining_state (
                    id, last_feedback_id, last_drift_fingerprint,
                    last_reviewed_at, last_attempt_feedback_id,
                    last_attempt_drift_fingerprint, last_attempt_reviewed_at,
                    updated_at
                )
                VALUES (1, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    last_feedback_id = excluded.last_feedback_id,
                    last_drift_fingerprint = excluded.last_drift_fingerprint,
                    last_reviewed_at = excluded.last_reviewed_at,
                    last_attempt_feedback_id = excluded.last_attempt_feedback_id,
                    last_attempt_drift_fingerprint =
                        excluded.last_attempt_drift_fingerprint,
                    last_attempt_reviewed_at = excluded.last_attempt_reviewed_at,
                    updated_at = excluded.updated_at
                """,
                (
                    feedback_id,
                    drift_fingerprint,
                    reviewed_at or datetime.now(UTC).isoformat(),
                    feedback_id,
                    drift_fingerprint,
                    reviewed_at or datetime.now(UTC).isoformat(),
                    datetime.now(UTC).isoformat(),
                ),
            )

    def mark_successful_deployment(
        self,
        feedback_id: int,
        reviewed_at: str | None = None,
    ) -> None:
        """Advance the production cursor after deployment and canary success."""

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO retraining_state (
                    id, last_feedback_id, last_deployed_feedback_id,
                    last_deployed_reviewed_at, updated_at
                )
                VALUES (1, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    last_deployed_feedback_id = excluded.last_deployed_feedback_id,
                    last_deployed_reviewed_at = excluded.last_deployed_reviewed_at,
                    updated_at = excluded.updated_at
                """,
                (
                    feedback_id,
                    feedback_id,
                    reviewed_at or datetime.now(UTC).isoformat(),
                    datetime.now(UTC).isoformat(),
                ),
            )

    def mark_retraining_succeeded(
        self,
        feedback_id: int,
        drift_fingerprint: str | None,
        reviewed_at: str | None = None,
    ) -> None:
        """Backward-compatible alias for recording a completed attempt."""

        self.mark_retraining_attempt(feedback_id, drift_fingerprint, reviewed_at)

    def export_reviewed(
        self,
        path: Path,
        after_id: int = 0,
        through_id: int | None = None,
        reviewed_after: str | None = None,
        reviewed_through: str | None = None,
    ) -> int:
        """Export only reviewed examples in a retraining batch."""

        with self._connect() as connection:
            if reviewed_after is None:
                upper_bound = through_id if through_id is not None else 2**63 - 1
                frame = pd.read_sql_query(
                    """
                    SELECT id, text, reviewed_label AS label
                    FROM feedback
                    WHERE reviewed_label IS NOT NULL
                      AND id > ?
                      AND id <= ?
                    ORDER BY id
                    """,
                    connection,
                    params=(after_id, upper_bound),
                )
            else:
                upper_bound = reviewed_through or datetime.now(UTC).isoformat()
                frame = pd.read_sql_query(
                    """
                    SELECT id, text, reviewed_label AS label
                    FROM feedback
                    WHERE reviewed_label IS NOT NULL
                      AND reviewed_at > ?
                      AND reviewed_at <= ?
                    ORDER BY reviewed_at, id
                    """,
                    connection,
                    params=(reviewed_after, upper_bound),
                )
        frame["source"] = "human_feedback"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return len(frame)

    def _get(self, item_id: int) -> FeedbackItem:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, text, predicted_label, confidence,
                       reviewed_label, reviewed_at, review_status
                FROM feedback WHERE id = ?
                """,
                (item_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"feedback item not found: {item_id}")
        return FeedbackItem(**dict(row))
