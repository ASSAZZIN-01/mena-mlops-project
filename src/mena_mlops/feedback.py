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
                    reviewed_at TEXT
                )
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
                       reviewed_label, reviewed_at
                FROM feedback
                WHERE reviewed_label IS NULL
                ORDER BY id
                LIMIT 1
                """
            ).fetchone()
        return FeedbackItem(**dict(row)) if row else None

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
                SET reviewed_label = ?, reviewed_at = ?
                WHERE id = ? AND reviewed_label IS NULL
                """,
                (reviewed_label, datetime.now(UTC).isoformat(), item_id),
            )

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

    def thresholds_reached(self, thresholds: dict[str, int]) -> bool:
        """Return whether every configured class threshold has been reached."""

        counts = self.counts()
        return all(
            counts.get(label, 0) >= required
            for label, required in thresholds.items()
        )

    def export_reviewed(self, path: Path) -> int:
        """Export reviewed examples for training without pending records."""

        with self._connect() as connection:
            frame = pd.read_sql_query(
                """
                SELECT id, text, reviewed_label AS label
                FROM feedback
                WHERE reviewed_label IS NOT NULL
                ORDER BY id
                """,
                connection,
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
                       reviewed_label, reviewed_at
                FROM feedback WHERE id = ?
                """,
                (item_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"feedback item not found: {item_id}")
        return FeedbackItem(**dict(row))
