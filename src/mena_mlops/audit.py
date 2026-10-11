"""Durable prediction audit records."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


class PredictionAuditStore:
    """Persist prediction metadata without changing the prediction contract."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    request_id TEXT PRIMARY KEY,
                    input_id TEXT,
                    source TEXT NOT NULL,
                    text TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
                    probabilities TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    review_status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def record(
        self,
        *,
        text: str,
        prediction: str,
        probabilities: dict[str, float],
        model_version: str,
        source: str,
        input_id: str | None = None,
        request_id: str | None = None,
        review_min_confidence: float = 0.25,
        review_max_confidence: float = 0.75,
    ) -> str:
        """Record one prediction and return its stable request ID."""

        if not text.strip():
            raise ValueError("audit text must not be empty")
        if source not in {"live", "batch", "stream"}:
            raise ValueError(f"unsupported audit source: {source}")
        confidence = float(probabilities[prediction])
        status = (
            "pending"
            if review_min_confidence <= confidence <= review_max_confidence
            else "not_required"
        )
        identifier = request_id or str(uuid4())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO predictions (
                    request_id, input_id, source, text, prediction, confidence,
                    probabilities, model_version, review_status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    input_id,
                    source,
                    text,
                    prediction,
                    confidence,
                    json.dumps(probabilities, ensure_ascii=False, sort_keys=True),
                    model_version,
                    status,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return identifier
