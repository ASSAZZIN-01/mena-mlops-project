"""Durable release lifecycle records for the MENA operations dashboard."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class ReleaseRegistry:
    """Persist retraining, gate, and deployment decisions in SQLite."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS releases (
                    release_id TEXT PRIMARY KEY,
                    trigger TEXT NOT NULL,
                    status TEXT NOT NULL,
                    candidate_variant TEXT,
                    decision_uri TEXT,
                    vertex_job_id TEXT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    deployment_status TEXT NOT NULL DEFAULT 'not_started',
                    decision_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS release_gates (
                    release_id TEXT NOT NULL,
                    gate_name TEXT NOT NULL,
                    accepted INTEGER NOT NULL,
                    failures_json TEXT NOT NULL DEFAULT '[]',
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (release_id, gate_name),
                    FOREIGN KEY (release_id) REFERENCES releases(release_id)
                );
                CREATE TABLE IF NOT EXISTS release_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    release_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    status TEXT,
                    timestamp TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (release_id) REFERENCES releases(release_id)
                );
                CREATE INDEX IF NOT EXISTS release_events_release_idx
                    ON release_events(release_id, timestamp);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _event(
        self,
        connection: sqlite3.Connection,
        release_id: str,
        event_type: str,
        status: str | None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO release_events
                (release_id, event_type, status, timestamp, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                release_id,
                event_type,
                status,
                utc_now(),
                json.dumps(payload or {}, ensure_ascii=False, default=str),
            ),
        )

    def start_release(
        self,
        release_id: str,
        *,
        trigger: str,
        vertex_job_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO releases
                    (release_id, trigger, status, vertex_job_id, started_at)
                VALUES (?, ?, 'training', ?, ?)
                ON CONFLICT(release_id) DO UPDATE SET
                    trigger = excluded.trigger,
                    vertex_job_id = COALESCE(
                        excluded.vertex_job_id, releases.vertex_job_id
                    )
                """,
                (release_id, trigger, vertex_job_id, utc_now()),
            )
            self._event(
                connection,
                release_id,
                "retraining_started",
                "running",
                payload,
            )

    def record_decision(
        self,
        release_id: str,
        *,
        decision_uri: str,
        decision: dict[str, Any],
        payload: dict[str, Any] | None = None,
    ) -> None:
        status = str(decision.get("status", "unknown"))
        selected_variant = decision.get("selected_variant")
        gates = (
            "quality_gate",
            "optimization_gate",
            "deployed_model_gate",
        )
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE releases
                SET status = ?, candidate_variant = ?, decision_uri = ?,
                    completed_at = ?, decision_json = ?
                WHERE release_id = ?
                """,
                (
                    status,
                    str(selected_variant) if selected_variant else None,
                    decision_uri,
                    utc_now(),
                    json.dumps(decision, ensure_ascii=False),
                    release_id,
                ),
            )
            for gate in gates:
                result = decision.get(gate)
                if not isinstance(result, dict):
                    continue
                connection.execute(
                    """
                    INSERT INTO release_gates
                        (release_id, gate_name, accepted, failures_json, metrics_json)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(release_id, gate_name) DO UPDATE SET
                        accepted = excluded.accepted,
                        failures_json = excluded.failures_json,
                        metrics_json = excluded.metrics_json
                    """,
                    (
                        release_id,
                        gate,
                        int(bool(result.get("accepted"))),
                        json.dumps(result.get("failures", []), ensure_ascii=False),
                        json.dumps(result.get("metrics", {}), ensure_ascii=False),
                    ),
                )
            self._event(connection, release_id, "promotion_decision", status, payload)

    def record_deployment(
        self,
        release_id: str,
        *,
        status: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE releases
                SET deployment_status = ?,
                    status = CASE
                        WHEN ? = 'deployed' THEN 'deployed'
                        WHEN ? = 'rolled_back' THEN 'rolled_back'
                        ELSE status
                    END
                WHERE release_id = ?
                """,
                (status, status, status, release_id),
            )
            self._event(connection, release_id, "deployment", status, payload)

    def list_releases(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT release_id, trigger, status, candidate_variant,
                       decision_uri, vertex_job_id, started_at, completed_at,
                       deployment_status
                FROM releases
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_release(self, release_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            release = connection.execute(
                "SELECT * FROM releases WHERE release_id = ?",
                (release_id,),
            ).fetchone()
            if release is None:
                return None
            gates = connection.execute(
                """
                SELECT gate_name, accepted, failures_json, metrics_json
                FROM release_gates WHERE release_id = ?
                ORDER BY gate_name
                """,
                (release_id,),
            ).fetchall()
            events = connection.execute(
                """
                SELECT event_type, status, timestamp, payload_json
                FROM release_events WHERE release_id = ?
                ORDER BY timestamp
                """,
                (release_id,),
            ).fetchall()
        result = dict(release)
        result["decision"] = json.loads(result.pop("decision_json") or "{}")
        result["gates"] = [
            {
                "gate_name": row["gate_name"],
                "accepted": bool(row["accepted"]),
                "failures": json.loads(row["failures_json"]),
                "metrics": json.loads(row["metrics_json"]),
            }
            for row in gates
        ]
        result["events"] = [
            {
                "event_type": row["event_type"],
                "status": row["status"],
                "timestamp": row["timestamp"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in events
        ]
        return result

    def overview(self) -> dict[str, Any]:
        releases = self.list_releases(limit=1)
        return {
            "latest_release": releases[0] if releases else None,
            "release_count": self._count("releases"),
        }

    def _count(self, table: str) -> int:
        if table not in {"releases", "release_events"}:
            raise ValueError("unsupported registry table")
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT COUNT(*) AS count FROM {table}"
            ).fetchone()
        return int(row["count"])
