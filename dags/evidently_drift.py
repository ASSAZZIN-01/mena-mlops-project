"""Airflow schedule for privacy-safe Evidently drift reports."""

from __future__ import annotations

import os
import shlex

from airflow.decorators import dag
from airflow.operators.bash import BashOperator
from pendulum import datetime


@dag(
    dag_id="mena_evidently_drift",
    schedule="0 2 * * *",
    start_date=datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["monitoring", "evidently"],
)
def evidently_drift() -> None:
    """Run one daily report over configured monitoring windows."""

    project_root = os.getenv("MENA_PROJECT_ROOT", "/opt/mena-mlops-project")
    reference = os.getenv(
        "MENA_DRIFT_REFERENCE",
        f"{project_root}/reports/monitoring/reference.jsonl",
    )
    current = os.getenv(
        "MENA_DRIFT_CURRENT",
        f"{project_root}/reports/monitoring/current.jsonl",
    )
    output_dir = os.getenv(
        "MENA_DRIFT_OUTPUT",
        f"{project_root}/reports/drift",
    )
    run_report = " ".join(
        [
            "cd",
            shlex.quote(project_root),
            "&& uv run python scripts/run_drift.py",
            "--reference",
            shlex.quote(reference),
            "--current",
            shlex.quote(current),
            "--output-dir",
            shlex.quote(output_dir),
        ]
    )
    BashOperator(
        task_id="run_evidently_report",
        bash_command=run_report,
    )


evidently_drift()
