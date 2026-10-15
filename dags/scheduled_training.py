"""Daily 03:00 full-data Vertex training job."""

from __future__ import annotations

import os
import shlex

from airflow.decorators import dag
from airflow.operators.bash import BashOperator
from pendulum import datetime


@dag(
    dag_id="mena_scheduled_training",
    schedule=os.getenv("MENA_TRAINING_SCHEDULE", "0 3 * * *"),
    start_date=datetime(
        2026,
        1,
        1,
        tz=os.getenv("MENA_TRAINING_TIMEZONE", "UTC"),
    ),
    catchup=False,
    max_active_runs=1,
    tags=["training", "vertex", "scheduled"],
)
def scheduled_training() -> None:
    """Run the normal full-data training path on Vertex at 03:00 local time."""

    project_root = os.getenv("MENA_PROJECT_ROOT", "/opt/mena-mlops-project")
    timezone = os.getenv("MENA_TRAINING_TIMEZONE", "UTC")
    command = " ".join(
        [
            "cd",
            shlex.quote(project_root),
            "&& echo",
            shlex.quote(
                f"03:00 {timezone} scheduled training: "
                "submitting full-data Vertex job"
            ),
            "&& python scripts/submit_vertex_retraining.py",
            "--database",
            shlex.quote(f"{project_root}/data/feedback/reviewed.db"),
            "--feedback-config",
            shlex.quote(f"{project_root}/configs/feedback.yaml"),
            "--data-dir",
            shlex.quote(f"{project_root}/data/processed"),
            "--scheduled",
            "--job-prefix",
            "scheduled-training",
        ]
    )
    BashOperator(task_id="submit_vertex_training", bash_command=command)


scheduled_training()
