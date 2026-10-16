"""Airflow schedule for privacy-safe Evidently drift reports."""

from __future__ import annotations

import os
import shlex

from airflow.decorators import dag
from airflow.operators.bash import BashOperator
from pendulum import datetime


@dag(
    dag_id="mena_evidently_drift",
    schedule=os.getenv("MENA_DRIFT_SCHEDULE", "*/10 * * * *"),
    start_date=datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["monitoring", "evidently"],
)
def evidently_drift() -> None:
    """Run one daily report over configured monitoring windows."""

    project_root = os.getenv("MENA_PROJECT_ROOT", "/opt/mena-mlops-project")
    deployment_root = os.getenv(
        "MENA_AUTO_DEPLOY_PROJECT_ROOT",
        project_root,
    )
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
    release_decision_output = os.getenv(
        "MENA_RETRAINING_RELEASE_DECISION",
        f"{project_root}/reports/retraining/release-decision-uri.txt",
    )
    deployment_output_dir = os.getenv(
        "MENA_AUTO_DEPLOY_OUTPUT",
        f"{deployment_root}/models/automatic-candidate-release",
    )
    canary_duration = os.getenv(
        "MENA_AUTO_CANARY_STAGE_DURATION_SECONDS",
        "600",
    )
    candidate_percent = os.getenv("MENA_AUTO_DEPLOY_CANDIDATE_PERCENT", "5")
    feedback_config = f"{project_root}/configs/feedback.yaml"
    run_report = " ".join(
        [
            "cd",
            shlex.quote(project_root),
            "&& python scripts/run_drift.py",
            "--reference",
            shlex.quote(reference),
            "--current",
            shlex.quote(current),
            "--output-dir",
            shlex.quote(output_dir),
        ]
    )
    drift_report = BashOperator(
        task_id="run_evidently_report",
        bash_command=run_report,
    )
    retrain = BashOperator(
        task_id="run_vertex_retraining",
        bash_command=(
            "cd "
            + shlex.quote(project_root)
            + " && python scripts/submit_vertex_retraining.py"
            + " --feedback-config "
            + shlex.quote(feedback_config)
            + " --drift-summary "
            + shlex.quote(f"{output_dir}/drift_summary.json")
            + " --release-decision-output "
            + shlex.quote(release_decision_output)
        ),
    )
    deploy = BashOperator(
        task_id="deploy_approved_retraining_release",
        bash_command=" ".join(
            [
                "if [ ! -s",
                shlex.quote(release_decision_output),
                "]; then",
                "echo 'No successful retraining release to deploy';",
                "exit 0;",
                "fi;",
                "decision_uri=$(cat",
                shlex.quote(release_decision_output),
                ");",
                "cd",
                shlex.quote(deployment_root),
                "&&",
                "python",
                "scripts/deploy_approved_candidate.py",
                "--decision",
                '"$decision_uri"',
                "--output-dir",
                shlex.quote(deployment_output_dir),
                "--candidate-percent",
                shlex.quote(candidate_percent),
                "--stage-duration-seconds",
                shlex.quote(canary_duration),
            ]
        ),
    )
    drift_report >> retrain >> deploy


evidently_drift()
