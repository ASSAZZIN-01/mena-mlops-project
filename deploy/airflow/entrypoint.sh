#!/usr/bin/env bash
set -euo pipefail

chown -R airflow:root /opt/airflow
mkdir -p /opt/mena-mlops-project/reports/drift
chown -R airflow:root /opt/mena-mlops-project/reports/drift
mkdir -p /opt/mena-mlops-project/data/feedback
chown airflow:root /opt/mena-mlops-project/data/feedback
if [[ -f /opt/mena-mlops-project/data/feedback/reviewed.db ]]; then
    chown airflow:root /opt/mena-mlops-project/data/feedback/reviewed.db
fi

exec su airflow -s /bin/bash -c "exec airflow $(printf '%q ' "$@")"
