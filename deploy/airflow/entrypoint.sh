#!/usr/bin/env bash
set -euo pipefail

if [[ -s /run/secrets/gcp-key.json ]]; then
    install -o airflow -g root -m 0400 \
        /run/secrets/gcp-key.json /tmp/mena-gcp-key.json
else
    rm -f /tmp/mena-gcp-key.json
fi

chown airflow:root /opt/airflow
mkdir -p /opt/airflow/logs /opt/airflow/plugins
find /opt/airflow -mindepth 1 -path /opt/airflow/dags -prune -o \
    -exec chown airflow:root {} +
mkdir -p /opt/mena-mlops-project/reports/drift
chown -R airflow:root /opt/mena-mlops-project/reports/drift
mkdir -p /opt/mena-mlops-project/data/feedback
chown airflow:root /opt/mena-mlops-project/data/feedback
if [[ -f /opt/mena-mlops-project/data/feedback/reviewed.db ]]; then
    chown airflow:root /opt/mena-mlops-project/data/feedback/reviewed.db
fi

exec su airflow -s /bin/bash -c "exec airflow $(printf '%q ' "$@")"
