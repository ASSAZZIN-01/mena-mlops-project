#!/usr/bin/env bash
set -euo pipefail

mkdir -p /opt/mena-mlops-project/reports/drift
chown -R airflow:root /opt/mena-mlops-project/reports/drift

exec su airflow -s /bin/bash -c "exec airflow $(printf '%q ' "$@")"
