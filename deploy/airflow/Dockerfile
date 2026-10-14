FROM apache/airflow:2.10.4-python3.12

COPY pyproject.toml /opt/mena-mlops-project/pyproject.toml
COPY src /opt/mena-mlops-project/src
COPY scripts /opt/mena-mlops-project/scripts
COPY configs /opt/mena-mlops-project/configs
COPY deploy/airflow/entrypoint.sh /usr/local/bin/mena-airflow-entrypoint

USER airflow

RUN pip install --no-cache-dir \
    "evidently>=0.7" \
    "pandas>=2.2" \
    "google-auth>=2.35" \
    "google-cloud-storage>=2.18"

ENV PYTHONPATH=/opt/mena-mlops-project/src

USER root

RUN chmod +x /usr/local/bin/mena-airflow-entrypoint
