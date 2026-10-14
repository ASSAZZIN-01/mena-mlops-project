#!/usr/bin/env bash
set -euo pipefail

MLFLOW_IMAGE="${MLFLOW_IMAGE:-mena-mlflow:3.16.1}"
MLFLOW_DATA_DIR="${MLFLOW_DATA_DIR:-$HOME/mlflow/data}"
MLFLOW_ARTIFACT_LOCATION="${MLFLOW_ARTIFACT_LOCATION:-gs://mena-mlops-project-dvc/mlflow}"
MLFLOW_ALLOWED_HOSTS="${MLFLOW_ALLOWED_HOSTS:-localhost,127.0.0.1,mlflow-server}"

mkdir -p "$MLFLOW_DATA_DIR"

cat > /tmp/mena-mlflow.Dockerfile <<'EOF'
FROM ghcr.io/mlflow/mlflow:v3.16.1

RUN pip install --no-cache-dir "google-cloud-storage>=2.18"
EOF

docker build -t "$MLFLOW_IMAGE" -f /tmp/mena-mlflow.Dockerfile /tmp
docker rm -f mlflow 2>/dev/null || true

docker run -d \
  --name mlflow \
  --restart unless-stopped \
  -p 5000:5000 \
  -v "$MLFLOW_DATA_DIR:/mlflow" \
  "$MLFLOW_IMAGE" \
  mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:////mlflow/mlflow.db \
    --default-artifact-root "$MLFLOW_ARTIFACT_LOCATION" \
    --serve-artifacts \
    --allowed-hosts "$MLFLOW_ALLOWED_HOSTS"

for attempt in $(seq 1 30); do
  if curl --fail --silent http://127.0.0.1:5000/health >/dev/null; then
    echo "MLflow is healthy on port 5000."
    exit 0
  fi
  sleep 2
done

echo "MLflow did not become healthy." >&2
docker logs --tail 100 mlflow >&2
exit 1
