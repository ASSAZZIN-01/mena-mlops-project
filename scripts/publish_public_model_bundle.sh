#!/usr/bin/env bash
set -euo pipefail

# Publish only the reviewer-safe serving files. Private release metadata,
# model cards, datasets, and MLflow artifacts are never copied.

RELEASE_ID="${RELEASE_ID:-clean-candidate-v21-20261018-134733}"
MODEL_VERSION="${MODEL_VERSION:-clean-candidate-v21}"
MLFLOW_VERSION="${MLFLOW_VERSION:-21}"
PRIVATE_BUCKET="${PRIVATE_BUCKET:-gs://mena-mlops-project-dvc}"
PUBLIC_BUCKET="${PUBLIC_BUCKET:-gs://mena-mlops-public-models}"
REMOVE_LEGACY="${REMOVE_LEGACY:-false}"

command -v gcloud >/dev/null || {
  echo "gcloud is required; run this from authenticated Google Cloud Shell." >&2
  exit 1
}
command -v curl >/dev/null || {
  echo "curl is required." >&2
  exit 1
}

source_uri="${PRIVATE_BUCKET}/vertex-staging/${RELEASE_ID}/release"
public_uri="${PUBLIC_BUCKET}/models/${MODEL_VERSION}"
work_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$work_dir"
}
trap cleanup EXIT

for filename in model-int8.onnx tokenizer.json tokenizer_config.json; do
  gcloud storage cp "${source_uri}/${filename}" "${work_dir}/${filename}"
done

python3 - "$work_dir" "$MODEL_VERSION" "$RELEASE_ID" "$MLFLOW_VERSION" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

work_dir = Path(sys.argv[1])
model_version, release_id, mlflow_version = sys.argv[2:]
files = {}
for name in ("model-int8.onnx", "tokenizer.json", "tokenizer_config.json"):
    path = work_dir / name
    files[name] = {
        "object": f"models/{model_version}/{name}",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }

(work_dir / "manifest.json").write_text(
    json.dumps(
        {
            "model_version": model_version,
            "backend": "onnxruntime",
            "variant": "onnx-int8",
            "source_release": release_id,
            "mlflow_model": "mena-mlops-arabert",
            "mlflow_version": mlflow_version,
            "files": files,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

gcloud storage cp \
  "${work_dir}/model-int8.onnx" \
  "${work_dir}/tokenizer.json" \
  "${work_dir}/tokenizer_config.json" \
  "${work_dir}/manifest.json" \
  "${public_uri}/"

manifest_url="https://storage.googleapis.com/${public_uri#gs://}/manifest.json"
curl --fail --silent --show-error "$manifest_url" >/dev/null
echo "Verified public manifest: ${manifest_url}"

if [[ "$REMOVE_LEGACY" == "true" ]]; then
  gcloud storage rm --recursive "${PUBLIC_BUCKET}/models/onnx-int8-v1/"
  echo "Removed legacy public bundle: ${PUBLIC_BUCKET}/models/onnx-int8-v1/"
fi
