# Operations runbook

This runbook is the shortest path from a clean checkout to a reproducible
experiment, deployment, canary, rollback, and evidence capture.

## 1. Configure the environment

Create a local `.env` from `.env.example` and set paths to credentials without
committing either the file or the credentials:

```bash
cp .env.example .env
export GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/gcp-service-account.json
```

The service account needs access to:

- `gs://mena-mlops-project-dvc/dvc` for DVC data;
- `gs://mena-mlops-project-dvc/mlflow` for MLflow artifacts.

The default MLflow configuration is:

```text
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_ARTIFACT_LOCATION=gs://mena-mlops-project-dvc/mlflow
```

The same variables can point training at a deployed MLflow server. Do not
replace the GCS artifact location with a local Docker volume when collecting
reproducibility evidence.

## 2. Pull data and start the stack

```bash
uv sync
uv run dvc pull
docker compose up -d --build
docker compose ps
```

Wait until both model services, MLflow, and the supporting services are
healthy. The main endpoints are:

| Component | URL |
| --- | --- |
| Gateway | http://localhost:8080 |
| Reviewer dashboard | http://localhost:8501 |
| Feedback review | http://localhost:8502 |
| Grafana | http://localhost:3000 |
| MLflow | http://localhost:5000 |
| Airflow | http://localhost:8081 |
| Prometheus | http://localhost:9090 |

Smoke-test the gateway:

```bash
curl --fail http://localhost:8080/health
curl --fail -X POST http://localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"هذا المنتج ممتاز","source":"live","input_id":"runbook-1"}'
```

## 3. Run an experiment

An experiment is intentionally independent of the feedback retraining gate.
Use the training module directly for a controlled 2-epoch oversampled
experiment:

```bash
uv run python - <<'PY'
from pathlib import Path
import yaml

config = yaml.safe_load(Path("configs/training.yaml").read_text())
config["paths"]["output_dir"] = "models/arabert-candidate-oversampled"
config["paths"]["report_dir"] = "reports/training/candidate-oversampled"
config["mlflow"]["run_name"] = "arabert-oversampled-2epoch-experiment"
Path("/tmp/training-candidate.yaml").write_text(
    yaml.safe_dump(config, sort_keys=False)
)
PY
uv run python -m mena_mlops.training.train --config /tmp/training-candidate.yaml
```

The run logs parameters, metrics, reports, and the model to the centralized
MLflow server. Verify the run in the MLflow UI and verify that its model
artifacts exist below `gs://mena-mlops-project-dvc/mlflow`.

For an on-demand full-data Vertex experiment, bypass the automatic trigger
gate explicitly:

```bash
uv run python scripts/submit_vertex_training.py \
  --job-prefix manual-training
```

This always submits a Vertex T4 job and never trains on the local GPU. The
automatic `mena_evidently_drift` DAG remains responsible for production
retraining when PSI/Evidently drift or reviewed-feedback thresholds trigger.
The manual command uses the selected training YAML unchanged, while the
automatic retraining path has its own trigger, release, and watermark logic.

## 3.1 Inference runtime selection

The inference image is separate from the training image and contains no
PyTorch dependency. It serves the published ONNX representation through CPU
ONNX Runtime by default:

```bash
ONNXRUNTIME_PACKAGE=onnxruntime INFERENCE_DEVICE=cpu \
docker compose up -d --build model-stable model-candidate feedback-consumer
```

For a GPU inference host with compatible CUDA libraries, select the GPU
runtime explicitly:

```bash
ONNXRUNTIME_PACKAGE=onnxruntime-gpu \
INFERENCE_DEVICE=cuda \
ONNX_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider \
docker compose up -d --build model-stable model-candidate feedback-consumer
```

Provider selection is validated at startup. The service fails clearly when a
requested provider is not available instead of silently falling back to an
unexpected execution backend.

## 4. Export and compare a candidate

Build deployment variants from the experiment output:

```bash
uv run python scripts/build_model_variants.py \
  --model-path models/arabert-candidate-oversampled \
  --output-dir models/arabert-candidate-oversampled-variants

uv run python scripts/evaluate_model_variants.py \
  --model-path models/arabert-candidate-oversampled \
  --variants-dir models/arabert-candidate-oversampled-variants \
  --test-data data/processed/test.parquet \
  --output reports/optimization-candidate-oversampled.json
```

Review the generated report before deployment. The selected representation
must pass the configured optimization tolerance. Candidate promotion also
requires the verified-label quality gate in `configs/promotion.yaml`; the
optimization gate and the promotion gate are separate decisions.

## 5. Deploy a candidate and run a canary

For an automatically evaluated Vertex release, inspect the decision first:

```bash
uv run python scripts/check_release_readiness.py \
  --decision gs://mena-mlops-project-dvc/vertex-staging/<job-id>/release/promotion_decision.json
```

Only an `APPROVED` result should be deployed. The guarded deployment command
downloads the approved ONNX INT8 bundle, starts the candidate slot and
Prometheus, and sends 5% of gateway traffic to it:

```bash
uv run python scripts/deploy_approved_candidate.py \
  --decision gs://mena-mlops-project-dvc/vertex-staging/<job-id>/release/promotion_decision.json
```

The command refuses rejected decisions and never changes the stable model
directory. By default it automatically progresses through 5%, 20%, 50%, and
100% candidate traffic, observing each stage for ten minutes and rolling back
to 100% stable on a failed health, error-rate, p95-latency, or prediction
contract check. Use `--manual-canary` to stop at 5% and control the split
manually.

The deployment command requires the host environment to load the project
credentials before reading a GCS release:

```bash
source /home/az-wsl/setup_env.sh
set -a
source .env
set +a
```

If Docker Desktop reports a file/directory bind-mount mismatch for Nginx or
Prometheus, recreate only the affected service; do not remove persistent
PostgreSQL or Airflow volumes:

```bash
docker compose rm -sf nginx prometheus
docker compose up -d prometheus nginx
curl --fail http://localhost:9090/-/ready
```

The canary controller removes and recreates only Nginx for each traffic split,
so model containers and their health state are not unnecessarily restarted.

Point the candidate slot at a distinct immutable artifact:

```bash
export CANDIDATE_MODEL_PATH=./models/arabert-candidate-oversampled-variants
export CANDIDATE_MODEL_VERSION=onnx-int8-oversampled-v1
export STABLE_WEIGHT=95
export CANDIDATE_WEIGHT=5
docker compose up -d model-candidate nginx
```

Verify that the response reports the expected model version:

```bash
curl --fail -X POST http://localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"هذا المنتج ممتاز","source":"live","input_id":"canary-1"}'
```

Use the canary control script for progressive traffic:

```bash
uv run python scripts/canary.py --candidate 5
uv run python scripts/canary.py --candidate 20
uv run python scripts/canary.py --candidate 50
```

After each step, inspect Grafana request totals, error rate, latency, and
model-version traffic. A candidate must not be promoted based on one
successful request.

## 6. Roll back

Rollback is a router change and does not mutate either model artifact:

```bash
uv run python scripts/canary.py --rollback
curl --fail http://localhost:8080/health
```

Capture the post-rollback health response and the Grafana panel showing that
new traffic returns to the stable version.

## 7. Evidence capture

Capture evidence only after the corresponding behavior has been verified:

1. Dashboard live prediction card.
2. Dashboard batch CSV with generated `ID` and default `source`.
3. Feedback UI with a reviewed or deferred example.
4. MLflow experiment page showing the run and metrics.
5. GCS artifact listing for the same MLflow run.
6. Grafana before/after traffic and stable/candidate version metrics.
7. Airflow DAG page and a successful task instance.
8. Prometheus readiness and gateway health.
9. Canary traffic split and rollback result.
10. Locust HTML summary and CSV output.

Use [`rubric-evidence.md`](rubric-evidence.md) to record the command, result,
and screenshot for each rubric item.
