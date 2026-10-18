# Reviewer walkthrough

This page is the shortest reproducibility path for a reviewer. It explains
what can be launched without private credentials, where the current release is
recorded, and which optional cloud workflows require access to the course
project.

## 1. Launch the public serving stack

From a clean checkout with Docker and `uv` installed:

```bash
uv sync --locked
python scripts/download_model_bundle.py
docker compose up --build -d
```

The bundle is the immutable `clean-candidate-v21` ONNX INT8 serving release.
The download script verifies the SHA-256 manifest before Compose starts. The
public bundle contains only the model and tokenizer files; datasets, feedback,
MLflow artifacts, and credentials remain private.

Verify the gateway:

```bash
curl --fail http://localhost:8080/health
curl --fail -X POST http://localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"هذا المنتج ممتاز","source":"live","input_id":"reviewer-1"}'
```

Open the local interfaces:

| Interface | URL |
| --- | --- |
| Reviewer dashboard | http://localhost:8501 |
| Control Center | http://localhost:8503 |
| Feedback reviewer | http://localhost:8503/reviewer |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| Local MLflow fallback | http://localhost:5000 |
| Airflow | http://localhost:8081 |

The Control Center reports local Compose MLflow and, when configured, the
remote GCP MLflow endpoint separately. The course training runs and registry
version 21 are hosted at:

```text
http://34.58.163.106:5000
```

## 2. Reproduce the quality and serving checks

Run the repository checks:

```bash
make check
```

Inspect the current release evidence:

- `docs/evidence/screenshots/11-control-center-clean-release.png`
- `docs/evidence/screenshots/12-terminal-feedback-exclusions.png`
- `docs/evidence/screenshots/17-terminal-promotion-gates.png`
- `docs/evidence/screenshots/19-terminal-production-inference.png`
- `docs/evidence/screenshots/21-terminal-final-tests.png`
- `docs/evidence/screenshots/22-terminal-canary-rollout.png`
- `docs/evidence/screenshots/10-locust-summary.html`

The current production release is `clean-candidate-v21-20261018-134733`,
serves the `onnx-int8` variant, and is linked to MLflow registered model
`mena-mlops-arabert` version `21`.

## 3. Review the lifecycle interfaces

The Control Center provides:

- release history and exact gate outcomes;
- current `ON PROD` model identity and linked release URI;
- local and remote MLflow health;
- feedback readiness and protected-row counts;
- links to the configured MLflow, Airflow, Grafana, and Prometheus services.

The Feedback Reviewer demonstrates human review while protected overlaps with
train, validation, and test splits remain quarantined and excluded from
retraining exports. The release registry preserves rejected candidates and
successful deployment history.

## 4. Optional cloud workflows

DVC reproduction, Vertex retraining, private MLflow artifacts, and Airflow
automatic retraining require course-project credentials and should not be
attempted with credentials copied into the repository. Follow
[`docs/cloud-deployment.md`](cloud-deployment.md) for the GCP setup and
[`docs/operations-runbook.md`](operations-runbook.md) for retraining,
promotion, canary, rollback, monitoring, and feedback operations.

Never make the DVC, MLflow artifact, Vertex staging, feedback, or model-card
locations public. Only the dedicated sanitized serving bundle is public.
