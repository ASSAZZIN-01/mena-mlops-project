# Deployment and monitoring

The deployment work uses two independently started instances of the same
FastAPI service. Each instance loads one immutable model directory and exposes
its `MODEL_VERSION` through `/health`, `/predict`, and Prometheus labels.

## Run one local instance

The trained model must be present locally, either from the MLflow artifact or
from the training output directory:

```bash
MODEL_PATH=models/arabert-debug \
MODEL_VERSION=baseline-v1 \
uv run python scripts/serve.py
```

The API exposes:

- `GET /health` for readiness checks.
- `POST /predict` with `{"text": "..."}` for inference.
- `GET /metrics` for Prometheus scraping.

## Release strategy

The next deployment slice will run a stable and a candidate instance behind a
traffic router. The router starts with a small candidate percentage, such as
5%, while the candidate's error rate, latency, health, and model-version
traffic are observed. Promotion changes the router configuration only after a
quality gate passes.

Rollback is deliberately boring: set candidate traffic to `0%`, restore the
previous stable route, and keep the candidate artifacts and metrics for
diagnosis. The model registry alias and router configuration must change
together; a version number embedded in application code is not a rollback
mechanism.

No production promotion is enabled by this branch yet. The first milestone is
to make the health, prediction, version, and metrics contracts testable before
adding Docker, Nginx, and dashboard automation.
