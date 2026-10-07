# Deployment and monitoring

The deployment work uses two independently started instances of the same
BentoML service. Each instance loads one immutable model directory and exposes
its `MODEL_VERSION` through `/health` and `/predict`. Prometheus scrapes each
instance separately so stable and candidate traffic can be compared.

## Run one local instance

The trained model must be present locally, either from the MLflow artifact or
from the training output directory:

```bash
MODEL_PATH=models/arabert-debug \
MODEL_VERSION=baseline-v1 \
uv run bentoml serve \
  mena_mlops.serving.bento_service:SentimentService \
  --host 0.0.0.0 --port 8001
```

The public gateway exposes:

- `GET /health` for readiness checks (mapped to BentoML's `/healthz`).
- `POST /predict` with `{"text": "..."}` for inference.
- `GET /metrics` for Prometheus scraping.

BentoML also exposes its service API directly inside the containers. The
custom `health` method is a POST endpoint; the gateway intentionally uses
BentoML's standard GET `/healthz` endpoint for load-balancer readiness.

## Start the local deployment stack

Docker Desktop with WSL integration is required:

```bash
docker compose up --build
```

The model services expose Docker health checks, and Nginx waits for both
models to become healthy before starting. The first startup can therefore
take a few seconds while the model artifacts load; use `docker compose ps` to
confirm both model services are healthy.

The public model endpoint is available at `http://localhost:8080`. Grafana is
available at `http://localhost:3000` and Prometheus at
`http://localhost:9090`.

The default local traffic split is 95% stable and 5% candidate. It can be
changed without changing model code:

```bash
STABLE_WEIGHT=80 CANDIDATE_WEIGHT=20 docker compose up -d nginx
```

Or use the validated canary control script:

```bash
uv run python scripts/canary.py --candidate 20
uv run python scripts/canary.py --rollback
```

## Batch inference

Offline scoring accepts JSONL or Parquet files containing a required `text`
column. Input fields are preserved and prediction metadata is appended:

```bash
uv run python scripts/run_batch_inference.py \
  data/input/reviews.jsonl \
  data/predictions/reviews.jsonl \
  --model-path models/arabert-debug \
  --model-version debug-v1
```

Each output record contains `prediction`, JSON-encoded `probabilities`,
`model_version`, and an ISO-8601 `predicted_at` timestamp.

## Redis Streams inference

The Compose stack includes Redis on port `6379`. The streaming consumer reads
review events from `reviews:input`, publishes predictions to
`reviews:predictions`, and acknowledges an input only after the prediction has
been written successfully:

```bash
REDIS_URL=redis://localhost:6379/0 \
uv run python scripts/run_stream_consumer.py
```

Publish an input event from another terminal:

```bash
redis-cli XADD reviews:input '*' text 'هذا المنتج ممتاز' event_id review-1
redis-cli XRANGE reviews:predictions - +
```

Failed messages are logged and remain pending in the consumer group so they
can be retried instead of being silently discarded.

Set `STABLE_MODEL_PATH`, `CANDIDATE_MODEL_PATH`, and the corresponding model
version variables to compare two different artifacts.

The script recreates only the gateway. Zero-weight upstreams are omitted from
the generated Nginx configuration because Nginx rejects `weight=0`. Rollback
therefore sets candidate traffic to 0% and stable traffic to 100% safely.

## Locust load testing

Start the Docker stack first, then run the benchmark through the public Nginx
gateway:

```bash
uv run bash scripts/run_load_test.sh 20 10 60s reports/locust/gateway
```

The generated CSV and HTML reports record request count, throughput, failures,
median latency, p95, and p99 latency. The scenario also validates the semantic
prediction response, so an HTTP 200 with an invalid label or probability
distribution is marked as a failure.

The first benchmark is a baseline, not an assumed capacity claim. Increase
users gradually and record the highest load that satisfies the agreed service
SLA. The course scenario is approximately 50,000 reviews per day, or 0.58
requests per second on average; peak capacity must be measured separately.

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

No production promotion is enabled by this branch yet. Promotion remains gated
until health, prediction, version, and monitoring contracts are validated.

## Evidently drift reports

The drift job derives privacy-safe features from review text and can optionally
include prediction labels and confidence values. Raw review text is never
written into Prometheus labels or the drift summary.

Provide newline-delimited JSON monitoring windows with a `text` field:

```bash
uv run python scripts/run_drift.py \
  --reference reports/monitoring/reference.jsonl \
  --current reports/monitoring/current.jsonl \
  --output-dir reports/drift
```

The job writes an HTML report for investigation, a JSON report for detailed
inspection, and `drift_summary.json` for automation. Current drift coverage
includes input and prediction drift; verified labels will be added as a
separate quality evaluation gate because they arrive later than predictions.
