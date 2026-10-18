# Deployment and monitoring

The deployment work uses two independently started instances of the same
BentoML service. Each instance loads one immutable model directory and exposes
its `MODEL_VERSION` through `/health` and `/predict`. Prometheus scrapes each
instance separately so stable and candidate traffic can be compared.

`MODEL_VERSION` is the immutable serving-bundle identity (for example,
`onnx-int8-v1`). It is intentionally separate from MLflow Registry versions:
`mena-mlops-arabert` version numbers identify registered model records and may
have several versions for the same serving family. The dashboard displays the
serving identity returned by the live gateway, not an MLflow registry number.

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

The local React dashboard runs on port 8501 and calls the gateway on port
8080. The Nginx gateway therefore handles `OPTIONS` preflight requests and
allows the dashboard origin for `GET` and `POST` inference calls.

Successful vectorized requests expose
`mena_model_batch_requests_total` and
`mena_model_batch_latency_seconds`, with model-version and batch-size labels.
The gateway also emits a `batch_prediction_completed` event containing batch
size and elapsed milliseconds.

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

Grafana distinguishes live request latency from batch latency. The live
`/predict` endpoint handles one prediction per request, while `/predict_batch`
handles multiple predictions per request. The dashboard therefore reports
live p50/p95 request latency separately, shows whole-chunk batch latency, and
reports effective batch latency per prediction using
`mena_model_batch_row_latency_seconds`. Throughput is available in Prometheus
for deeper performance analysis, but is intentionally not shown in the
default dashboard because sparse batch traffic made the visualization
misleading.

The Grafana **Confidence Stats Per Class** table reports minimum, average, and
maximum confidence for predictions whose argmax class is the listed class.
These values are not class probabilities across every prediction. All three
statistics use the selected Prometheus time range: average uses histogram
sum/count increases, while minimum and maximum use the observed extrema gauge
history over that range. This keeps the columns comparable across serving
process restarts.

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

Offline scoring accepts CSV, JSONL, or Parquet files containing a required
`text` column. Input fields are preserved and prediction metadata is appended.
The dashboard and BentoML service use vectorized batches of up to 128 texts.
Dashboard uploads are `audit_only` by default; enable reviewable feedback only
when the dataset is intentionally meant for human annotation:

```bash
uv run python scripts/run_batch_inference.py \
  data/input/reviews.jsonl \
  data/predictions/reviews.jsonl \
  --model-path models/arabert-debug \
  --model-version debug-v1
```

Each output record contains `prediction`, JSON-encoded `probabilities`,
`model_version`, and an ISO-8601 `predicted_at` timestamp.

Run the scalar and vectorized serving benchmarks separately:

```bash
bash scripts/run_load_test.sh 10 10 30s reports/locust/scalar scalar
LOCUST_BATCH_SIZE=32 bash scripts/run_load_test.sh \
  10 10 30s reports/locust/batch-32 batch
```

The batch load test sends `feedback_policy=audit_only`, so benchmark traffic
is audited without creating reviewer items. Use batch sizes 32, 64, and 128
to compare throughput and p95 tail latency; the benchmark report records
request latency, while throughput must be multiplied by the batch size.

## Redis Streams inference

The Compose stack includes Redis on port `6379`. The streaming consumer reads
review events from `reviews:input`, publishes predictions to
`reviews:predictions`, and acknowledges an input only after the prediction has
been written successfully. It is launched automatically by `docker compose up`
and persists low-confidence predictions in `data/feedback/reviewed.db`.

The serving services persist every live and batch prediction in
`data/audit/predictions.db`. Each audit record includes the source, optional
input ID, text, prediction, confidence, probabilities, model version, and
whether human review is required.

Each synchronous prediction also emits a structured JSON event without
including the raw text. The event contains a request ID (from `X-Request-ID`
when supplied), source, input length, label, confidence, model version,
deployment stage, and latency. Prometheus exposes gateway-level
`mena_gateway_predictions_total`, `mena_gateway_prediction_confidence`,
`mena_model_predictions_total`, and `mena_model_prediction_confidence`
metrics; source and deployment stage are bounded labels and must not contain
user text or IDs.

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

Vertex retraining now performs the release decision before the attempt cursor
is committed. It evaluates the held-out test set, exports ONNX FP32
and dynamic-INT8 artifacts, benchmarks both optimized variants against the
PyTorch candidate, and selects the fastest variant that stays within the
configured macro-F1 loss limit. The selected variant is then checked against
the stable test macro-F1 and per-class floors in `configs/promotion.yaml`.
The job publishes the variants, checksums, and `promotion_decision.json` below
its Vertex staging prefix. Only an `approved` decision is production-ready.

The local Nginx canary remains an explicit deployment operation rather than an
implicit side effect of retraining. This keeps a rejected candidate from
changing production traffic and allows health, prediction-version, error-rate,
and latency checks before increasing the canary percentage.

Retraining feedback uses two independent cursors. The last-attempt cursor
prevents unchanged feedback from launching another job after a rejected
candidate. The last-successful-deployment cursor advances only after
deployment and canary verification succeed, and drives the Control Center
count of reviewed feedback that is not yet represented in production.

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

For Airflow, the Compose image includes the monitoring package and copies the
DAG from `dags/evidently_drift.py` into Airflow's DAG directory. The default
schedule runs every six hours (`0 */6 * * *`), and the DAG accepts manual
triggers from the Airflow UI or CLI:

```bash
docker compose exec airflow airflow dags trigger mena_evidently_drift
```

Only one DAG run is active at a time. Wait for the active run to finish before
manually triggering another retraining; this prevents duplicate Vertex jobs.
The schedule can be overridden with `MENA_DRIFT_SCHEDULE`. The DAG accepts
these environment variables:

- `MENA_PROJECT_ROOT`
- `MENA_DRIFT_REFERENCE`
- `MENA_DRIFT_CURRENT`
- `MENA_DRIFT_OUTPUT`
- `MENA_DRIFT_SCHEDULE` (default: every six hours)
- `VERTEX_PROJECT_ID`
- `VERTEX_REGION` (default: `us-central1`)
- `VERTEX_MACHINE_TYPE` (default: `n1-standard-4`)
- `VERTEX_ACCELERATOR_TYPE` (default: `NVIDIA_TESLA_T4`)
- `VERTEX_TRAINING_IMAGE_URI`
- `VERTEX_STAGING_BUCKET`
- `MLFLOW_TRACKING_URI` (must be reachable from Vertex AI)

## Human feedback and retraining thresholds

Uncertain predictions can be persisted with `FeedbackStore` in
`src/mena_mlops/feedback.py`. The default confidence window and review
thresholds are configured in `configs/feedback.yaml`:

- confidence between 25% and 75%;
- 100 validated negative examples;
- 20 validated neutral examples;
- 100 validated positive examples.

Incorrect reviews require the reviewer to select the corrected class. The
reviewed records remain separate from validation and test data and can be
merged into the training split only after all configured thresholds are met.
For the local demonstration, `configs/feedback.yaml` uses temporary `5/1/5`
thresholds; restore the production `100/20/100` values afterward.

The reviewer interface is served by the MLOps Control Center:

[`http://localhost:8503/reviewer`](http://localhost:8503/reviewer)

The browser reviewer provides pending and reviewed queues with Previous/Next
navigation. Completed items are read-only until **Fix it** is selected; edits
update the existing feedback row rather than creating duplicates. Correct,
Wrong, and I don't know decisions require explicit confirmation. Wrong
requires a corrected class, while I don't know is deferred and excluded from
retraining thresholds.

The Control Center also exposes release gate evidence, exact rejection
reasons, the active `ON PROD` baseline, model identity, feedback readiness,
and links to Vertex AI, MLflow, Airflow, Grafana, and Prometheus. It displays
p50, mean, and p95 latency when available. Older baselines without p50 data
are shown as `Not recorded`, not inferred.

The Redis inference consumer automatically adds predictions whose maximum
class probability falls within the configured confidence window to this
feedback database.

Check whether enough reviewed data exists to start retraining:

```bash
uv run python scripts/check_retraining_gate.py
```

In the Compose deployment, Airflow runs the drift report first and then calls
`scripts/submit_vertex_retraining.py` with the generated
`drift_summary.json`. The submitter uses an OR gate:

- Evidently reports drift or PSI exceeds `0.25`; or
- all reviewed-feedback thresholds are reached.

It skips safely when neither trigger is ready. When either trigger is ready,
it stages the processed splits and reviewed feedback in
`VERTEX_STAGING_BUCKET`, submits a temporary Vertex AI Custom Job, and waits
for its terminal state. A drift-only trigger is supported; the reviewed
feedback append can be empty in that case. The job uses the configured GPU
worker, downloads the staged data, runs `scripts/vertex_train.py`, and
terminates after training. The training container logs the run through the
cloud-reachable `MLFLOW_TRACKING_URI` and configured GCS artifact location.

Reviewed feedback is not deleted after retraining. The feedback database stores
two durable cursors: the last completed retraining attempt, which prevents an
unchanged batch from retriggering every schedule, and the last successful
deployment, which identifies feedback not yet represented in production.
Automatic trigger thresholds use the attempt cursor; the production gap uses
the deployment cursor. The deployment cursor advances only after the candidate
passes its gates, deployment completes, and canary verification succeeds.
If training or deployment fails, production remains unchanged and the feedback
gap remains visible for the next release attempt.
Airflow also limits the DAG to one active run so overlapping schedules cannot
submit the same batch concurrently.

The local MLflow URL (`http://localhost:5000`) cannot be used directly by a
Vertex worker. To follow the same cloud MLflow server locally, expose it
through an SSH tunnel or another secure port-forward to local port 5000.

Candidate promotion must also pass the verified-label quality gate in
`configs/promotion.yaml`. It requires a minimum macro-F1, improvement over
the stable model, and minimum F1 floors for each sentiment class.

When the feedback thresholds are reached, launch candidate retraining manually
outside Airflow:

```bash
uv run python scripts/submit_vertex_training.py \
  --job-prefix manual-training
```

After evaluating the stable and candidate reports on trusted labels, promote
only through the quality gate:

```bash
uv run python scripts/promote_candidate.py \
  --stable reports/quality/stable.json \
  --candidate reports/quality/candidate.json \
  --candidate-percent 20
```
