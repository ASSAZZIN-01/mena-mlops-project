# Rubric evidence map

Use this page as the final review checklist. Each row should eventually have a
command output, UI screenshot, or CI link attached to the course submission.

| Area | Implementation | Verification still needed |
| --- | --- | --- |
| Versioned data | DVC and GCS remote in `.dvc/` | `dvc pull`, `dvc repro`, and remote status |
| Reproducible training | `src/mena_mlops/training/` and `configs/training.yaml` | MLflow run with config, metrics, and data summary |
| Oversampling | Training-only oversampling in `training/data.py` | Class counts before/after and untouched validation/test counts |
| Experiment tracking | Compose MLflow server and GCS artifact root | MLflow run screenshot plus GCS artifact listing |
| Model registry | MLflow registered model `mena-mlops-arabert` | Registered version and source run screenshot; rejected distillation candidates remain unregistered |
| Optimization | `build_model_variants.py` and `evaluate_model_variants.py` | FP32/INT8 quality and latency comparison |
| Serving | BentoML stable and candidate services | Gateway health and prediction response |
| Batch inference | `run_batch_inference.py` and dashboard upload | CSV result with generated ID and optional source behavior |
| Canary deployment | Nginx weights and `scripts/canary.py` | 5%/20% traffic metrics and model-version counts |
| Rollback | `scripts/canary.py --rollback` | Stable-only traffic after rollback |
| Monitoring | Prometheus/Grafana provisioning and PSI Pushgateway publishing | Request, latency, error, distribution, confidence, and `mena_drift_psi_max` panels |
| Drift detection | Evidently/PSI report and Airflow DAG | Successful Airflow task and generated report |
| Feedback loop | Redis consumer and browser reviewer at `/reviewer` | Automatic uncertain example, confirmed Correct/Wrong/I don't know review, revision without duplicates, and deferred status |
| Auditability | `data/audit/predictions.db` | Live, batch, and stream provenance records |
| Retraining gate | `configs/feedback.yaml`, dual feedback cursors, and `submit_vertex_retraining.py` | Threshold check, attempt deduplication, successful-deployment progress, and candidate run when thresholds are met |
| Release observability | Control Center, release registry, and deployed baseline | Release history, gate failure reasons, `ON PROD` state, model identity, p50/p95 latency, and operational links |
| CI/CD | `.github/workflows/ci.yml` | Passing visible jobs, Compose smoke job, and registry-delivery configuration |
| Load testing | Locust scalar and batch scenarios | Final run through Nginx with throughput, p95, and failures |

## Current implementation evidence

The following requirements are implemented and covered by code/tests; the
remaining UI captures are intentionally deferred until the final joint
evidence session:

- Public gateway `/health` returns JSON with `status: healthy`.
- Compose model readiness continues to use Bento's `/healthz` liveness route.
- Drift writes `psi.prom`, publishes the same exposition to Pushgateway, and
  Prometheus scrapes the Pushgateway job.
- BentoML includes both explicit vectorized batching and an adaptive
  `batchable=True` API with bounded batch size/latency.
- The six-layer student model distilled from the twelve-layer AraBERT teacher
  was trained successfully on Vertex AI and evaluated through the project
  optimization and promotion gates. The student ONNX INT8 representation
  passed the optimization gate but the student quality gate rejected it
  (macro-F1 `0.5246`, neutral F1 `0.0000`), so it was not promoted.
- The Vertex distillation run records the train, validation, and test splits as
  MLflow dataset inputs. Each input includes its GCS staging URI, DVC path,
  full dataset SHA-256, DVC lock checksum, and source commit. The rejected
  candidate remains an MLflow run/artifact, not a registered deployable model.
- TensorRT FP16 remains an explicitly isolated GPU comparison path. No
  `trtexec` environment was available for this run, so TensorRT evidence is
  still pending and no synthetic measurements are reported.
- The required peer-review text is in `docs/peer-review.md`.
- README contains the exact three-command reviewer quickstart, final
  architecture diagram, and session changelog.
- The live dashboard evidence is captured in
  `docs/evidence/screenshots/01-dashboard-live.png`; it shows a successful
  production prediction, confidence/probabilities, and model version.

## Screenshot naming

Use stable names so evidence can be matched to the report:

```text
01-dashboard-live.png
02-dashboard-batch.png
03-feedback-review.png
04-mlflow-experiment-overview.png
05-mlflow-distillation-run.png
06-grafana-psi-latency.png
07-airflow-success.png
08-canary-rollout.png
09-canary-rollback.png
10-locust-summary.html
```

Screenshots should show the URL or page title where possible. Never include
credentials, service-account contents, or private tokens.

## Current known limitations

- Airflow availability and DAG visibility are verified; a successful scheduled
  task instance still needs to be captured.
- Candidate deployment and optimization are verified locally; the progressive
  canary and rollback evidence still needs to be captured.
- The neutral class remains weaker than negative and positive classes and
  should be discussed rather than hidden.
