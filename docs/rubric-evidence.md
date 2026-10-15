# Rubric evidence map

Use this page as the final review checklist. Each row should eventually have a
command output, UI screenshot, or CI link attached to the course submission.

| Area | Implementation | Verification still needed |
| --- | --- | --- |
| Versioned data | DVC and GCS remote in `.dvc/` | `dvc pull`, `dvc repro`, and remote status |
| Reproducible training | `src/mena_mlops/training/` and `configs/training.yaml` | MLflow run with config, metrics, and data summary |
| Oversampling | Training-only oversampling in `training/data.py` | Class counts before/after and untouched validation/test counts |
| Experiment tracking | Compose MLflow server and GCS artifact root | MLflow run screenshot plus GCS artifact listing |
| Model registry | MLflow registered model `mena-mlops-arabert` | Registered version and source run screenshot |
| Optimization | `build_model_variants.py` and `evaluate_model_variants.py` | FP32/INT8 quality and latency comparison |
| Serving | BentoML stable and candidate services | Gateway health and prediction response |
| Batch inference | `run_batch_inference.py` and dashboard upload | CSV result with generated ID and optional source behavior |
| Canary deployment | Nginx weights and `scripts/canary.py` | 5%/20% traffic metrics and model-version counts |
| Rollback | `scripts/canary.py --rollback` | Stable-only traffic after rollback |
| Monitoring | Prometheus and Grafana provisioning | Request, latency, error, distribution, and confidence panels |
| Drift detection | Evidently/PSI report and Airflow DAG | Successful Airflow task and generated report |
| Feedback loop | Redis consumer and Streamlit reviewer | Automatic uncertain example, review, and deferred status |
| Auditability | `data/audit/predictions.db` | Live, batch, and stream provenance records |
| Retraining gate | `configs/feedback.yaml` and `submit_vertex_retraining.py` | Threshold check and candidate run when thresholds are met |
| CI/CD | `.github/workflows/ci.yml` | Passing visible jobs and Compose smoke job |
| Load testing | Locust scenario and reports | Final run through Nginx with throughput, p95, and failures |

## Screenshot naming

Use stable names so evidence can be matched to the report:

```text
01-dashboard-live.png
02-dashboard-batch.csv
03-feedback-review.png
04-mlflow-run.png
05-grafana-canary.png
06-airflow-success.png
07-canary-rollback.png
08-locust-summary.html
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
