# Cloud deployment guide

This guide deploys the course project with a temporary Vertex AI GPU training
job and a small Compute Engine VM running MLflow. The GPU worker exists only
for the duration of a training job; the MLflow VM is a separate low-cost
resource.

## 1. Required Google Cloud resources

Use one Google Cloud project and region consistently:

- Project: `mena-mlops-project`
- Region: `us-central1`
- Vertex accelerator: `NVIDIA_TESLA_T4`

The setup uses two different compute resources:

- A small Compute Engine VM that stays available for MLflow.
- A temporary Vertex AI worker that is created only for retraining and
  released after the job reaches a terminal state.

Enable these APIs:

- Agent Platform API (`aiplatform.googleapis.com`)
- Artifact Registry API (`artifactregistry.googleapis.com`)
- Cloud Storage API (`storage.googleapis.com`)
- Cloud Logging API (`logging.googleapis.com`)
- Compute Engine API (`compute.googleapis.com`)

The service account used by Airflow and staging is:

```text
mena-mlops-dvc@mena-mlops-project.iam.gserviceaccount.com
```

Grant it:

- Agent Platform User (`roles/aiplatform.user`)
- Service Usage Consumer (`roles/serviceusage.serviceUsageConsumer`)
- Storage Object Admin (`roles/storage.objectAdmin`)
- Artifact Registry Reader (`roles/artifactregistry.reader`)
- Logs Writer (`roles/logging.logWriter`)

## 2. Enable APIs and grant IAM

In Google Cloud Console, select the `mena-mlops-project` project and enable:

1. **Agent Platform API** (`aiplatform.googleapis.com`). Google may display
   this as Agent Platform even though the API service name is unchanged.
2. **Artifact Registry API** (`artifactregistry.googleapis.com`).
3. **Cloud Storage API** (`storage.googleapis.com`).
4. **Cloud Logging API** (`logging.googleapis.com`).
5. **Compute Engine API** (`compute.googleapis.com`).

Grant the following roles to:

```text
mena-mlops-dvc@mena-mlops-project.iam.gserviceaccount.com
```

- Agent Platform User (`roles/aiplatform.user`)
- Service Usage Consumer (`roles/serviceusage.serviceUsageConsumer`)
- Storage Object Admin (`roles/storage.objectAdmin`)
- Artifact Registry Reader (`roles/artifactregistry.reader`)
- Logs Writer (`roles/logging.logWriter`)

The renamed **Agent Platform User** role must include
`aiplatform.locations.list`. This is the role needed to submit a Custom Job.

## 3. Create the Artifact Registry training image

Create a Docker repository named `mena-mlops` in `us-central1`. The training
image is published at:

```text
us-central1-docker.pkg.dev/mena-mlops-project/mena-mlops/training:latest
```

To rebuild and publish it from the project workstation:

```bash
docker build \
  -f deploy/training/Dockerfile \
  -t us-central1-docker.pkg.dev/mena-mlops-project/mena-mlops/training:latest \
  .
docker push \
  us-central1-docker.pkg.dev/mena-mlops-project/mena-mlops/training:latest
```

Authenticate Docker using the configured Google Cloud identity before pushing.
Never commit a service-account key or print an access token.

## 4. MLflow Compute Engine VM

Create a small VM in `us-central1`, for example:

- Name: `mlflow-server`
- Machine type: `e2-medium` (2 vCPU, 4 GiB memory)
- Debian or Ubuntu
- 20 GB standard persistent boot disk

An `e2-small` VM may be too constrained for MLflow's server process.

The VM service account must have `Storage Object Admin` on the project or,
preferably, on the MLflow artifact bucket.

Connect once through the Console SSH button. Install Docker using the
distribution package:

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
```

Reconnect after changing the group, then copy or check out this repository's
`deploy/mlflow/setup_vm.sh` on the VM and run:

```bash
chmod +x deploy/mlflow/setup_vm.sh
MLFLOW_ALLOWED_HOSTS="localhost,localhost:5000,127.0.0.1,127.0.0.1:5000,mlflow-server,<VM_EXTERNAL_IP>,<VM_EXTERNAL_IP>:5000" \
MLFLOW_CORS_ALLOWED_ORIGINS="http://localhost:5000,http://127.0.0.1:5000,http://<VM_EXTERNAL_IP>:5000" \
  deploy/mlflow/setup_vm.sh
```

The script creates a persistent SQLite backend under `~/mlflow/data` and
stores artifacts in:

```text
gs://mena-mlops-project-dvc/mlflow
```

Verify the service:

```bash
curl --fail http://127.0.0.1:5000/health
docker logs --tail 50 mlflow
```

For a local browser tunnel, use:

```bash
gcloud compute ssh mlflow-server \
  --zone us-central1-a \
  -- -N -L 5000:localhost:5000
```

Then open `http://localhost:5000`. The Vertex worker must use the VM's
cloud-reachable address, not `localhost`.

The initial demonstration can allow TCP port 5000 from the internet, but this
should be replaced with HTTPS and authentication before a public deployment.
MLflow is not an authentication boundary by itself.

### Publishing a reviewer-safe serving bundle

The current production release is not automatically public. The existing
public URL used by `scripts/download_model_bundle.py` is a separate sanitized
fixture and must not be confused with the private release under
`gs://mena-mlops-project-dvc/vertex-staging/`.

Only publish an approved, immutable serving bundle after deciding that its
model files and tokenizer may be public. Do not make the DVC, MLflow, Vertex
staging, model-card, or release buckets public. The publication should use a
dedicated public bucket or prefix containing only:

- the selected ONNX/INT8 model;
- tokenizer files required by the inference image;
- a checksum manifest;
- no raw data, feedback, credentials, or private run metadata.

Before publication, verify the release decision and model-card status locally:

```bash
uv run python scripts/check_release_readiness.py \
  --decision gs://mena-mlops-project-dvc/vertex-staging/<release-id>/release/promotion_decision.json
```

Then run the repository helper from authenticated Cloud Shell. It copies only
the approved serving files, generates the checksum manifest, and tests the
exact URL from an unauthenticated request:

```bash
RELEASE_ID=clean-candidate-v21-20261018-134733 \
  MODEL_VERSION=clean-candidate-v21 \
  MLFLOW_VERSION=21 \
  bash scripts/publish_public_model_bundle.sh
```

Do not overwrite an existing public release prefix in place. After the
current production release is published, update the README and reviewer
evidence with the immutable URL and release ID. After the new manifest has
verified, the old fixture may be removed explicitly:

```bash
REMOVE_LEGACY=true bash scripts/publish_public_model_bundle.sh
```

Only use that option after the v21 manifest has returned successfully.

`MLFLOW_CORS_ALLOWED_ORIGINS` must include every browser origin used to open
the MLflow UI. Without it, the backend APIs remain reachable by `curl` and
Vertex workers, but the browser UI displays `Cross-origin request blocked`.
MLflow validates the `Host` header including the port, so the corresponding
`localhost:5000`, `127.0.0.1:5000`, and `<VM_EXTERNAL_IP>:5000` entries must
also be present in `MLFLOW_ALLOWED_HOSTS`.

The VM's ephemeral external IP can change after a stop/start. Either reserve a
static external IP or update `MLFLOW_TRACKING_URI` whenever the address
changes. Verify the address from the local workstation, not from inside the VM:

```bash
curl --fail http://<VM_EXTERNAL_IP>:5000/health
```

## 5. Fill in `.env` step by step

From the project workstation:

```bash
cp .env.example .env
```

Keep `.env` local; it is ignored by Git. Fill in the values as follows:

```dotenv
VERTEX_PROJECT_ID=mena-mlops-project
VERTEX_REGION=us-central1
VERTEX_MACHINE_TYPE=n1-standard-4
VERTEX_ACCELERATOR_TYPE=NVIDIA_TESLA_T4
VERTEX_TRAINING_IMAGE_URI=us-central1-docker.pkg.dev/mena-mlops-project/mena-mlops/training:latest
VERTEX_STAGING_BUCKET=gs://mena-mlops-project-dvc/vertex-staging
MLFLOW_TRACKING_URI=http://<VM_EXTERNAL_IP>:5000
MLFLOW_ARTIFACT_LOCATION=gs://mena-mlops-project-dvc/mlflow
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
```

Use these sources for each value:

| Variable | Value |
|---|---|
| `GOOGLE_APPLICATION_CREDENTIALS` | Absolute path to the local service-account JSON. The VM does not need this key. |
| `VERTEX_PROJECT_ID` | The project selector value in Google Cloud Console. |
| `VERTEX_REGION` | `us-central1`, matching the image repository and quota region. |
| `VERTEX_MACHINE_TYPE` | `n1-standard-4` for the temporary Vertex worker. |
| `VERTEX_ACCELERATOR_TYPE` | `NVIDIA_TESLA_T4` after T4 training quota approval. |
| `VERTEX_TRAINING_IMAGE_URI` | The Artifact Registry image URI from Section 3. |
| `VERTEX_STAGING_BUCKET` | A `gs://` prefix in the project data bucket. |
| `MLFLOW_TRACKING_URI` | `http://<current-MLflow-VM-IP>:5000`, never `localhost`. |
| `MLFLOW_ARTIFACT_LOCATION` | `gs://mena-mlops-project-dvc/mlflow`. |

Do not paste service-account keys, Kaggle tokens, Hugging Face tokens, or
`.env` contents into issues, chat, screenshots, or Git.

Verify the non-secret values without printing them:

```bash
for key in VERTEX_PROJECT_ID VERTEX_REGION VERTEX_TRAINING_IMAGE_URI \
  VERTEX_STAGING_BUCKET MLFLOW_TRACKING_URI; do
  grep -q "^${key}=" .env && echo "$key: present" || echo "$key: missing"
done
```

Verify the MLflow endpoint:

```bash
curl --fail "$MLFLOW_TRACKING_URI/health"
```

Start Airflow and verify the DAG:

```bash
docker compose up -d airflow
curl --fail http://localhost:8081/health
```

The DAG runs drift reporting first and then evaluates two independent
retraining triggers:

1. Evidently reports drift or the maximum PSI exceeds `0.25`.
2. Reviewed feedback reaches every class threshold in
   `configs/feedback.yaml`.

The Vertex job is submitted when **either** trigger is true. The submission
script receives the generated `drift_summary.json` through `--drift-summary`,
so a drift-only job does not wait for feedback thresholds. If neither trigger
is active, the task exits safely without creating a GPU job.

The feedback database maintains two cursors. The last-attempt cursor advances
after a training attempt and prevents unchanged feedback from launching the
same job repeatedly after a rejected candidate. The last-successful-deployment
cursor advances only after deployment and canary verification succeed; it
drives the user-facing count of feedback not yet represented in production.
Historical rows remain available for audit, and a stable drift summary is
fingerprinted to avoid resubmitting the same drift event.

## 6. Vertex quota and job checks

Before submitting a GPU job, check **Quotas & System Limits** for:

```text
Service: Agent Platform API
Quota: Custom model training Nvidia T4 GPUs per region
Region: us-central1
Required limit: 1
```

An **Enabled** quota-adjustment configuration does not mean the effective
quota is already `1`. If the current limit is `0`, the Custom Job returns
`RESOURCE_EXHAUSTED` before any machine starts. Submit a quota increase and
wait for approval.

Before enabling automatic retraining, verify:

1. T4 quota is available in `us-central1`.
2. `VERTEX_TRAINING_IMAGE_URI` exists in Artifact Registry.
3. The VM MLflow endpoint is reachable from the Vertex worker.
4. The MLflow run appears in the remote server.
5. GCS contains the staged data and MLflow artifacts.
6. The Vertex job reaches `JOB_STATE_SUCCEEDED` and terminates.

Stop the MLflow VM when it is not needed. Its persistent disk remains available
for the next session. Delete the VM and disk only when the project is finished.

## 7. Credential hygiene

`.env` and service-account JSON files are ignored locally and must never be
committed. Tokens pasted into chats, attachments, terminals, or screenshots
must be revoked and regenerated even if they were never committed to Git.
