# Final architecture

```mermaid
flowchart LR
    DVC[DVC datasets] --> T[Training and evaluation]
    T --> M[MLflow tracking and registry]
    M --> S[BentoML stable model]
    M --> C[BentoML candidate model]
    S --> N[Nginx weighted canary router]
    C --> N
    N --> API[Online HTTP inference]
    N --> L[Locust load tests]
    R[Redis Streams] --> W[Inference consumer]
    W --> O[Prediction stream]
    W --> F[Low-confidence feedback]
    API --> P[Prometheus]
    S --> P
    C --> P
    P --> G[Grafana and alerts]
    O --> E[Monitoring windows]
    E --> A[Airflow scheduled Evidently and PSI]
    A --> G{Drift OR feedback thresholds?}
    F --> U[Control Center reviewer]
    U --> Q[Retraining thresholds]
    Q --> G
    G -->|yes| V[Vertex GPU retraining]
    G -->|no| X[Skip retraining]
    V --> E[Held-out test evaluation]
    E --> O[ONNX FP32 and INT8 optimization]
    O --> QG[Quality and optimization gates]
    QG -->|approved artifact| N
    QG -->|rejected| X
    N --> R[Release registry and production baseline]
    R --> CC[MLOps Control Center]
    CC --> U
```

The same model artifact supports online HTTP inference, batch inference, and
Redis Streams inference. Stable/candidate routing is operationally separate
from model training, which makes rollback a router change rather than a model
mutation. The Control Center is the operator-facing browser UI for release
history, gate evidence, deployment state, model identity, feedback readiness,
and the human reviewer queue. Durable lifecycle records are stored in
`reports/control-plane/releases.db`; the active serving baseline is stored in
`reports/retraining/deployed_baseline.json`.
