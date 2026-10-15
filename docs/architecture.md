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
    F --> U[Streamlit reviewer]
    U --> Q[Retraining thresholds]
    Q --> G
    G -->|yes| V[Vertex GPU retraining]
    G -->|no| X[Skip retraining]
    V --> E[Held-out test evaluation]
    E --> O[ONNX FP32 and INT8 optimization]
    O --> QG[Quality and optimization gates]
    QG -->|approved artifact| N
    QG -->|rejected| X
```

The same model artifact supports online HTTP inference, batch inference, and
Redis Streams inference. Stable/candidate routing is operationally separate
from model training, which makes rollback a router change rather than a model
mutation.
