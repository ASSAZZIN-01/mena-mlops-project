# Optimization benchmark

The benchmark uses the same `models/arabert-debug` artifact and the same
1,225-row test split for every representation. The PyTorch model remains the
canonical artifact; ONNX files are deployment variants generated from it.

## Production deployment

The Compose deployment now defaults to the selected ONNX INT8 artifact as the
stable production service:

```text
stable   = models/arabert-debug-variants/model-int8.onnx
candidate = independently configured ONNX model version
rollback = models/arabert-debug (canonical PyTorch model)
```

Stable and candidate are model lifecycle slots, not permanently different
frameworks. A future candidate can be tested and promoted without changing the
production routing contract. To roll back to PyTorch, set
`STABLE_MODEL_BACKEND=pytorch` and `STABLE_MODEL_PATH=./models/arabert-debug`.

## Offline quality and single-request CPU latency

| Variant | Accuracy | Macro-F1 | Neutral F1 | Mean latency | P95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch | 0.8196 | 0.5523 | 0.0000 | 107.75 ms | 214.52 ms |
| ONNX FP32 | 0.8196 | 0.5523 | 0.0000 | 65.68 ms | 99.57 ms |
| ONNX INT8 | 0.8098 | 0.5462 | 0.0000 | 16.96 ms | 23.82 ms |

ONNX FP32 is quality-equivalent to PyTorch and reduces measured mean latency
by approximately 39%. ONNX INT8 reduces mean latency by approximately 84%;
its macro-F1 drop is approximately 1.11% relative to the baseline, which is
within the configured 5% optimization tolerance.

## Deployment decision

ONNX INT8 is the selected optimization candidate for the next canary
comparison. Promotion uses the canonical PyTorch model as the baseline and
allows a macro-F1 drop strictly below 5% when the variant improves mean
latency. Eligible variants are ranked by mean latency and then p95 latency;
the fastest eligible variant is selected. The PyTorch model remains the
canonical training artifact, and INT8 is promoted only as a deployment
representation.

The artifacts and metrics are generated with:

```bash
uv run python scripts/build_model_variants.py \
  --model-path models/arabert-debug \
  --output-dir models/arabert-debug-variants
uv run python scripts/evaluate_model_variants.py \
  --model-path models/arabert-debug \
  --variants-dir models/arabert-debug-variants \
  --test-data data/processed/test.parquet \
  --output reports/optimization-comparison.json
```

Locust load tests use the same `SentimentGatewayUser` scenario and validate
the response contract in addition to HTTP status. They should be run against
the selected backend with `LOCUST_HOST` set to the service URL after the
deployment topology is selected.

## System-level serving benchmark

The same 10-user, 30-second Locust scenario was run against each backend with
the response contract enabled:

| Backend | Requests | Failures | Throughput | Prediction median | Prediction p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch stable | 403 | 17 (4.22%) | 13.52 req/s | 330 ms | 510 ms |
| ONNX INT8 stable | 830 | 6 (0.72%) | 27.92 req/s | 14 ms | 28 ms |

The ONNX INT8 run delivered approximately 2.06x the observed prediction
throughput and approximately 95.8% lower prediction median latency. The
non-zero failures occurred as connection resets during the local single-worker
stress run; repeat the benchmark behind the full Compose/Nginx topology before
using these figures as a production SLO.
