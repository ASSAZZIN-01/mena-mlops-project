# Optimization benchmark

The benchmark uses the same `models/arabert-debug` artifact and the same
1,225-row test split for every representation. The PyTorch model remains the
canonical artifact; ONNX files are deployment variants generated from it.

## Offline quality and single-request CPU latency

| Variant | Accuracy | Macro-F1 | Neutral F1 | Mean latency | P95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch | 0.8196 | 0.5523 | 0.0000 | 107.75 ms | 214.52 ms |
| ONNX FP32 | 0.8196 | 0.5523 | 0.0000 | 65.68 ms | 99.57 ms |
| ONNX INT8 | 0.8098 | 0.5462 | 0.0000 | 16.96 ms | 23.82 ms |

ONNX FP32 is quality-equivalent to PyTorch and reduces measured mean latency
by approximately 39%. ONNX INT8 reduces mean latency by approximately 84%, but
its macro-F1 is below the current quality baseline (`0.5523`), so it is not
eligible for promotion under the existing quality gate.

## Deployment decision

ONNX FP32 is the selected optimization candidate for the next canary
comparison. INT8 remains a reproducible artifact for further calibration or
quantization-aware training, but it must not replace the canonical model or be
promoted without recovering the macro-F1 regression.

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
