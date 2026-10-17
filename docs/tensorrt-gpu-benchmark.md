# Optional TensorRT FP16 benchmark

Production Compose remains CPU-only and uses ONNX Runtime INT8. TensorRT is
isolated to the approved GPU optimization environment so CUDA dependencies do
not slow or change local serving.

On a GPU host with NVIDIA TensorRT and `trtexec` installed:

```bash
uv run python scripts/build_tensorrt_engine.py \
  --onnx models/arabert-debug-variants/model.onnx \
  --engine reports/tensorrt/model-fp16.engine \
  --report reports/tensorrt/benchmark.json
```

The command builds an FP16 engine with dynamic sequence shapes, runs
`trtexec` latency/throughput benchmarking, records the exact commands and
output, and stores the engine SHA-256 in the manifest. The resulting report
must be compared with the existing PyTorch, ONNX FP32, and ONNX INT8 table
before any deployment decision.

The local CPU environment intentionally does not install TensorRT or CUDA.
Failure is explicit when `trtexec` is unavailable; no CPU result is mislabeled
as TensorRT evidence.
