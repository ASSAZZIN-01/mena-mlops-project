"""Build ONNX FP32 and ONNX dynamic-INT8 variants from a model artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

from mena_mlops.optimization import export_onnx_model
from mena_mlops.serving.app import _load_predictor


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("models/arabert-debug"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/arabert-debug-variants"),
    )
    args = parser.parse_args()

    import os

    os.environ["MODEL_PATH"] = str(args.model_path)
    os.environ["MODEL_DEVICE"] = "cpu"
    predictor = _load_predictor()
    onnx_path = args.output_dir / "model.onnx"
    int8_path = args.output_dir / "model-int8.onnx"
    export_onnx_model(predictor.model, predictor.tokenizer, onnx_path)
    quantize_dynamic(
        str(onnx_path),
        str(int8_path),
        weight_type=QuantType.QInt8,
    )
    manifest = {
        "base_model_path": str(args.model_path),
        "base_model_version": predictor.model_version,
        "variants": {
            "onnx-fp32": {"path": str(onnx_path), "sha256": sha256(onnx_path)},
            "onnx-int8": {"path": str(int8_path), "sha256": sha256(int8_path)},
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
