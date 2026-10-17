"""Build and benchmark an optional TensorRT FP16 engine on a GPU host."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "TensorRT requires trtexec on the approved GPU build host"
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"TensorRT command failed ({error.returncode}): "
            f"{' '.join(command)}\n{error.stderr}"
        ) from error
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--duration", type=int, default=30)
    args = parser.parse_args()
    if not args.onnx.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {args.onnx}")
    trtexec = shutil.which("trtexec")
    if trtexec is None:
        raise RuntimeError(
            "TensorRT requires trtexec on the approved GPU build host"
        )

    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    max_shape = f"1x{args.max_length}"
    build_command = [
        trtexec,
        f"--onnx={args.onnx}",
        f"--saveEngine={args.engine}",
        "--fp16",
        "--minShapes=input_ids:1x1,attention_mask:1x1,token_type_ids:1x1",
        f"--optShapes=input_ids:{max_shape},attention_mask:{max_shape},token_type_ids:{max_shape}",
        f"--maxShapes=input_ids:{max_shape},attention_mask:{max_shape},token_type_ids:{max_shape}",
        "--skipInference",
    ]
    build_output = run(build_command)
    benchmark_command = [
        trtexec,
        f"--loadEngine={args.engine}",
        f"--iterations={args.iterations}",
        f"--duration={args.duration}",
        "--fp16",
    ]
    benchmark_output = run(benchmark_command)
    manifest = {
        "format": "tensorrt-fp16",
        "onnx_path": str(args.onnx),
        "engine_path": str(args.engine),
        "engine_sha256": sha256(args.engine),
        "max_length": args.max_length,
        "build_command": build_command,
        "benchmark_command": benchmark_command,
        "build_output": build_output[-4000:],
        "benchmark_output": benchmark_output[-8000:],
    }
    args.report.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
