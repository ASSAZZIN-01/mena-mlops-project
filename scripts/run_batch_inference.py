"""Run offline model inference over a JSONL or Parquet file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from mena_mlops.inference.batch import (
    predict_batch,
    read_batch_input,
    write_batch_output,
)
from mena_mlops.serving.app import _load_predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input JSONL or Parquet file")
    parser.add_argument("output", type=Path, help="Output JSONL or Parquet file")
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Model directory; defaults to MODEL_PATH or models/arabert-debug",
    )
    parser.add_argument(
        "--model-version",
        help="Version recorded in outputs; defaults to MODEL_VERSION or model name",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="Inference device; defaults to MODEL_DEVICE or auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model_path is not None:
        os.environ["MODEL_PATH"] = str(args.model_path)
    if args.model_version is not None:
        os.environ["MODEL_VERSION"] = args.model_version
    if args.device is not None:
        os.environ["MODEL_DEVICE"] = args.device

    frame = read_batch_input(args.input)
    predictor = _load_predictor()
    records = predict_batch(frame.to_dict(orient="records"), predictor)
    write_batch_output(records, args.output)
    print(f"Scored {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
