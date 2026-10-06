"""Run an Evidently drift report over two JSONL monitoring windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mena_mlops.monitoring.drift import run_drift_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/drift"))
    args = parser.parse_args()
    summary = run_drift_report(
        pd.read_json(args.reference, lines=True),
        pd.read_json(args.current, lines=True),
        args.output_dir,
    )
    print(summary)


if __name__ == "__main__":
    main()
