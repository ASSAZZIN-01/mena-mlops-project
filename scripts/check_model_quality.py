"""Fail when evaluated model quality regresses below the project baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from mena_mlops.quality import check_quality


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/quality.yaml"))
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["model_quality"]
    failures = check_quality(
        metrics,
        baseline_macro_f1=float(config["baseline_test_macro_f1"]),
        minimum_neutral_f1=float(config["minimum_test_neutral_f1"]),
    )
    if failures:
        raise SystemExit("Model quality gate failed: " + "; ".join(failures))
    print("Model quality gate passed")


if __name__ == "__main__":
    main()
