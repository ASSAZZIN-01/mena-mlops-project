"""Evaluate candidate quality metrics before applying a canary promotion."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

from mena_mlops.deployment.quality_gate import evaluate_quality_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stable", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/promotion.yaml"))
    parser.add_argument("--candidate-percent", type=int, default=20)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["quality_gate"]
    stable = json.loads(args.stable.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    accepted, reasons = evaluate_quality_gate(
        stable,
        candidate,
        minimum_macro_f1=float(config["minimum_macro_f1"]),
        minimum_improvement=float(config["minimum_improvement"]),
        minimum_class_f1=config["minimum_class_f1"],
    )
    if not accepted:
        raise SystemExit("Promotion blocked: " + "; ".join(reasons))
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/canary.py",
            "--candidate",
            str(args.candidate_percent),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
