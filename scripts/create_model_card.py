"""Create a reproducible model card for a manually promoted model."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

from mena_mlops.model_card import build_model_card, sha256_file, write_model_card


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--mlflow-run-id", required=True)
    parser.add_argument("--training-job-id")
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dvc-lock", type=Path, default=Path("dvc.lock"))
    parser.add_argument("--training-image", default="unknown")
    args = parser.parse_args()

    config = yaml.safe_load(args.training_config.read_text(encoding="utf-8"))
    data_config = yaml.safe_load(args.data_config.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    decision = json.loads(args.decision.read_text(encoding="utf-8"))
    lineage = {
        "git_commit": _git_commit(),
        "dvc_commit": _git_commit(),
        "dvc_lock_sha256": sha256_file(args.dvc_lock),
        "training_image": args.training_image,
        "reproduction_command": (
            f"python scripts/submit_vertex_training.py --config "
            f"{args.training_config} --data-config {args.data_config} "
            f"--data-dir {args.data_dir}"
        ),
    }
    card = build_model_card(
        model_id=args.model_id,
        training_job_id=args.training_job_id,
        mlflow_run_id=args.mlflow_run_id,
        config=config,
        data_config=data_config,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        release_dir=args.release_dir,
        evaluation=evaluation,
        promotion_decision=decision,
        lineage=lineage,
    )
    write_model_card(card, args.output_dir)
    print(json.dumps(card, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
