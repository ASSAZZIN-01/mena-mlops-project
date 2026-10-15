"""Check whether reviewed feedback has reached retraining thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from mena_mlops.feedback import FeedbackStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/feedback/reviewed.db"),
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/feedback.yaml")
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    thresholds = {
        str(label): int(value)
        for label, value in config["retraining_thresholds"].items()
    }
    store = FeedbackStore(args.database)
    checkpoint = store.retraining_checkpoint()
    counts = store.counts_since(checkpoint.last_feedback_id)
    ready = all(
        counts.get(label, 0) >= required
        for label, required in thresholds.items()
    )
    print(
        json.dumps(
            {
                "ready": ready,
                "counts": counts,
                "total_counts": store.counts(),
                "thresholds": thresholds,
                "last_feedback_id": checkpoint.last_feedback_id,
            }
        )
    )
    raise SystemExit(0 if ready else 1)


if __name__ == "__main__":
    main()
