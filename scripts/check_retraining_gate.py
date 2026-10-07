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
    counts = store.counts()
    ready = store.thresholds_reached(thresholds)
    print(json.dumps({"ready": ready, "counts": counts, "thresholds": thresholds}))
    raise SystemExit(0 if ready else 1)


if __name__ == "__main__":
    main()
