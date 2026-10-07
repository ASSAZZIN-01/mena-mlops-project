"""Gate and launch a candidate retraining run with reviewed feedback."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import yaml

from mena_mlops.feedback import FeedbackStore
from mena_mlops.training.train import load_yaml, run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database", type=Path, default=Path("data/feedback/reviewed.db")
    )
    parser.add_argument(
        "--feedback-config",
        type=Path,
        default=Path("configs/feedback.yaml"),
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/training.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/arabert-candidate"),
    )
    args = parser.parse_args()

    feedback_config = yaml.safe_load(args.feedback_config.read_text(encoding="utf-8"))
    thresholds = {
        str(label): int(value)
        for label, value in feedback_config["retraining_thresholds"].items()
    }
    store = FeedbackStore(args.database)
    if not store.thresholds_reached(thresholds):
        raise SystemExit("Retraining gate is not ready; thresholds have not been met.")

    training_config = load_yaml(args.training_config)
    source_dir = Path(training_config["paths"]["data_dir"])
    with tempfile.TemporaryDirectory(prefix="mena-retraining-") as temporary:
        data_dir = Path(temporary)
        for split in ("validation", "test"):
            shutil.copy2(source_dir / f"{split}.parquet", data_dir / f"{split}.parquet")
        train = pd.read_parquet(source_dir / "train.parquet")
        feedback_path = data_dir / "feedback.parquet"
        store.export_reviewed(feedback_path)
        reviewed = pd.read_parquet(feedback_path)
        combined = pd.concat([train, reviewed], ignore_index=True)
        combined.to_parquet(data_dir / "train.parquet", index=False)

        candidate_config = dict(training_config)
        candidate_config["paths"] = dict(training_config["paths"])
        candidate_config["paths"]["data_dir"] = str(data_dir)
        candidate_config["paths"]["output_dir"] = str(args.output_dir)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", encoding="utf-8", delete=False
        ) as config_file:
            yaml.safe_dump(candidate_config, config_file)
            config_path = Path(config_file.name)
        run_id = run_training(config_path)
    print(f"MLflow candidate run: {run_id}")


if __name__ == "__main__":
    main()
