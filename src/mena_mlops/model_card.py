"""Build reproducible identity cards for trained and promoted models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _artifact_hashes(directory: Path) -> dict[str, str]:
    if not directory.exists():
        return {}
    return {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def build_model_card(
    *,
    model_id: str,
    training_job_id: str | None,
    mlflow_run_id: str,
    config: dict[str, Any],
    data_config: dict[str, Any],
    data_dir: Path,
    model_dir: Path,
    release_dir: Path | None,
    evaluation: dict[str, Any] | None,
    promotion_decision: dict[str, Any] | None,
    lineage: dict[str, Any],
) -> dict[str, Any]:
    required_lineage = ("git_commit", "dvc_lock_sha256", "training_image")
    missing_lineage = [
        field for field in required_lineage if not lineage.get(field)
    ]
    if missing_lineage:
        raise ValueError(
            "model card requires lineage fields: " + ", ".join(missing_lineage)
        )
    split_files = {
        split: {
            "path": f"{split}.parquet",
            "sha256": sha256_file(data_dir / f"{split}.parquet"),
            "rows": _row_count(data_dir / f"{split}.parquet"),
        }
        for split in ("train", "validation", "test")
    }
    selected_variant = (
        promotion_decision.get("selected_variant")
        if promotion_decision
        else None
    )
    return {
        "schema_version": "1.0",
        "model_id": model_id,
        "parent_model_id": None,
        "status": (
            promotion_decision.get("status", "trained")
            if promotion_decision
            else "trained"
        ),
        "intended_use": "Arabic sentiment classification for the MENA MLOps project.",
        "limitations": [
            "Evaluation is only representative of the recorded test split.",
            "Subgroup performance must be reviewed before production use.",
            (
                "The card does not authorize deployment when the promotion "
                "status is not approved."
            ),
        ],
        "training": {
            "job_id": training_job_id,
            "mlflow_run_id": mlflow_run_id,
            "model_name": config["model"]["name"],
            "seed": config["seed"],
            "parameters": config["training"],
            "config": config,
            "config_sha256": _json_hash(config),
            "data_config": data_config,
            "data_config_sha256": _json_hash(data_config),
        },
        "data": {
            "splits": split_files,
            "labels": data_config["labels"],
            "split": data_config["split"],
        },
        "lineage": lineage,
        "evaluation": evaluation,
        "promotion": promotion_decision,
        "artifacts": {
            "base_model": _artifact_hashes(model_dir),
            "release": _artifact_hashes(release_dir)
            if release_dir is not None
            else {},
            "selected_variant": selected_variant,
        },
    }


def _row_count(path: Path) -> int:
    import pandas as pd

    return int(len(pd.read_parquet(path)))


def write_model_card(card: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_card.json").write_text(
        json.dumps(card, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    training = card["training"]
    promotion = card.get("promotion") or {}
    lines = [
        f"# Model Card: `{card['model_id']}`",
        "",
        f"- Status: **{card['status']}**",
        f"- Model: `{training['model_name']}`",
        f"- MLflow run: `{training['mlflow_run_id']}`",
        f"- Vertex job: `{training.get('job_id') or 'not applicable'}`",
        f"- Parent model: `{card.get('parent_model_id') or 'none'}`",
        "",
        "## Reproducibility",
        "",
        f"- Training config SHA-256: `{training['config_sha256']}`",
        f"- Data config SHA-256: `{training['data_config_sha256']}`",
        f"- Git commit: `{card['lineage'].get('git_commit', 'unknown')}`",
        f"- DVC lock SHA-256: `{card['lineage'].get('dvc_lock_sha256', 'unknown')}`",
        f"- DVC commit: `{card['lineage'].get('dvc_commit', 'same as Git commit')}`",
        f"- Training image: `{card['lineage'].get('training_image', 'unknown')}`",
        (
            "- Reproduction command: "
            f"`{card['lineage'].get('reproduction_command', 'recorded by launcher')}`"
        ),
        "",
        "## Data splits",
        "",
    ]
    for split, details in card["data"]["splits"].items():
        lines.append(
            f"- `{split}`: {details['rows']} rows, "
            f"`{details['sha256']}`"
        )
    lines.extend(
        [
            "",
            "## Evaluation and promotion",
            "",
            f"- Selected variant: `{promotion.get('selected_variant') or 'none'}`",
            f"- Quality gate: `{promotion.get('quality_gate', {}).get('accepted')}`",
            (
                "- Optimization gate: "
                f"`{promotion.get('optimization_gate', {}).get('accepted')}`"
            ),
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {limitation}" for limitation in card["limitations"])
    (output_dir / "MODEL_CARD.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
