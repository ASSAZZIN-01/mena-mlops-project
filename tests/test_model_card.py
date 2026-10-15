from pathlib import Path

import pandas as pd

from mena_mlops.model_card import build_model_card, write_model_card


def test_model_card_contains_reproducibility_identity(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for split in ("train", "validation", "test"):
        pd.DataFrame({"text": ["نص"], "label": ["positive"]}).to_parquet(
            data_dir / f"{split}.parquet",
            index=False,
        )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"model")
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    (release_dir / "model-int8.onnx").write_bytes(b"optimized")

    card = build_model_card(
        model_id="manual-model-1",
        training_job_id="projects/example/jobs/1",
        mlflow_run_id="run-1",
        config={
            "seed": 42,
            "model": {"name": "model-name"},
            "training": {"max_length": 384},
        },
        data_config={
            "labels": {"output": ["negative", "positive"]},
            "split": {"debug": True},
        },
        data_dir=data_dir,
        model_dir=model_dir,
        release_dir=release_dir,
        evaluation={"test_macro_f1": 0.8},
        promotion_decision={
            "status": "approved",
            "selected_variant": "onnx-int8",
            "quality_gate": {"accepted": True},
            "optimization_gate": {"accepted": True},
        },
        lineage={
            "git_commit": "abc123",
            "dvc_commit": "abc123",
            "dvc_lock_sha256": "dvc-hash",
            "training_image": "image@sha256:digest",
            "reproduction_command": "python scripts/submit_vertex_training.py",
        },
    )
    write_model_card(card, tmp_path / "card")

    assert card["model_id"] == "manual-model-1"
    assert card["parent_model_id"] is None
    assert card["data"]["splits"]["test"]["rows"] == 1
    assert card["artifacts"]["release"]["model-int8.onnx"]
    assert (tmp_path / "card" / "model_card.json").exists()
    assert "DVC lock SHA-256" in (tmp_path / "card" / "MODEL_CARD.md").read_text()
