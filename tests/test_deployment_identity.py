import json
from pathlib import Path

import pytest

from mena_mlops.deployment_identity import build_deployment_identity


def test_deployment_identity_links_approved_card_and_release(tmp_path: Path) -> None:
    card_dir = tmp_path / "model-card"
    card_dir.mkdir()
    (card_dir / "model_card.json").write_text(
        json.dumps(
            {
                "model_id": "model-1",
                "status": "approved",
                "promotion": {"selected_variant": "onnx-int8"},
            }
        ),
        encoding="utf-8",
    )
    decision = {"status": "approved", "selected_variant": "onnx-int8"}

    identity = build_deployment_identity(
        decision=decision,
        decision_uri="gs://bucket/release/promotion_decision.json",
        bundle="gs://bucket/release",
        output_dir=tmp_path,
        selected_variant="onnx-int8",
    )

    assert identity["model_id"] == "model-1"
    assert identity["model_card_uri"] == (
        "gs://bucket/release/model-card/model_card.json"
    )
    assert len(identity["model_card_sha256"]) == 64
    assert identity["source_revision"] == "unknown"
    assert identity["deployment_stage"] == "candidate"


def test_deployment_identity_rejects_missing_card(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="missing model-card"):
        build_deployment_identity(
            decision={"status": "approved", "selected_variant": "onnx-int8"},
            decision_uri="release/promotion_decision.json",
            bundle=str(tmp_path),
            output_dir=tmp_path,
            selected_variant="onnx-int8",
        )


def test_deployment_identity_rejects_variant_mismatch(tmp_path: Path) -> None:
    card_dir = tmp_path / "model-card"
    card_dir.mkdir()
    (card_dir / "model_card.json").write_text(
        json.dumps(
            {
                "model_id": "model-1",
                "status": "approved",
                "promotion": {"selected_variant": "onnx-fp32"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="selected variant"):
        build_deployment_identity(
            decision={"status": "approved", "selected_variant": "onnx-int8"},
            decision_uri="release/promotion_decision.json",
            bundle=str(tmp_path),
            output_dir=tmp_path,
            selected_variant="onnx-int8",
        )
