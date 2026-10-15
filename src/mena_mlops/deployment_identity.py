"""Validate and persist the identity linking deployments to model cards."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from mena_mlops.model_card import sha256_file


def _json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def build_deployment_identity(
    *,
    decision: dict[str, Any],
    decision_uri: str,
    bundle: str,
    output_dir: Path,
    selected_variant: str,
) -> dict[str, object]:
    card_path = output_dir / "model-card" / "model_card.json"
    if not card_path.is_file():
        raise RuntimeError(
            "release bundle is missing model-card/model_card.json; "
            "refusing deployment without an explicit model identity"
        )
    card = json.loads(card_path.read_text(encoding="utf-8"))
    if not isinstance(card, dict):
        raise RuntimeError("model card must be a JSON object")
    model_id = card.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError("model card has no model_id")
    if card.get("status") != "approved":
        raise RuntimeError("model card is not approved for deployment")
    promotion = card.get("promotion")
    if not isinstance(promotion, dict):
        raise RuntimeError("model card has no promotion decision")
    if promotion.get("selected_variant") != selected_variant:
        raise RuntimeError(
            "model card selected variant does not match the promotion decision"
        )
    return {
        "schema_version": "1.0",
        "model_id": model_id,
        "model_card_uri": (
            f"{bundle.rstrip('/')}/model-card/model_card.json"
            if urlparse(bundle).scheme == "gs"
            else str(card_path)
        ),
        "model_card_sha256": sha256_file(card_path),
        "release_uri": bundle,
        "promotion_decision_uri": decision_uri,
        "promotion_decision_sha256": _json_hash(decision),
        "selected_variant": selected_variant,
        "deployment_stage": "candidate",
        "deployed_at": datetime.now(UTC).isoformat(),
    }
