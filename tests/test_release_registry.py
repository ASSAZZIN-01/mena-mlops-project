from pathlib import Path

from mena_mlops.release_registry import ReleaseRegistry


def test_registry_persists_decision_gates_and_deployment(tmp_path: Path) -> None:
    registry = ReleaseRegistry(tmp_path / "releases.db")
    registry.start_release("release-1", trigger="manual")
    decision = {
        "status": "rejected",
        "selected_variant": "onnx-int8",
        "quality_gate": {"accepted": True, "failures": []},
        "deployed_model_gate": {
            "accepted": False,
            "failures": ["latency regression"],
        },
    }
    registry.record_decision(
        "release-1",
        decision_uri="gs://bucket/release-1/promotion_decision.json",
        decision=decision,
    )
    registry.record_deployment("release-1", status="not_attempted")

    result = registry.get_release("release-1")

    assert result is not None
    assert result["status"] == "rejected"
    assert result["deployment_status"] == "not_attempted"
    assert {gate["gate_name"] for gate in result["gates"]} == {
        "quality_gate",
        "deployed_model_gate",
    }
    assert any(
        event["event_type"] == "promotion_decision"
        for event in result["events"]
    )
    assert result["decision"]["deployed_model_gate"]["failures"] == [
        "latency regression"
    ]
