import json

from mena_mlops.observability import emit_event


def test_emit_event_is_json_and_contains_correlation_fields(
    capsys, monkeypatch
) -> None:
    monkeypatch.setenv("MENA_MODEL_ID", "model-1")
    event = emit_event(
        "promotion_completed",
        "promotion",
        status="approved",
        release_id="release-1",
        duration_ms=12.5,
    )

    output = json.loads(capsys.readouterr().out)
    assert output == event
    assert output["event_type"] == "promotion_completed"
    assert output["component"] == "promotion"
    assert output["status"] == "approved"
    assert output["model_id"] == "model-1"
    assert output["release_id"] == "release-1"
    assert output["duration_ms"] == 12.5
    assert output["timestamp"].endswith("+00:00")
