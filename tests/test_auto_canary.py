import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "auto_canary",
    Path(__file__).parents[1] / "scripts" / "auto_canary.py",
)
assert _SPEC is not None and _SPEC.loader is not None
auto_canary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(auto_canary)


def test_canary_progression_uses_all_stages(monkeypatch) -> None:
    applied = []
    monkeypatch.setattr(auto_canary, "_check_prediction_contract", lambda *_: None)
    monkeypatch.setattr(auto_canary, "check_stage", lambda *args, **kwargs: [])

    auto_canary.run_canary(
        prometheus_url="http://prometheus",
        gateway_url="http://gateway",
        stages=(5, 20, 50, 100),
        duration_seconds=0,
        apply=applied.append,
        sleep=lambda _: None,
    )

    assert [split.candidate for split in applied] == [5, 20, 50, 100]


def test_prediction_probe_uses_supported_live_source(monkeypatch) -> None:
    requests = []

    def fake_request(url, payload):
        requests.append(payload)
        return {
            "label": "positive",
            "probabilities": {
                "negative": 0.1,
                "neutral": 0.1,
                "positive": 0.8,
            },
            "model_version": "candidate",
        }

    monkeypatch.setattr(auto_canary, "_json_request", fake_request)

    assert auto_canary._check_prediction_contract("http://gateway", 1) is None
    assert requests[0]["source"] == "live"
