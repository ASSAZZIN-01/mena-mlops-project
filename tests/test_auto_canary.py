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
