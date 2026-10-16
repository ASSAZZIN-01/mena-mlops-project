from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "submit_vertex_retraining",
    Path(__file__).parents[1] / "scripts" / "submit_vertex_retraining.py",
)
assert _SPEC is not None and _SPEC.loader is not None
submit_vertex_retraining = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(submit_vertex_retraining)


def test_release_decision_handoff_contains_release_uri(tmp_path: Path) -> None:
    output = tmp_path / "reports" / "release-decision-uri.txt"

    decision_uri = submit_vertex_retraining._write_release_decision_uri(
        output,
        "gs://bucket/vertex-staging/job-123/release/",
    )

    assert decision_uri == (
        "gs://bucket/vertex-staging/job-123/release/promotion_decision.json"
    )
    assert output.read_text(encoding="utf-8") == decision_uri + "\n"


def test_release_decision_handoff_replaces_stale_release(tmp_path: Path) -> None:
    output = tmp_path / "release-decision-uri.txt"
    output.write_text("gs://bucket/old-release/promotion_decision.json\n")

    submit_vertex_retraining._write_release_decision_uri(
        output,
        "gs://bucket/new-release",
    )

    assert output.read_text(encoding="utf-8") == (
        "gs://bucket/new-release/promotion_decision.json\n"
    )
