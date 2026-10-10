"""Tests for reviewer dashboard input normalization."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_DASHBOARD_PATH = Path(__file__).parents[1] / "scripts" / "dashboard.py"
_SPEC = importlib.util.spec_from_file_location("dashboard", _DASHBOARD_PATH)
assert _SPEC and _SPEC.loader
_DASHBOARD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DASHBOARD)
prepare_batch_frame = _DASHBOARD.prepare_batch_frame
validate_frame = _DASHBOARD.validate_frame


def test_prepare_batch_frame_adds_id_and_source() -> None:
    result = prepare_batch_frame(pd.DataFrame({"text": ["نص جيد"]}))

    assert result["ID"].tolist() == ["row-000001"]
    assert result["source"].tolist() == [None]


def test_prepare_batch_frame_preserves_input_metadata() -> None:
    result = prepare_batch_frame(
        pd.DataFrame({"ID": ["review-1"], "text": ["نص"], "source": ["test-set"]})
    )

    assert result.to_dict("records") == [
        {"ID": "review-1", "text": "نص", "source": "test-set"}
    ]


def test_validate_frame_rejects_whitespace_text() -> None:
    with pytest.raises(ValueError, match="empty or null"):
        validate_frame(pd.DataFrame({"text": ["   "]}))
