import json

import pandas as pd

from mena_mlops.monitoring.drift import (
    build_monitoring_features,
    run_drift_report,
)
from mena_mlops.retraining import (
    drift_triggered,
    feedback_thresholds_reached,
)


def test_monitoring_features_do_not_include_raw_text() -> None:
    frame = pd.DataFrame(
        {
            "text": ["منتج ممتاز 123"],
            "prediction": ["positive"],
            "prediction_confidence": [0.9],
        }
    )

    features = build_monitoring_features(frame)

    assert "text" not in features
    assert features.loc[0, "arabic_char_ratio"] > 0
    assert features.loc[0, "prediction"] == "positive"


def test_drift_report_writes_machine_summary(tmp_path) -> None:
    reference = pd.DataFrame({"text": ["جيد", "ممتاز", "رائع"]})
    current = pd.DataFrame(
        {"text": ["1234567890", "1234567890", "1234567890"]}
    )

    summary = run_drift_report(reference, current, tmp_path)

    assert summary["reference_rows"] == 3
    assert summary["current_rows"] == 3
    assert (tmp_path / "drift_report.html").is_file()
    saved = json.loads((tmp_path / "drift_summary.json").read_text())
    assert saved["drift_detected"] is True


def test_retraining_triggers_when_drift_is_detected_without_feedback() -> None:
    assert drift_triggered({"drift_detected": True, "psi_alert": False}) is True
    assert feedback_thresholds_reached(
        {"negative": 0, "neutral": 0, "positive": 0},
        {"negative": 5, "neutral": 1, "positive": 5},
    ) is False


def test_retraining_triggers_when_feedback_is_ready_without_drift() -> None:
    assert drift_triggered({"drift_detected": False, "psi_alert": False}) is False
    assert feedback_thresholds_reached(
        {"negative": 5, "neutral": 1, "positive": 5},
        {"negative": 5, "neutral": 1, "positive": 5},
    ) is True


def test_retraining_skips_when_neither_trigger_is_ready() -> None:
    assert drift_triggered({"drift_detected": False, "psi_alert": False}) is False
    assert feedback_thresholds_reached(
        {"negative": 4, "neutral": 1, "positive": 5},
        {"negative": 5, "neutral": 1, "positive": 5},
    ) is False
