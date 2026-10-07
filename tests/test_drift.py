import json

import pandas as pd

from mena_mlops.monitoring.drift import (
    build_monitoring_features,
    run_drift_report,
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
