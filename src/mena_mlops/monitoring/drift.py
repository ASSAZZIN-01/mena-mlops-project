"""Reference-based input and prediction drift analysis with Evidently."""

from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
WORD_RE = re.compile(r"\S+")


def build_monitoring_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create privacy-safe numeric features from review and prediction data."""

    if "text" not in frame:
        raise ValueError("monitoring data must contain a text column")
    texts = frame["text"].astype(str)
    result = pd.DataFrame(index=frame.index)
    result["text_length"] = texts.str.len()
    result["word_count"] = texts.map(lambda value: len(WORD_RE.findall(value)))
    result["arabic_char_ratio"] = texts.map(
        lambda value: len(ARABIC_RE.findall(value)) / max(len(value), 1)
    )
    result["digit_ratio"] = texts.map(
        lambda value: sum(char.isdigit() for char in value) / max(len(value), 1)
    )
    result["punctuation_ratio"] = texts.map(
        lambda value: sum(char in string.punctuation for char in value)
        / max(len(value), 1)
    )
    for column in ("prediction", "prediction_confidence"):
        if column in frame:
            result[column] = frame[column].values
    return result.reset_index(drop=True)


def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    """Run Evidently and persist both a human report and machine summary."""

    reference_features = build_monitoring_features(reference)
    current_features = build_monitoring_features(current)
    report = Report([DataDriftPreset()]).run(
        reference_data=reference_features,
        current_data=current_features,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_dir / "drift_report.html"))
    payload = report.dict()
    report.save_json(str(output_dir / "drift_report.json"))
    first_metric = payload["metrics"][0]["value"]
    summary = {
        "reference_rows": len(reference_features),
        "current_rows": len(current_features),
        "drifted_columns": int(first_metric["count"]),
        "drift_share": float(first_metric["share"]),
        "drift_detected": bool(first_metric["share"] > 0),
    }
    (output_dir / "drift_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
