"""Tests for offline batch inference contracts."""

from pathlib import Path

import pandas as pd
import pytest

from mena_mlops.inference.batch import (
    predict_batch,
    read_batch_input,
    write_batch_output,
)


class FakePredictor:
    model_version = "test-v1"

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        label = "positive" if "ممتاز" in text else "negative"
        return label, {"negative": 0.1, "neutral": 0.2, "positive": 0.7}


def test_batch_prediction_preserves_input_and_adds_metadata() -> None:
    records = predict_batch([{"id": "a", "text": "خدمة ممتازة"}], FakePredictor())

    assert records[0]["id"] == "a"
    assert records[0]["prediction"] == "positive"
    assert records[0]["model_version"] == "test-v1"
    assert '"positive": 0.7' in str(records[0]["probabilities"])
    assert records[0]["predicted_at"]


def test_batch_jsonl_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "input.jsonl"
    output = tmp_path / "output.jsonl"
    pd.DataFrame([{"id": "a", "text": "خدمة ممتازة"}]).to_json(
        source, orient="records", lines=True, force_ascii=False
    )

    frame = read_batch_input(source)
    write_batch_output(predict_batch(frame.to_dict("records"), FakePredictor()), output)

    result = pd.read_json(output, lines=True)
    assert result.loc[0, "prediction"] == "positive"


def test_batch_input_requires_text(tmp_path: Path) -> None:
    source = tmp_path / "input.jsonl"
    pd.DataFrame([{"id": "a"}]).to_json(source, orient="records", lines=True)

    with pytest.raises(ValueError, match="text"):
        read_batch_input(source)
