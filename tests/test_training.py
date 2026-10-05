import pandas as pd
import pytest

from mena_mlops.training.data import build_label_maps, encode_labels
from mena_mlops.training.evaluation import evaluate_by_source, evaluate_predictions


def test_label_mapping_is_explicit_and_stable() -> None:
    labels = ["negative", "neutral", "positive"]

    label_to_id, id_to_label = build_label_maps(labels)

    assert label_to_id == {"negative": 0, "neutral": 1, "positive": 2}
    assert id_to_label == {0: "negative", 1: "neutral", 2: "positive"}
    assert encode_labels(
        pd.DataFrame({"label": ["positive", "negative"]}), label_to_id
    ) == [2, 0]


def test_unknown_label_fails_before_training() -> None:
    with pytest.raises(ValueError, match="Unknown labels"):
        encode_labels(
            pd.DataFrame({"label": ["unknown"]}),
            {"negative": 0, "positive": 1},
        )


def test_evaluation_reports_per_class_and_source_metrics() -> None:
    labels = [0, 1, 2, 0]
    predictions = [0, 2, 2, 1]
    frame = pd.DataFrame({"source": ["a", "a", "b", "b"]})

    metrics = evaluate_predictions(
        labels, predictions, ["negative", "neutral", "positive"]
    )
    by_source = evaluate_by_source(
        frame, labels, predictions, ["negative", "neutral", "positive"]
    )

    assert metrics["accuracy"] == 0.5
    assert set(metrics["per_class"]) == {"negative", "neutral", "positive"}
    assert set(by_source) == {"a", "b"}


def test_evaluation_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        evaluate_predictions([0], [], ["negative", "positive"])
