"""Metrics and reports for sentiment model evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def evaluate_predictions(
    labels: list[int],
    predictions: list[int],
    label_names: list[str],
) -> dict[str, Any]:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have equal lengths")
    per_class = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(len(label_names))),
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "per_class": {
            name: {
                "precision": float(per_class[0][index]),
                "recall": float(per_class[1][index]),
                "f1": float(per_class[2][index]),
                "support": int(per_class[3][index]),
            }
            for index, name in enumerate(label_names)
        },
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=list(range(len(label_names))
        )).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=list(range(len(label_names))),
            target_names=label_names,
            zero_division=0,
            output_dict=True,
        ),
    }


def evaluate_by_source(
    frame: Any,
    labels: list[int],
    predictions: list[int],
    label_names: list[str],
) -> dict[str, Any]:
    results = {}
    for source in sorted(frame["source"].unique()):
        indexes = np.flatnonzero(frame["source"].to_numpy() == source).tolist()
        results[str(source)] = evaluate_predictions(
            [labels[index] for index in indexes],
            [predictions[index] for index in indexes],
            label_names,
        )
    return results
