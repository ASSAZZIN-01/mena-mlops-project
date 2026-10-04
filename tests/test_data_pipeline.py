import pandas as pd

from mena_mlops.data_pipeline import deduplicate, normalize_frame, split_data


def config() -> dict:
    return {
        "seed": 42,
        "cleaning": {
            "min_text_length": 2,
            "remove_tatweel": True,
            "remove_diacritics": False,
        },
        "labels": {
            "output": ["negative", "neutral", "positive"],
            "mappings": {
                "positive": "positive",
                "negative": "negative",
                "neutral": "neutral",
            },
            "rating": {"1": "negative", "2": "negative", "3": "neutral",
                       "4": "positive", "5": "positive"},
            "binary": {"0": "negative", "1": "positive"},
            "signed": {"-1": "negative", "0": "neutral", "1": "positive"},
        },
        "split": {
            "debug": False,
            "outer_folds": 10,
            "full_test_fold": 9,
            "full_validation_fold": 8,
        },
    }


def test_normalize_frame_maps_ratings_and_generates_ids() -> None:
    source = {
        "name": "example",
        "columns": {"text": "review", "label": "stars"},
        "labels": {"kind": "rating"},
    }
    frame = pd.DataFrame({"review": ["جيد ـ جدا", "سيئ"], "stars": [5, 1]})

    result = normalize_frame(frame, source, config())

    assert list(result.columns) == ["id", "text", "label", "source"]
    assert result["label"].tolist() == ["positive", "negative"]
    assert result["text"].tolist() == ["جيد جدا", "سيئ"]
    assert result["id"].is_unique


def test_deduplicate_removes_repeated_text_and_reports_conflicts() -> None:
    frame = pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "text": ["نص", "نص", "نص مختلف"],
            "label": ["positive", "positive", "negative"],
            "source": ["a", "b", "a"],
        }
    )

    result, report = deduplicate(frame)

    assert len(result) == 2
    assert report["duplicate_rows_removed"] == 1
    assert report["conflicting_texts"] == 0


def test_full_split_is_reproducible_and_keeps_source_label_strata() -> None:
    rows = [
        {
            "id": f"{source}-{label}-{index}",
            "text": f"text-{source}-{label}-{index}",
            "label": label,
            "source": source,
        }
        for source in ["a", "b"]
        for label in ["negative", "positive"]
        for index in range(10)
    ]
    frame = pd.DataFrame(rows)

    first = split_data(frame, config())
    second = split_data(frame, config())

    for name in ["train", "validation", "test"]:
        assert first[name]["id"].tolist() == second[name]["id"].tolist()
    assert set(first["train"]["id"]).isdisjoint(first["test"]["id"])
    assert set(first["validation"]["id"]).isdisjoint(first["test"]["id"])
