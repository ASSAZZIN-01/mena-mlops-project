"""Streamlit interface for reviewing uncertain predictions."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
import yaml

from mena_mlops.feedback import LABELS, FeedbackStore


def load_thresholds() -> dict[str, int]:
    config_path = Path(
        os.getenv("FEEDBACK_CONFIG", "configs/feedback.yaml")
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return {
        str(label): int(value)
        for label, value in config["retraining_thresholds"].items()
    }


def main() -> None:
    st.set_page_config(page_title="MENA feedback review", page_icon="✅")
    st.title("MENA sentiment feedback review")
    st.caption("Review one low-confidence prediction at a time.")

    store = FeedbackStore(
        Path(os.getenv("FEEDBACK_DB", "data/feedback/reviewed.db"))
    )
    thresholds = load_thresholds()
    counts = store.counts()
    st.subheader("Validated examples")
    st.write({label: f"{counts[label]} / {thresholds[label]}" for label in LABELS})

    item = store.next_pending()
    if item is None:
        st.success("No pending predictions need review.")
        return

    st.divider()
    st.write(f"**Review ID:** {item.id}")
    st.write(f"**Review:** {item.text}")
    st.write(f"**Model prediction:** `{item.predicted_label}`")
    st.write(f"**Confidence:** `{item.confidence:.1%}`")

    decision = st.radio(
        "Is the prediction correct?",
        options=("Correct", "Wrong"),
        horizontal=True,
        key=f"decision-{item.id}",
    )
    corrected_label = None
    if decision == "Wrong":
        corrected_label = st.radio(
            "Choose the correct label",
            options=LABELS,
            horizontal=True,
            key=f"label-{item.id}",
        )

    if st.button("Save review", type="primary", key=f"save-{item.id}"):
        store.review(
            item.id,
            correct=decision == "Correct",
            corrected_label=corrected_label,
        )
        st.rerun()


if __name__ == "__main__":
    main()
