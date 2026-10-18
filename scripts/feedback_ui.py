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
    exclusions = store.exclusion_counts()
    annotation = store.annotation_summary(
        Path(os.getenv("AUDIT_DB", "data/audit/predictions.db"))
    )

    with st.expander("Where does this feedback come from?", expanded=True):
        st.markdown(
            """
            The inference services write every prediction to the audit database.
            Predictions in the configured low-confidence window are also copied
            to this review queue. **Correct** validates the model label, **Wrong**
            records your corrected label, and **I don't know** defers ambiguous
            text. Rows matching protected train, validation, or test splits are
            quarantined and never count toward retraining thresholds.
            """
        )

    total_predictions = sum(
        values["total_predictions"] for values in annotation.values()
    )
    suspected = sum(values["suspected"] for values in annotation.values())
    reviewed = sum(values["reviewed"] for values in annotation.values())
    pending = sum(values["pending"] for values in annotation.values())
    metric_columns = st.columns(5)
    metric_columns[0].metric("All predictions", total_predictions)
    metric_columns[1].metric("Suspected queue", suspected)
    metric_columns[2].metric("Reviewed", reviewed)
    metric_columns[3].metric("Pending", pending)
    metric_columns[4].metric(
        "Protected rows",
        sum(exclusions.values()),
        help="Rows excluded because they match a protected data split.",
    )

    st.subheader("Class and annotation progress")
    for label in LABELS:
        values = annotation[label]
        queue_total = values["suspected"]
        if queue_total:
            reviewed_width = values["reviewed"] / queue_total * 100
            confusing_width = values["confusing"] / queue_total * 100
            pending_width = values["pending"] / queue_total * 100
        else:
            reviewed_width = confusing_width = pending_width = 0
        st.markdown(
            f"**{label.title()}** · {values['suspected']} suspected of "
            f"{values['total_predictions']} predictions · "
            f"{values['reviewed']} reviewed · {values['confusing']} confusing · "
            f"{values['pending']} pending"
        )
        st.markdown(
            f"""
            <div class="feedback-progress" style="display:flex;height:12px;
            border-radius:8px;overflow:hidden;background:#26364a">
              <span style="width:{reviewed_width:.2f}%;background:#35b779"></span>
              <span style="width:{confusing_width:.2f}%;background:#f5bd5b"></span>
              <span style="width:{pending_width:.2f}%;background:#ef7c83"></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Retraining threshold: "
            f"{counts[label]} / {thresholds[label]} validated labels"
        )

    st.caption("Green = reviewed · Amber = confusing/deferred · Red = pending")

    item = store.next_pending()
    if item is None:
        st.success("No pending predictions need review.")
        return

    st.divider()
    pending_before_item = pending
    st.subheader("Current annotation")
    context_columns = st.columns(3)
    context_columns[0].metric("Queue position", f"1 / {pending_before_item}")
    context_columns[1].metric(
        f"{item.predicted_label.title()} queue",
        f"{annotation[item.predicted_label]['pending']} pending",
    )
    context_columns[2].metric("Model confidence", f"{item.confidence:.1%}")
    st.caption(
        f"Review ID {item.id} · model version "
        f"{os.getenv('MODEL_VERSION', 'stable service')} · oldest pending item first"
    )
    st.write(f"**Review:** {item.text}")
    st.write(f"**Model prediction:** `{item.predicted_label}`")

    decision = st.radio(
        "Is the prediction correct?",
        options=("Correct", "Wrong", "I don't know"),
        horizontal=True,
        help=(
            "Correct keeps the predicted label. Wrong requires the true label. "
            "I don't know defers ambiguous text and excludes it from retraining."
        ),
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

    button_label = "Defer review" if decision == "I don't know" else "Save review"
    if st.button(button_label, type="primary", key=f"save-{item.id}"):
        if decision == "I don't know":
            store.defer(item.id)
        else:
            store.review(
                item.id,
                correct=decision == "Correct",
                corrected_label=corrected_label,
            )
        st.rerun()


if __name__ == "__main__":
    main()
