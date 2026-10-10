"""Reviewer dashboard for live and batch sentiment inference."""

from __future__ import annotations

import io
import json
import os
from html import escape
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

API_URL = os.getenv("MODEL_API_URL", "http://localhost:8080").rstrip("/")
LINKS = {
    "Grafana": os.getenv("GRAFANA_URL", "http://localhost:3000"),
    "Prometheus": os.getenv("PROMETHEUS_URL", "http://localhost:9090"),
    "MLflow": os.getenv("MLFLOW_URL", "http://localhost:5000"),
    "Airflow": os.getenv("AIRFLOW_URL", "http://localhost:8081"),
}


def predict(text: str) -> dict[str, object]:
    request = Request(
        f"{API_URL}/predict",
        data=json.dumps({"text": text}, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Prediction API returned HTTP {error.code}: {detail}"
        ) from error
    except URLError as error:
        raise RuntimeError(f"Prediction API is unavailable: {error.reason}") from error


def validate_frame(frame: pd.DataFrame) -> None:
    if "text" not in frame.columns:
        raise ValueError("CSV must contain a 'text' column")
    if frame.empty:
        raise ValueError("CSV must contain at least one row")
    text_values = frame["text"].astype("string")
    if text_values.isna().any() or text_values.str.strip().eq("").any():
        raise ValueError("CSV contains an empty or null text value")


def prepare_batch_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the reviewer CSV shape with generated IDs and optional source."""

    validate_frame(frame)
    prepared = frame.copy()
    if "ID" not in prepared.columns:
        prepared.insert(
            0,
            "ID",
            [f"row-{index:06d}" for index in range(1, len(prepared) + 1)],
        )
    else:
        missing_ids = prepared["ID"].isna() | (
            prepared["ID"].astype(str).str.strip().eq("")
        )
        prepared.loc[missing_ids, "ID"] = [
            f"row-{index:06d}" for index in range(1, missing_ids.sum() + 1)
        ]
    if "source" not in prepared.columns:
        prepared["source"] = None
    else:
        prepared["source"] = prepared["source"].where(prepared["source"].notna(), None)
    return prepared


def batch_predict(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    progress = st.progress(0, text="Starting batch inference")
    total = len(frame)
    for index, record in enumerate(frame.to_dict(orient="records"), start=1):
        result = predict(str(record["text"]))
        rows.append(
            {
                **record,
                **result,
                "probabilities": json.dumps(
                    result["probabilities"], ensure_ascii=False, sort_keys=True
                ),
            }
        )
        progress.progress(index / total, text=f"Scored {index}/{total} rows")
    progress.empty()
    return pd.DataFrame(rows)


def render_prediction(result: dict[str, object]) -> None:
    probabilities = result["probabilities"]
    if not isinstance(probabilities, dict):
        st.json(result)
        return
    label = escape(str(result["label"]))
    version = escape(str(result["model_version"]))
    confidence = float(probabilities.get(result["label"], 0.0))
    st.markdown(
        f"""
        <div class="prediction-card">
          <div class="prediction-card__eyebrow">Production prediction</div>
          <div class="prediction-card__label">{label}</div>
          <div class="prediction-card__confidence">{confidence:.1%} confidence</div>
          <div class="prediction-card__version">Model: {version}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    columns = st.columns(len(probabilities))
    for column, (name, probability) in zip(columns, probabilities.items()):
        with column:
            st.metric(name.capitalize(), f"{float(probability):.1%}")
            st.progress(float(probability))


def main() -> None:
    st.set_page_config(page_title="MENA MLOps dashboard", page_icon="🧭")
    st.markdown(
        """
        <style>
        .prediction-card {
          background: linear-gradient(135deg, #102a43, #1f6f8b);
          border-radius: 18px;
          color: white;
          margin: 1rem 0 1.25rem;
          padding: 1.5rem 1.75rem;
        }
        .prediction-card__eyebrow {
          font-size: .78rem;
          letter-spacing: .08em;
          opacity: .78;
          text-transform: uppercase;
        }
        .prediction-card__label {
          font-size: 2.2rem;
          font-weight: 700;
          margin: .3rem 0;
        }
        .prediction-card__confidence { font-size: 1.05rem; }
        .prediction-card__version {
          font-size: .82rem;
          margin-top: .7rem;
          opacity: .78;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("MENA MLOps dashboard")
    st.caption("Inference is sent through the promoted production gateway.")

    with st.sidebar:
        st.subheader("Operational tools")
        for label, url in LINKS.items():
            st.link_button(f"Open {label}", url, use_container_width=True)
        st.caption(f"Gateway: `{API_URL}`")

    live_tab, batch_tab = st.tabs(["Live inference", "Batch inference"])
    with live_tab:
        text = st.text_area(
            "Arabic text",
            value="المنتج ممتاز والخدمة سريعة",
            max_chars=4_000,
        )
        if st.button("Predict", type="primary"):
            if not text.strip():
                st.error("Enter non-empty text.")
            else:
                try:
                    result = predict(text)
                except RuntimeError as error:
                    st.error(str(error))
                else:
                    render_prediction(result)

    with batch_tab:
        st.write("Upload a CSV with `text` required; `ID` and `source` are optional.")
        uploaded = st.file_uploader("Input CSV", type="csv")
        if uploaded is not None:
            try:
                frame = pd.read_csv(io.BytesIO(uploaded.getvalue()))
                frame = prepare_batch_frame(frame)
            except (ValueError, pd.errors.ParserError) as error:
                st.error(str(error))
            else:
                st.dataframe(frame.head(10), use_container_width=True)
                if st.button("Run batch inference", type="primary"):
                    try:
                        results = batch_predict(frame)
                    except RuntimeError as error:
                        st.error(str(error))
                    else:
                        st.success(f"Scored {len(results)} rows.")
                        st.dataframe(results, use_container_width=True)
                        st.download_button(
                            "Download predictions CSV",
                            results.to_csv(index=False).encode("utf-8"),
                            file_name="predictions.csv",
                            mime="text/csv",
                        )


if __name__ == "__main__":
    main()
