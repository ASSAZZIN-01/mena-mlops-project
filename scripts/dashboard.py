"""Reviewer dashboard for live and batch sentiment inference."""

from __future__ import annotations

import io
import json
import os
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
    if frame["text"].isna().any() or (frame["text"].astype(str).str.len() == 0).any():
        raise ValueError("CSV contains an empty or null text value")


def batch_predict(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    progress = st.progress(0, text="Starting batch inference")
    total = len(frame)
    for index, record in enumerate(frame.to_dict(orient="records"), start=1):
        result = predict(str(record["text"]))
        rows.append({**record, **result})
        progress.progress(index / total, text=f"Scored {index}/{total} rows")
    progress.empty()
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="MENA MLOps dashboard", page_icon="🧭")
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
                    st.success(f"Prediction: {result['label']}")
                    st.metric("Model version", str(result["model_version"]))
                    st.json(result["probabilities"])

    with batch_tab:
        st.write("Upload a CSV with a required `text` column.")
        uploaded = st.file_uploader("Input CSV", type="csv")
        if uploaded is not None:
            try:
                frame = pd.read_csv(io.BytesIO(uploaded.getvalue()))
                validate_frame(frame)
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
