import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import Papa from "papaparse";
import "./styles.css";

const API_URL = (import.meta.env.VITE_MODEL_API_URL || "http://localhost:8080").replace(/\/$/, "");
const links = [
  ["Grafana", "http://localhost:3000"],
  ["Prometheus", "http://localhost:9090"],
  ["Feedback review", "http://localhost:8502"],
  ["MLflow", "http://localhost:5000"],
  ["Airflow", "http://localhost:8081"],
];

async function requestJson(path, options) {
  const response = await fetch(`${API_URL}${path}`, options);
  if (!response.ok) {
    throw new Error(`Gateway returned HTTP ${response.status}: ${await response.text()}`);
  }
  return response.json();
}

function predictionTone(label) {
  return { positive: "positive", negative: "negative", neutral: "neutral" }[label] || "neutral";
}

function PredictionCard({ result }) {
  if (!result) return null;
  const probabilities = result.probabilities || {};
  const label = result.label || "unknown";
  return (
    <section className={`prediction-shell ${predictionTone(label)}`}>
      <div className="prediction-hero">
        <span className="eyebrow">Production prediction</span>
        <h2>{label}</h2>
        <p>{(Number(probabilities[label] || 0) * 100).toFixed(1)}% confidence</p>
        <small>Model: {result.model_version}</small>
      </div>
      <div className="probability-grid">
        {Object.entries(probabilities).map(([name, value]) => (
          <div className={`probability ${name === label ? "selected" : ""}`} key={name}>
            <div className="probability-heading">
              <span>{name}</span><strong>{(Number(value) * 100).toFixed(1)}%</strong>
            </div>
            <div className="bar"><span style={{ width: `${Number(value) * 100}%` }} /></div>
          </div>
        ))}
      </div>
    </section>
  );
}

function App() {
  const [tab, setTab] = useState("live");
  const [text, setText] = useState("المنتج ممتاز والخدمة سريعة");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [batchFile, setBatchFile] = useState(null);
  const [reviewable, setReviewable] = useState(false);
  const [batchProgress, setBatchProgress] = useState(0);
  const [batchStatus, setBatchStatus] = useState("");
  const [batchResults, setBatchResults] = useState([]);

  const batchPreview = useMemo(() => batchResults.slice(0, 8), [batchResults]);

  async function runLive() {
    if (!text.trim()) {
      setError("Enter non-empty Arabic text.");
      return;
    }
    setLoading(true); setError(""); setResult(null);
    try {
      const response = await requestJson("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Source": "live" },
        body: JSON.stringify({ text, source: "live" }),
      });
      setResult(response);
    } catch (cause) {
      setError(cause.message);
    } finally {
      setLoading(false);
    }
  }

  async function runBatch() {
    if (!batchFile) return;
    setLoading(true); setError(""); setBatchResults([]); setBatchProgress(0);
    try {
      const rows = await parseCsv(batchFile);
      const predictions = [];
      const chunkSize = 32;
      for (let start = 0; start < rows.length; start += chunkSize) {
        const chunk = rows.slice(start, start + chunkSize);
        const response = await requestJson("/predict_batch", {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-Source": "batch" },
          body: JSON.stringify({
            texts: chunk.map((row) => row.text),
            source: "batch",
            input_ids: chunk.map((row, index) => row.ID || `row-${start + index + 1}`),
            feedback_policy: reviewable ? "reviewable" : "audit_only",
          }),
        });
        const batchPredictions = Array.isArray(response)
          ? response
          : response?.predictions;
        if (!Array.isArray(batchPredictions)) {
          throw new Error("Gateway returned an invalid batch prediction response.");
        }
        predictions.push(...batchPredictions.map((prediction, index) => ({
          ...chunk[index], ...prediction,
        })));
        setBatchProgress(Math.round((predictions.length / rows.length) * 100));
        setBatchStatus(`Scored ${predictions.length.toLocaleString()} / ${rows.length.toLocaleString()} rows`);
      }
      setBatchResults(predictions);
      setBatchStatus(`Completed ${predictions.length.toLocaleString()} rows`);
    } catch (cause) {
      setError(cause.message);
      setBatchStatus("");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-mark">M</div>
        <span className="eyebrow">MENA MLOps</span>
        <h1>Dashboard</h1>
        <p className="muted">Production inference workspace</p>
        <nav>{links.map(([name, href]) => <a href={href} target="_blank" rel="noreferrer" key={name}>{name}<span>↗</span></a>)}</nav>
        <div className="gateway">Gateway <code>{API_URL}</code></div>
      </aside>
      <main className="content">
        <header className="page-header">
          <div><span className="eyebrow">Operational console</span><h1>MENA sentiment intelligence</h1><p className="muted">Inference is sent through the promoted production gateway.</p></div>
          <div className="status-pill"><i /> Production online</div>
        </header>
        <div className="tabs"><button className={tab === "live" ? "active" : ""} onClick={() => setTab("live")}>Live inference</button><button className={tab === "batch" ? "active" : ""} onClick={() => setTab("batch")}>Batch inference</button></div>
        {error && <div className="alert">{error}</div>}
        {tab === "live" ? (
          <section className="workspace">
            <div className="panel input-panel"><span className="eyebrow">Interactive prediction</span><h2>Analyze Arabic feedback</h2><textarea value={text} onChange={(event) => setText(event.target.value)} maxLength={4000} /><button className="primary" onClick={runLive} disabled={loading}>{loading ? "Analyzing..." : "Run prediction"}<span>→</span></button><p className="hint">Predictions are recorded with model version and confidence for auditability.</p></div>
            <PredictionCard result={result} />
          </section>
        ) : (
          <section className="workspace batch-workspace">
            <div className="panel input-panel"><span className="eyebrow">Vectorized scoring</span><h2>Score a feedback file</h2><p className="muted">Upload a CSV with a required <code>text</code> column. ID and source columns are optional.</p><label className="dropzone"><input type="file" accept=".csv,text/csv" onChange={(event) => setBatchFile(event.target.files?.[0] || null)} /><strong>{batchFile ? batchFile.name : "Choose a CSV file"}</strong><span>{batchFile ? `${(batchFile.size / 1024 / 1024).toFixed(1)} MB selected` : "Drag and drop or browse"}</span></label><label className="check-row"><input type="checkbox" checked={reviewable} onChange={(event) => setReviewable(event.target.checked)} /> Send uncertain predictions to the review queue</label>{batchFile && <button className="primary" onClick={runBatch} disabled={loading}>{loading ? "Scoring..." : "Run batch inference"}<span>→</span></button>}{loading && <div className="progress-wrap"><div className="progress-heading"><span>{batchStatus || "Preparing batch..."}</span><strong>{batchProgress}%</strong></div><div className="progress-track"><span style={{ width: `${batchProgress}%` }} /></div></div>}{batchStatus && !loading && <div className="success">{batchStatus}</div>}</div>
            {batchResults.length > 0 && <div className="panel results-panel"><div className="panel-heading"><div><span className="eyebrow">Results</span><h2>Prediction preview</h2></div><button className="secondary" onClick={() => downloadResults(batchResults)}>Download CSV</button></div><p className="muted">Showing {batchPreview.length} of {batchResults.length.toLocaleString()} scored rows.</p><div className="table-wrap"><table><thead><tr><th>Input</th><th>Prediction</th><th>Confidence</th><th>Model</th></tr></thead><tbody>{batchPreview.map((row, index) => <tr key={row.ID || index}><td>{row.text}</td><td><span className={`label-chip ${predictionTone(row.label)}`}>{row.label}</span></td><td>{(Number(row.probabilities?.[row.label] || 0) * 100).toFixed(1)}%</td><td>{row.model_version}</td></tr>)}</tbody></table></div></div>}
          </section>
        )}
      </main>
    </div>
  );
}

async function parseCsv(file) {
  const parsed = Papa.parse(await file.text(), {
    header: true,
    skipEmptyLines: true,
  });
  if (parsed.errors.length > 0) {
    throw new Error(`CSV could not be parsed: ${parsed.errors[0].message}`);
  }
  if (!parsed.meta.fields?.includes("text")) {
    throw new Error("CSV must contain a 'text' column.");
  }
  return parsed.data.map((row) => {
    const normalized = Object.fromEntries(
      Object.entries(row).map(([key, value]) => [key, String(value ?? "").trim()]),
    );
    if (!normalized.text) throw new Error("CSV contains an empty text value.");
    return normalized;
  });
}

function downloadResults(rows) {
  const headers = ["ID", "text", "label", "model_version", "probabilities"];
  const csv = [headers.join(","), ...rows.map((row) => headers.map((header) => JSON.stringify(row[header] ?? "")).join(","))].join("\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  const anchor = document.createElement("a"); anchor.href = url; anchor.download = "predictions.csv"; anchor.click(); URL.revokeObjectURL(url);
}

createRoot(document.getElementById("root")).render(<App />);
