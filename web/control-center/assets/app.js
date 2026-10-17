const state = { overview: null, releases: [], deployments: null, selectedRelease: null };

const $ = (selector) => document.querySelector(selector);
const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;").replaceAll('"', "&quot;");
const statusClass = (value) => `status status-${String(value || "unknown").replaceAll(" ", "_")}`;
const statusBadge = (value) => `<span class="${statusClass(value)}">${escapeHtml(value || "unknown")}</span>`;
const productionBadge = () => '<span class="production-badge">ON PROD</span>';
const formatDate = (value) => value ? new Date(value).toLocaleString() : "—";
const latencyMetric = (latency, key) => latency?.[key] == null
  ? "Not recorded"
  : `${Number(latency[key]).toFixed(2)} ms`;

async function getJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
  return response.json();
}

function renderSummary() {
  const latest = state.overview.latest_release;
  const current = state.deployments?.current;
  const feedback = state.overview.feedback || {};
  const counts = feedback.counts || {};
  const thresholds = feedback.thresholds || {};
  const feedbackText = Object.keys(thresholds).map(
    (label) => `${label}: ${counts[label] || 0}/${thresholds[label]}`
  ).join(" · ");
  $("#summary").innerHTML = [
    ["Latest release", latest ? statusBadge(latest.status) : "No releases", latest?.release_id || "Waiting for first run"],
    ["Production", current ? `${productionBadge()} ${escapeHtml(current.model_id || "Active model")}` : "No baseline", current ? `Variant: ${current.selected_variant || "unknown"}` : "No deployed baseline recorded"],
    ["Feedback readiness", feedback.ready ? "READY" : "Collecting", feedbackText || "Feedback database unavailable"],
    ["Recorded releases", state.overview.release_count || 0, "Durable lifecycle history"],
  ].map(([label, value, detail]) => `<article class="card"><div class="label">${label}</div><div class="value">${value}</div><div class="detail">${escapeHtml(detail)}</div></article>`).join("");
}

function renderOverview() {
  const services = state.overview.services || {};
  const serviceHtml = Object.entries(services).map(([name, value]) =>
    `<div class="service"><span>${escapeHtml(name)}</span><span class="dot ${value.status === "healthy" ? "healthy" : ""}" title="${escapeHtml(value.detail)}"></span></div>`
  ).join("");
  const latest = state.overview.latest_release;
  $("#view-overview").innerHTML = `
    <div class="two-col">
      <article class="panel">
        <h2>Latest retraining outcome</h2>
        ${latest ? `<div class="release-row" onclick="showRelease('${escapeHtml(latest.release_id)}')">
          <div><strong>${escapeHtml(latest.release_id)}</strong><div class="muted">${escapeHtml(latest.trigger)} trigger · ${formatDate(latest.started_at)}</div></div>
          <div>${statusBadge(latest.status)}</div><div>${statusBadge(latest.deployment_status)}</div><div>View details →</div>
        </div>` : `<p class="empty">No release has been recorded yet.</p>`}
      </article>
      ${state.deployments?.current ? `<article class="panel production-panel">
        <h2>Active production baseline</h2>
        <div class="production-heading">${productionBadge()}<strong>${escapeHtml(state.deployments.current.model_id || "Active model")}</strong></div>
        <p class="muted">This baseline is active in serving but is not linked to a recorded release lifecycle entry.</p>
        <p>Variant: <strong>${escapeHtml(state.deployments.current.selected_variant || "unknown")}</strong> · Macro-F1: <strong>${Number(state.deployments.current.quality?.macro_f1 || 0).toFixed(4)}</strong></p>
        <p class="muted">Latency p95: ${latencyMetric(state.deployments.current.latency, "p95_ms")}</p>
      </article>` : ""}
      <article class="panel">
        <h2>Service health</h2>
        <div class="service-list">${serviceHtml || '<p class="empty">No health data.</p>'}</div>
      </article>
    </div>
    <article class="panel">
      <h2>How to read a release</h2>
      <p class="muted">A rejected candidate is a successful safety outcome: the model was evaluated, the failing gate explains why it was not promoted, and production remains unchanged. A deployment failure is different: all model gates passed, but serving or canary verification failed.</p>
    </article>`;
}

function renderFeedback() {
  const feedback = state.overview.feedback || {};
  const counts = feedback.counts || {};
  const deploymentCounts = feedback.deployment_counts || {};
  const thresholds = feedback.thresholds || {};
  const annotation = feedback.annotation || {};
  const labels = Object.keys(thresholds);
  const chart = (parts, colors) => {
    const total = parts.reduce((sum, part) => sum + part.value, 0);
    if (!total) return '<div class="donut donut-empty"><span>No data</span></div>';
    let offset = 0;
    const stops = parts.map((part, index) => {
      const start = offset / total * 100; offset += part.value;
      return `${colors[index]} ${start}% ${offset / total * 100}%`;
    }).join(", ");
    return `<div class="donut" style="background:conic-gradient(${stops})"><span>${total}</span></div>`;
  };
  const legend = (parts, colors) => parts.map((part, index) =>
    `<span class="legend-item"><i style="background:${colors[index]}"></i>${escapeHtml(part.label)}: ${part.value}</span>`
  ).join("");
  const classCards = labels.map((label) => {
    const values = annotation[label] || {};
    const total = Number(values.total_predictions || 0);
    const suspected = Number(values.suspected || 0);
    const reviewed = Number(values.reviewed || 0);
    const confusing = Number(values.confusing || 0);
    const pending = Number(values.pending || 0);
    const queueTotal = Math.max(suspected, 1);
    const segments = [
      ["reviewed", reviewed, "var(--accent)"],
      ["confusing", confusing, "var(--warning)"],
      ["pending", pending, "var(--danger)"],
    ];
    return `<article class="feedback-class panel">
      <div class="feedback-class-heading"><h3>${escapeHtml(label)}</h3>
        <span class="muted">${suspected} suspected / ${total} predictions</span></div>
      <div class="chart-row">
        <div><div class="chart-title">Queue share</div>${chart([
          {label: "Suspected", value: suspected},
          {label: "Other", value: Math.max(total - suspected, 0)},
        ], ["var(--warning)", "var(--border)"])}
        <div class="legend">${legend([
          {label: "Suspected", value: suspected},
          {label: "Other", value: Math.max(total - suspected, 0)},
        ], ["var(--warning)", "var(--border)"])}</div></div>
        <div><div class="chart-title">Annotation state</div>${chart([
          {label: "Reviewed", value: reviewed},
          {label: "Confusing", value: confusing},
          {label: "Pending", value: pending},
        ], ["var(--accent)", "var(--warning)", "var(--danger)"])}
        <div class="legend">${legend([
          {label: "Reviewed", value: reviewed},
          {label: "Confusing", value: confusing},
          {label: "Pending", value: pending},
        ], ["var(--accent)", "var(--warning)", "var(--danger)"])}</div></div>
      </div>
      <div class="feedback-progress" aria-label="${escapeHtml(label)} annotation progress">
        ${segments.map(([name, value, color]) => `<span class="progress-${name}" style="width:${value / queueTotal * 100}%;background:${color}" title="${name}: ${value}"></span>`).join("")}
      </div>
      <div class="progress-caption"><span>${reviewed + confusing} of ${suspected} triaged</span><span>${counts[label] || 0} / ${thresholds[label]} validated for retraining</span></div>
    </article>`;
  }).join("");
  const readinessDescription = "This is progress since the last successful retraining checkpoint, not lifetime feedback. Each validated example moves its class toward the next scheduled or manually triggered Airflow retraining run. The remaining number is how many more validated examples that class needs.";
  $("#view-feedback").innerHTML = `<article class="panel feedback-guidance"><h2>How feedback moves the model forward</h2>
    <p>Every prediction is recorded by the inference service. Low-confidence predictions enter the suspected queue for human review. Reviewed labels count toward retraining; confusing/deferred items remain visible but do not count toward the retraining gate.</p>
    <div class="feedback-legend"><span><i class="legend-dot reviewed"></i>Reviewed</span><span><i class="legend-dot confusing"></i>Confusing / deferred</span><span><i class="legend-dot pending"></i>Pending</span></div>
  </article><article class="panel"><h2>Feedback retraining readiness <span class="info-tip" tabindex="0" title="${escapeHtml(readinessDescription)}" aria-label="${escapeHtml(readinessDescription)}">i</span></h2>
    <p class="muted">Next trigger progress is measured since the last completed retraining attempt. Feedback still awaiting a successful deployment: ${labels.map((label) => `${escapeHtml(label)} ${deploymentCounts[label] || 0}`).join(" · ")}.</p>
    ${Object.keys(thresholds).map((label) => {
      const count = counts[label] || 0; const threshold = thresholds[label] || 1;
      const remaining = Math.max(0, threshold - count);
      const status = remaining ? `${remaining} remaining` : "0 remaining · threshold met";
      return `<div style="margin: 16px 0"><div class="readiness-line"><span>${escapeHtml(label)}</span><span><strong>${status}</strong> <span class="muted">(${count} validated)</span></span></div><div class="progress"><span style="width:${Math.min(100, count / threshold * 100)}%"></span></div></div>`;
    }).join("")}
    <p>${feedback.ready ? '<span class="status status-approved">Ready to retrain</span>' : '<span class="status status-training">Collecting validated feedback</span>'}</p>
    <a class="button-link" href="${escapeHtml(state.overview.links?.feedback || "/reviewer")}" target="_blank">Open reviewer interface →</a>
  </article><section class="feedback-grid">${classCards}</section>`;
}

async function renderDeployments() {
  const data = await getJson("/api/deployments");
  state.deployments = data;
  const current = data.current;
  $("#view-deployments").innerHTML = `<article class="panel"><h2>Current deployment</h2>
    ${current ? `<div class="two-col"><div><p class="eyebrow">Active model</p><div class="production-heading">${productionBadge()}<h2>${escapeHtml(current.model_id || "unknown")}</h2></div><p class="muted">Variant: ${escapeHtml(current.selected_variant || "unknown")}</p><p class="muted">The active baseline is not linked to a release record yet.</p></div><div><p class="eyebrow">Performance baseline</p><p>Macro-F1: <strong>${Number(current.quality?.macro_f1 || 0).toFixed(4)}</strong></p><p>Latency p50: <strong>${latencyMetric(current.latency, "p50_ms")}</strong></p><p>Latency p95: <strong>${latencyMetric(current.latency, "p95_ms")}</strong></p><p>Throughput: <strong>${Number(current.latency?.throughput_rps || 0).toFixed(2)} req/s</strong></p></div></div>` : '<p class="empty">No deployed baseline is recorded.</p>'}
  </article><article class="panel"><h2>Release deployment history</h2>
    ${data.history.length ? data.history.map((release) => `<div class="release-row" onclick="showRelease('${escapeHtml(release.release_id)}')"><div><strong>${escapeHtml(release.release_id)}</strong><div class="muted">${formatDate(release.completed_at)}</div></div><div>${statusBadge(release.status)}</div><div>${statusBadge(release.deployment_status)}</div><div>View →</div></div>`).join("") : '<p class="empty">No deployment attempts recorded.</p>'}</article>`;
}

async function renderModels() {
  const data = await getJson("/api/models");
  const current = data.current;
  const mlflow = data.mlflow || {};
  const latestRun = mlflow.latest_run || {};
  const runInfo = latestRun.info || {};
  const mlflowPanel = mlflow.status === "available"
    ? `<article class="panel"><h2>Latest MLflow experiment</h2>
      <p><strong>${escapeHtml(mlflow.experiment_name)}</strong></p>
      <p class="muted">Latest run: ${escapeHtml(runInfo.run_name || runInfo.run_id || "unnamed")}</p>
      <p>Status: <strong>${escapeHtml(runInfo.status || "unknown")}</strong></p>
      <p><a class="button-link" href="${escapeHtml(`${data.links.mlflow}/#/experiments/${mlflow.experiment_id}`)}" target="_blank" rel="noreferrer">Open experiment →</a>
      ${runInfo.run_id ? `<a class="button-link" href="${escapeHtml(`${data.links.mlflow}/#/experiments/${mlflow.experiment_id}/runs/${runInfo.run_id}`)}" target="_blank" rel="noreferrer">Open latest run →</a>` : ""}</p>
    </article>`
    : `<article class="panel"><h2>Latest MLflow experiment</h2><p class="empty">${escapeHtml(mlflow.detail || "MLflow experiment is unavailable.")}</p></article>`;
  const links = {
    ...(data.links || {}),
    ...(current?.model_card_uri ? { model_card: current.model_card_uri } : {}),
    ...(current?.release_uri ? { release_bundle: current.release_uri } : {}),
    ...(current?.promotion_decision_uri ? { promotion_decision: current.promotion_decision_uri } : {}),
  };
  const linkRows = Object.entries(links).map(([name, url]) =>
    `<tr><td>${escapeHtml(name.replaceAll("_", " "))}</td><td><a class="button-link" href="${escapeHtml(url)}" target="_blank" rel="noreferrer">Open →</a></td></tr>`
  ).join("");
  $("#view-models").innerHTML = `${mlflowPanel}<article class="panel"><h2>Model identity</h2>
    ${current ? `<table><tbody><tr><th>Model ID</th><td>${escapeHtml(current.model_id)}</td></tr><tr><th>Variant</th><td>${escapeHtml(current.selected_variant)}</td></tr><tr><th>Backend</th><td>${escapeHtml(current.hardware?.backend)} / ${escapeHtml(current.hardware?.provider)}</td></tr><tr><th>Macro-F1</th><td>${Number(current.quality?.macro_f1 || 0).toFixed(4)}</td></tr><tr><th>Latency p50</th><td>${latencyMetric(current.latency, "p50_ms")}</td></tr><tr><th>Latency p95</th><td>${latencyMetric(current.latency, "p95_ms")}</td></tr><tr><th>Throughput</th><td>${Number(current.latency?.throughput_rps || 0).toFixed(2)} req/s</td></tr></tbody></table>` : '<p class="empty">No model identity baseline found.</p>'}
  </article><article class="panel"><h2>Model and platform links</h2><table><thead><tr><th>Resource</th><th>Open</th></tr></thead><tbody>${linkRows || '<tr><td colspan="2" class="empty">No external links configured.</td></tr>'}</tbody></table></article><article class="panel"><h2>Candidate history</h2><p class="muted">Use Retraining & releases to inspect every gate and its evidence. This registry currently contains ${data.candidates.length} release record(s).</p></article>`;
}

function renderLinks() {
  const links = state.overview.links || {};
  $("#view-links").innerHTML = `<article class="panel"><h2>Specialist tools</h2><table><thead><tr><th>Tool</th><th>Purpose</th><th>Open</th></tr></thead><tbody>
    ${[
      ["airflow", "DAG orchestration, task logs, and manual triggers"],
      ["grafana", "Detailed runtime time-series and canary metrics"],
      ["mlflow", "Experiment runs, parameters, and artifacts"],
      ["prometheus", "Raw metrics and PromQL exploration"],
      ["feedback", "Human review of uncertain predictions"],
    ].map(([name, purpose]) => `<tr><td>${name}</td><td class="muted">${purpose}</td><td><a class="button-link" href="${escapeHtml(links[name] || "#")}" target="_blank">Open →</a></td></tr>`).join("")}
  </tbody></table></article>`;
}

function renderReleaseList() {
  $("#view-releases").innerHTML = `<article class="panel"><h2>Retraining and release history</h2>
    ${state.deployments?.current ? `<div class="release-row production-row">
      <div><strong>${escapeHtml(state.deployments.current.model_id || "Active production baseline")}</strong><div class="muted">Imported production baseline · current serving model</div></div>
      <div>${productionBadge()}</div><div>${escapeHtml(state.deployments.current.selected_variant || "—")}</div><div>${statusBadge("deployed")}</div>
    </div>` : ""}
    ${state.releases.length ? state.releases.map((release) => `<div class="release-row" onclick="showRelease('${escapeHtml(release.release_id)}')">
      <div><strong>${escapeHtml(release.release_id)}</strong><div class="muted">${escapeHtml(release.trigger)} · ${formatDate(release.started_at)}</div></div>
      <div>${statusBadge(release.status)}</div><div>${escapeHtml(release.candidate_variant || "—")}</div><div>${statusBadge(release.deployment_status)}</div>
    </div>`).join("") : '<p class="empty">No recorded retraining releases.</p>'}</article>`;
}

async function showRelease(releaseId) {
  try {
    const release = await getJson(`/api/releases/${encodeURIComponent(releaseId)}`);
    state.selectedRelease = release;
    const gateHtml = (release.gates || []).map((gate) => `<div class="gate ${gate.accepted ? "" : "failed"}">
      <div>${gate.accepted ? "✓" : "!"}</div><div><h3>${escapeHtml(gate.gate_name.replaceAll("_", " "))} ${statusBadge(gate.accepted ? "passed" : "failed")}</h3>
      ${(gate.failures || []).map((failure) => `<p>${escapeHtml(failure)}</p>`).join("") || '<p>All checks passed.</p>'}</div>
    </div>`).join("");
    const events = (release.events || []).map((event) => `<div class="timeline-item ${event.status === "failed" ? "failed" : ""}"><strong>${escapeHtml(event.event_type.replaceAll("_", " "))}</strong><small>${formatDate(event.timestamp)} · ${escapeHtml(event.status || "recorded")}</small></div>`).join("");
    $("#view-releases").innerHTML = `<article class="panel"><button class="button button-secondary" onclick="renderReleaseList()">← All releases</button><h2 style="margin-top:18px">${escapeHtml(release.release_id)}</h2>
      <p>${release.deployment_status === "deployed" ? productionBadge() : ""} ${statusBadge(release.status)} &nbsp; ${statusBadge(release.deployment_status)} &nbsp; <span class="muted">${escapeHtml(release.trigger)} trigger · ${escapeHtml(release.candidate_variant || "variant pending")}</span></p>
      <div class="two-col"><div><h2>Gate matrix</h2><div class="gate-list">${gateHtml || '<p class="empty">Gate results are not available yet.</p>'}</div></div><div><h2>Lifecycle</h2><div class="timeline">${events || '<p class="empty">No lifecycle events recorded.</p>'}</div></div></div>
      <details><summary>Decision evidence</summary><pre>${escapeHtml(JSON.stringify(release.decision || {}, null, 2))}</pre></details>
    </article>`;
    document.querySelector('[data-view="releases"]').click();
  } catch (error) { showError(error); }
}

function showError(error) {
  $("#alerts").innerHTML = `<div class="alert">Dashboard data could not be loaded: ${escapeHtml(error.message)}</div>`;
}

async function refresh() {
  try {
    state.overview = await getJson("/api/overview");
    state.releases = await getJson("/api/releases");
    state.deployments = await getJson("/api/deployments");
    renderSummary(); renderOverview(); renderFeedback(); renderLinks(); renderReleaseList();
    await renderDeployments(); await renderModels();
    $("#updated").textContent = `Updated ${new Date().toLocaleTimeString()}`;
    $("#alerts").innerHTML = "";
  } catch (error) { showError(error); }
}

document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
  document.querySelectorAll(".view").forEach((item) => item.classList.add("hidden"));
  tab.classList.add("active"); $(`#view-${tab.dataset.view}`).classList.remove("hidden");
}));
$("#refresh").addEventListener("click", refresh);
window.showRelease = showRelease;
window.renderReleaseList = renderReleaseList;
refresh();
setInterval(refresh, 15000);
