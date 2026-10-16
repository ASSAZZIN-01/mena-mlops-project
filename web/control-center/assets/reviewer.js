const $ = (selector) => document.querySelector(selector);
const labels = ["negative", "neutral", "positive"];
const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;").replaceAll('"', "&quot;");

async function getJson(path, options = {}) {
  const response = await fetch(path, {
    headers: {"Content-Type": "application/json"},
    ...options,
  });
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
  return response.json();
}

function showError(error) {
  $("#reviewer-alerts").innerHTML = `<div class="alert">${escapeHtml(error.message)}</div>`;
}

function renderSummary(summary) {
  const annotation = summary.annotation || {};
  const counts = summary.counts || {};
  const thresholds = summary.thresholds || {};
  const pending = labels.reduce((total, label) => total + Number(annotation[label]?.pending || 0), 0);
  const reviewed = labels.reduce((total, label) => total + Number(annotation[label]?.reviewed || 0), 0);
  $("#reviewer-summary").innerHTML = [
    ["Pending queue", pending, "Items waiting for a human decision"],
    ["Reviewed", reviewed, "Validated labels saved"],
    ["Next retraining progress", labels.map((label) => `${label}: ${counts[label] || 0}/${thresholds[label] || 0}`).join(" · "), "Validated since the last successful retraining"],
    ["Retraining gate", summary.ready ? "Ready" : "Collecting", "Validated labels only"],
  ].map(([label, value, detail]) =>
    `<article class="card"><div class="label">${escapeHtml(label)}</div><div class="value">${escapeHtml(value)}</div><div class="detail">${escapeHtml(detail)}</div></article>`
  ).join("");
}

function renderClasses(summary) {
  const annotation = summary.annotation || {};
  $("#reviewer-classes").innerHTML = labels.map((label) => {
    const values = annotation[label] || {};
    const total = Number(values.suspected || 0);
    const done = Number(values.reviewed || 0) + Number(values.confusing || 0);
    return `<div class="review-class">
      <div><strong>${escapeHtml(label)}</strong><span class="muted">${done} / ${total} triaged</span></div>
      <div class="progress"><span style="width:${total ? Math.min(100, done / total * 100) : 0}%"></span></div>
      <small class="muted">${values.pending || 0} pending · ${values.confusing || 0} confusing</small>
    </div>`;
  }).join("");
}

function renderItem(data) {
  renderSummary(data.summary);
  renderClasses(data.summary);
  const item = data.item;
  if (!item) {
    $("#reviewer-item").innerHTML = `<div class="empty-state"><h3>Queue complete</h3><p class="muted">There are no pending suspected predictions to review.</p></div>`;
    return;
  }
  const classSummary = data.class_summary || {};
  $("#reviewer-item").innerHTML = `
    <div class="review-meta">
      <span class="status status-training">Queue ${data.queue_position} of ${data.queue_total}</span>
      <span class="muted">Review ID ${item.id}</span>
    </div>
    <blockquote>${escapeHtml(item.text)}</blockquote>
    <div class="review-facts">
      <div><span class="label">Model prediction</span><strong>${escapeHtml(item.predicted_label)}</strong></div>
      <div><span class="label">Confidence</span><strong>${(Number(item.confidence) * 100).toFixed(1)}%</strong></div>
      <div><span class="label">${escapeHtml(item.predicted_label)} pending</span><strong>${classSummary.pending || 0}</strong></div>
    </div>
    <div class="decision-panel">
      <p class="label">Is this prediction correct?</p>
      <div class="decision-actions">
        <button class="button decision-button button-primary" data-decision="correct">Correct</button>
        <button class="button decision-button button-secondary" data-decision="wrong">Wrong</button>
        <button class="button decision-button button-secondary" data-decision="defer">I don't know</button>
      </div>
      <div id="correction" class="hidden correction-field">
        <label class="label" for="corrected-label">Correct sentiment</label>
        <select id="corrected-label">${labels.map((label) => `<option value="${label}">${label}</option>`).join("")}</select>
        <button id="save-correction" class="button button-primary">Save corrected label</button>
      </div>
    </div>`;
  document.querySelectorAll(".decision-button").forEach((button) => {
    button.addEventListener("click", () => {
      if (button.dataset.decision === "wrong") {
        $("#correction").classList.remove("hidden");
      } else {
        submit(button.dataset.decision, item.id);
      }
    });
  });
  $("#save-correction").addEventListener("click", () =>
    submit("wrong", item.id, $("#corrected-label").value)
  );
}

async function submit(decision, itemId, correctedLabel = null) {
  try {
    $("#reviewer-alerts").innerHTML = "";
    renderItem(await getJson("/api/reviewer/submit", {
      method: "POST",
      body: JSON.stringify({item_id: itemId, decision, corrected_label: correctedLabel}),
    }));
  } catch (error) {
    showError(error);
  }
}

async function refresh() {
  try {
    renderItem(await getJson("/api/reviewer/next"));
  } catch (error) {
    showError(error);
  }
}

$("#reviewer-refresh").addEventListener("click", refresh);
refresh();
