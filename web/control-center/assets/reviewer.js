const $ = (selector) => document.querySelector(selector);
const labels = ["negative", "neutral", "positive"];
const state = {
  history: [],
  pendingItems: [],
  nextItem: null,
  current: null,
  currentIsHistory: false,
  editing: false,
  selectedDecision: null,
};
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

function renderProgress(summary) {
  const counts = summary.counts || {};
  const thresholds = summary.thresholds || {};
  $("#reviewer-progress").innerHTML = `<div class="progress-heading">
    <div><p class="eyebrow">Feedback retraining readiness</p><h2>Progress toward the next trigger</h2></div>
    <span class="muted">${summary.ready ? "Threshold reached" : "Validated reviews still needed"}</span>
  </div>
  <div class="reviewer-progress-grid">${labels.map((label) => {
    const count = Number(counts[label] || 0);
    const threshold = Number(thresholds[label] || 1);
    const remaining = Math.max(0, threshold - count);
    return `<div class="reviewer-progress-item">
      <div><strong>${escapeHtml(label)}</strong><span>${remaining} remaining</span></div>
      <div class="progress"><span style="width:${Math.min(100, count / threshold * 100)}%"></span></div>
      <small class="muted">${count} / ${threshold} validated since the last attempt</small>
    </div>`;
  }).join("")}</div>`;
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

function renderNavigation() {
  const index = state.currentIsHistory
    ? state.history.findIndex((item) => item.id === state.current.id)
    : state.pendingItems.findIndex((item) => item.id === state.current.id);
  const navigationBlocked = state.editing;
  const previousDisabled = navigationBlocked || (
    state.currentIsHistory
      ? index <= 0
      : index <= 0 && state.history.length === 0
  );
  const nextDisabled = navigationBlocked || (
    state.currentIsHistory
      ? index >= state.history.length - 1 && state.pendingItems.length === 0
      : index >= state.pendingItems.length - 1
  );
  return `<div class="review-navigation">
    <button id="previous-item" class="button button-secondary" ${previousDisabled ? "disabled" : ""}>← Previous</button>
    <span class="muted">${state.currentIsHistory ? `Reviewed item ${index + 1} of ${state.history.length}` : "Current pending item"}</span>
    <button id="next-item" class="button button-secondary" ${nextDisabled ? "disabled" : ""}>Next →</button>
  </div>`;
}

function renderDecisionButtons(item) {
  const readOnly = state.currentIsHistory && !state.editing;
  const selected = state.selectedDecision;
  const selectedLabel = item.review_status === "validated"
    ? (item.reviewed_label === item.predicted_label ? "Correct" : `Wrong → ${item.reviewed_label}`)
    : "I don't know";
  if (readOnly) {
    return `<div class="reviewed-choice"><span class="decision-pill decision-${item.review_status}">${escapeHtml(selectedLabel)}</span>
      <button id="fix-item" class="button button-secondary">Fix it</button></div>`;
  }
  return `<p class="label">Is this prediction correct?</p>
    <div class="decision-actions">
      <button class="button decision-button decision-correct ${selected === "correct" ? "selected" : ""}" data-decision="correct">Correct</button>
      <button class="button decision-button decision-wrong ${selected === "wrong" ? "selected" : ""}" data-decision="wrong">Wrong</button>
      <button class="button decision-button decision-defer ${selected === "defer" ? "selected" : ""}" data-decision="defer">I don't know</button>
    </div>
    ${selected === "wrong" ? `<div class="correction-field">
      <label class="label" for="corrected-label">Correct sentiment</label>
      <select id="corrected-label">${labels.map((label) => `<option value="${label}" ${label === item.reviewed_label ? "selected" : ""}>${label}</option>`).join("")}</select>
    </div>` : ""}
    <button id="confirm-review" class="button button-primary" ${selected ? "" : "disabled"}>Confirm review</button>`;
}

function renderItem(data) {
  renderProgress(data.summary);
  renderClasses(data.summary);
  const item = state.current;
  if (!item) {
    $("#reviewer-item").innerHTML = `<div class="empty-state"><h3>Queue complete</h3><p class="muted">There are no pending suspected predictions to review.</p></div>`;
    return;
  }
  const classSummary = data.class_summary || {};
  const queueText = state.currentIsHistory
    ? `Reviewed item ${state.history.findIndex((entry) => entry.id === item.id) + 1} of ${state.history.length}`
    : `Queue ${state.pendingItems.findIndex((entry) => entry.id === item.id) + 1} of ${state.pendingItems.length}`;
  $("#reviewer-item").innerHTML = `
    ${renderNavigation()}
    <div class="review-meta"><span class="status ${state.currentIsHistory ? "status-approved" : "status-training"}">${queueText}</span><span class="muted">Review ID ${item.id}</span></div>
    <blockquote>${escapeHtml(item.text)}</blockquote>
    <div class="review-facts">
      <div><span class="label">Model prediction</span><strong>${escapeHtml(item.predicted_label)}</strong></div>
      <div><span class="label">Confidence</span><strong>${(Number(item.confidence) * 100).toFixed(1)}%</strong></div>
      <div><span class="label">${escapeHtml(item.predicted_label)} pending</span><strong>${classSummary.pending || 0}</strong></div>
    </div>
    <div class="decision-panel">${renderDecisionButtons(item)}</div>`;
  $("#previous-item").addEventListener("click", () => {
    if (state.currentIsHistory) {
      showHistory(state.history[state.history.findIndex((entry) => entry.id === item.id) - 1]);
    } else {
      const index = state.pendingItems.findIndex((entry) => entry.id === item.id);
      if (index > 0) showPending(state.pendingItems[index - 1]);
      else showHistory(state.history[state.history.length - 1]);
    }
  });
  $("#next-item").addEventListener("click", () => {
    if (state.currentIsHistory) {
      const index = state.history.findIndex((entry) => entry.id === item.id);
      if (index < state.history.length - 1) showHistory(state.history[index + 1]);
      else showPending(state.pendingItems[0]);
    } else {
      const index = state.pendingItems.findIndex((entry) => entry.id === item.id);
      showPending(state.pendingItems[index + 1]);
    }
  });
  $("#fix-item")?.addEventListener("click", () => {
    state.editing = true;
    state.selectedDecision = null;
    renderItem(data);
  });
  document.querySelectorAll(".decision-button").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedDecision = button.dataset.decision;
      renderItem(data);
    });
  });
  $("#confirm-review")?.addEventListener("click", () => {
    const corrected = $("#corrected-label")?.value || null;
    submit(state.selectedDecision, item.id, corrected, state.currentIsHistory);
  });
}

function showHistory(item) {
  if (!item) return;
  state.current = item;
  state.currentIsHistory = true;
  state.editing = false;
  state.selectedDecision = null;
  renderItem({
    item,
    queue_total: state.nextItem ? 1 : 0,
    summary: window.reviewerSummary,
    class_summary: {},
  });
}

function showPending(item) {
  if (!item) return;
  state.current = item;
  state.currentIsHistory = false;
  state.editing = false;
  state.selectedDecision = null;
  renderItem({
    item,
    queue_total: state.pendingItems.length,
    summary: window.reviewerSummary,
    class_summary: {},
  });
}

async function submit(decision, itemId, correctedLabel, revise) {
  try {
    $("#reviewer-alerts").innerHTML = "";
    const result = await getJson("/api/reviewer/submit", {
      method: "POST",
      body: JSON.stringify({
        item_id: itemId,
        decision,
        corrected_label: correctedLabel,
        revise,
      }),
    });
    const completed = result.completed_item;
    const existing = state.history.findIndex((item) => item.id === completed.id);
    if (existing >= 0) state.history[existing] = completed;
    else state.history.push(completed);
    state.pendingItems = result.pending_items;
    state.nextItem = result.item;
    state.current = result.item;
    state.currentIsHistory = false;
    state.editing = false;
    state.selectedDecision = null;
    window.reviewerSummary = result.summary;
    renderItem(result);
  } catch (error) {
    showError(error);
  }
}

async function refresh() {
  try {
    const result = await getJson("/api/reviewer/next");
    window.reviewerSummary = result.summary;
    state.history = result.reviewed_items;
    state.pendingItems = result.pending_items;
    state.nextItem = result.item;
    state.current = result.item;
    state.currentIsHistory = false;
    state.editing = false;
    state.selectedDecision = null;
    renderItem(result);
  } catch (error) {
    showError(error);
  }
}

$("#reviewer-refresh").addEventListener("click", refresh);
refresh();
