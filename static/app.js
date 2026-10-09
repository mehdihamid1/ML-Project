"use strict";

const byId = (id) => document.getElementById(id);
const ui = {
  messages: byId("messages"), notice: byId("notice"), files: byId("file-list"),
  activities: byId("activity-list"), input: byId("message-input"),
  evaluation: byId("evaluation-file"), prediction: byId("prediction-file"),
  fileInput: byId("file-input"), uploadZone: byId("upload-zone"),
};
const TOOL_NAMES = {
  predict_single: "Single prediction", predict_batch: "Batch prediction",
  evaluate: "Evaluation", evaluate_then_predict: "Evaluate, then predict",
};
const STATUS_WORDS = { started: "running", success: "finished", error: "failed", skipped: "skipped", recorded: "recorded" };
const STATUS_MARKS = { started: "…", success: "✓", error: "✕", skipped: "–", recorded: "•" };
const MAX_UPLOAD_BYTES = 5 * 1024 * 1024;
let csrfToken = "";
let files = [];
let busy = false;
let uploading = false;

function element(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value !== undefined) node.textContent = String(value);
  return node;
}

function notice(message, success = false) {
  ui.notice.textContent = message || "";
  ui.notice.className = success ? "notice success" : "notice";
}

function plural(count, word) {
  return `${Number(count).toLocaleString()} ${word}${count === 1 ? "" : "s"}`;
}

function fileLabel(id, withId = false) {
  const file = files.find((item) => item.id === id);
  const short = String(id || "").slice(0, 8);
  if (!file) return `file ${short}`;
  return withId ? `${file.name} (${short})` : file.name;
}

function percent(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "Unavailable";
  return value > 0 && value < 0.0001 ? "<0.01%" : `${(value * 100).toFixed(2)}%`;
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.method && options.method !== "GET") headers.set("X-CSRF-Token", csrfToken);
  let response;
  try {
    response = await fetch(path, { credentials: "same-origin", ...options, headers });
  } catch (_) {
    throw new Error("The service could not be reached. Check your connection and try again.");
  }
  let data;
  try { data = await response.json(); } catch (_) { throw new Error("The service returned an unexpected response. Try again."); }
  if (!response.ok) throw new Error(data.error || "The request could not be completed.");
  return data;
}

function setBusy(value) {
  busy = value;
  for (const id of ["send-button", "conditional-button", "reset-button"]) byId(id).disabled = value || uploading;
  ui.fileInput.disabled = value || uploading;
  byId("send-button").firstChild.textContent = value ? "Working " : "Send ";
}

// Replies are written by Python. Put each sentence on its own line and turn the
// download path into a link; the words themselves are unchanged.
function appendReply(container, text) {
  const formatted = text.replace(/\. (?=[A-Z])/g, ".\n");
  const pattern = /(Download results: )?(\/api\/download\/[A-Za-z0-9_-]+)/g;
  let last = 0;
  for (const match of formatted.matchAll(pattern)) {
    container.append(formatted.slice(last, match.index));
    const link = element("a", "", match[1] ? "Download results ↓" : "download link");
    link.href = match[2];
    container.append(link);
    last = match.index + match[0].length;
  }
  container.append(formatted.slice(last));
}

function message(role, text, pending = false) {
  const welcome = ui.messages.querySelector(".welcome-card");
  if (welcome) welcome.remove();
  const wrapper = element("div", `message ${role}${pending ? " pending" : ""}`);
  wrapper.append(element("div", "message-label", role === "user" ? "You" : "Assistant"));
  const body = element("div", "message-text");
  if (role === "assistant" && !pending) appendReply(body, text);
  else body.textContent = text;
  wrapper.append(body);
  ui.messages.append(wrapper);
  ui.messages.scrollTop = ui.messages.scrollHeight;
  return wrapper;
}

function describeArguments(args = {}) {
  const parts = [];
  if (args.file_id) parts.push(fileLabel(args.file_id));
  if (args.evaluation_file_id) parts.push(`evaluate ${fileLabel(args.evaluation_file_id)}`);
  if (args.prediction_file_id) parts.push(`predict ${fileLabel(args.prediction_file_id)}`);
  if (Number.isInteger(args.row_index)) parts.push(`row ${args.row_index}`);
  if (typeof args.min_accuracy === "number") parts.push(`minimum accuracy ${args.min_accuracy}`);
  return parts.join(", ");
}

function statusOf(entry) {
  return ["started", "success", "error", "skipped"].includes(entry.status) ? entry.status : "recorded";
}

// The tools behind one answer, in the order they ran.
function toolTrail(activity) {
  const trail = element("div", "tool-trail");
  trail.append(element("span", "tool-trail-label", "Tools"));
  if (!activity.length) {
    trail.append(element("span", "tool-step none", "No tool was called for this answer"));
    return trail;
  }
  for (const entry of activity) {
    const status = statusOf(entry);
    const step = element("span", `tool-step ${status}`);
    const mark = element("span", "tool-mark", STATUS_MARKS[status]);
    mark.setAttribute("aria-hidden", "true");
    const target = describeArguments(entry.arguments);
    step.append(mark, element("code", "", `${entry.tool}()`), `${target ? ` · ${target}` : ""} · ${STATUS_WORDS[status]}`);
    if (entry.reason || entry.error) step.title = entry.reason || entry.error;
    trail.append(step);
  }
  return trail;
}

function stat(label, value, className = "") {
  const item = element("div", `stat ${className}`.trim());
  item.append(element("div", "stat-label", label), element("div", "stat-value", value));
  return item;
}

function singleSummary(result) {
  const body = element("div", "result-body");
  const label = String(result.label || "unavailable");
  const verdict = element("div", `verdict ${label === "malware" ? "malware" : label === "goodware" ? "goodware" : ""}`.trim());
  verdict.append(element("span", "verdict-label", label[0].toUpperCase() + label.slice(1)));
  verdict.append(element("span", "verdict-detail", `Row ${result.row_index}${result.source_row ? ` · CSV line ${result.source_row}` : ""}`));
  const stats = element("div", "stats");
  stats.append(stat("Malware probability", percent(result.malware_probability)));
  if (typeof result.threshold === "number") stats.append(stat("Decision threshold", result.threshold.toFixed(2)));
  body.append(verdict, stats);
  return body;
}

function batchSummary(result) {
  const stats = element("div", "stats");
  stats.append(stat("Rows", Number(result.total_count || 0).toLocaleString()));
  stats.append(stat("Classified", Number(result.valid_count || 0).toLocaleString()));
  stats.append(stat("Malware", Number(result.malware_count || 0).toLocaleString(), "malware"));
  stats.append(stat("Goodware", Number(result.goodware_count || 0).toLocaleString(), "goodware"));
  stats.append(stat("Invalid", Number(result.invalid_count || 0).toLocaleString(), result.invalid_count ? "warning" : ""));
  return stats;
}

function confusionTable(matrix) {
  const table = element("table", "confusion-table");
  table.append(element("caption", "", "Confusion matrix"));
  const head = element("tr", "");
  head.append(element("td", ""));
  for (const text of ["Predicted goodware", "Predicted malware"]) {
    const cell = element("th", "", text);
    cell.scope = "col";
    head.append(cell);
  }
  table.append(head);
  const names = [["Correctly cleared", "Wrongly flagged"], ["Missed", "Correctly detected"]];
  matrix.forEach((values, row) => {
    const line = element("tr", "");
    const header = element("th", "", row === 0 ? "Actual goodware" : "Actual malware");
    header.scope = "row";
    line.append(header);
    values.forEach((value, column) => {
      const cell = element("td", row === column ? "correct" : "incorrect");
      cell.append(element("strong", "", Number(value).toLocaleString()), element("span", "", names[row][column]));
      line.append(cell);
    });
    table.append(line);
  });
  return table;
}

function evaluationSummary(result) {
  const body = element("div", "result-body stacked");
  const stats = element("div", "stats");
  stats.append(stat("AUC", typeof result.auc === "number" ? result.auc.toFixed(6) : "Unavailable"));
  stats.append(stat("Accuracy", percent(result.accuracy)));
  stats.append(stat("Rows evaluated", `${Number(result.evaluated_count || 0).toLocaleString()} of ${Number(result.total_count || 0).toLocaleString()}`));
  body.append(stats);
  const matrix = result.confusion_matrix;
  if (Array.isArray(matrix) && matrix.length === 2 && matrix.every((row) => Array.isArray(row) && row.length === 2)) {
    body.append(confusionTable(matrix));
  }
  return body;
}

function resultCard(result) {
  const card = element("div", "result-card");
  const heading = element("div", "result-heading");
  const title = element("div", "result-title");
  title.append(element("span", "result-kind", TOOL_NAMES[result.tool] || "Model result"));
  if (result.file_id) title.append(element("span", "result-file", fileLabel(result.file_id)));
  heading.append(title);
  if (/^[a-zA-Z0-9_-]+$/.test(result.download_id || "")) {
    const link = element("a", "download-link", "Download results ↓");
    link.href = `/api/download/${encodeURIComponent(result.download_id)}`;
    heading.append(link);
  }
  card.append(heading);
  if (result.tool === "predict_single") card.append(singleSummary(result));
  else if (result.tool === "predict_batch") card.append(batchSummary(result));
  else card.append(evaluationSummary(result));
  if (result.class_counts) card.append(element("p", "result-detail", `Labels evaluated: ${plural(result.class_counts["0"] || 0, "goodware file")} · ${plural(result.class_counts["1"] || 0, "malware file")}`));
  if (result.tool !== "predict_single" && result.tool !== "predict_batch" && result.auc === null) card.append(element("p", "result-detail", "AUC needs both classes among the valid labeled rows."));
  if (result.missing_label_count) card.append(element("p", "result-detail", `${plural(result.missing_label_count, "row")} had no label and ${result.missing_label_count === 1 ? "was" : "were"} left out of the metrics.`));
  if (result.invalid_count) {
    card.append(element("p", "result-detail warning", `${plural(result.invalid_count, "invalid row")}. ${result.tool === "predict_batch" ? "Every row is kept in the downloadable CSV, with its error." : "Metrics cover only the valid labeled rows."}`));
    const invalid = element("ul", "invalid-row-list");
    for (const row of result.invalid_rows || []) {
      invalid.append(element("li", "", `Row ${row.row_index}${row.source_row ? ` (CSV line ${row.source_row})` : ""}: ${(row.errors || []).join("; ")}`));
    }
    if (result.invalid_rows_truncated) invalid.append(element("li", "", "Showing the first 10 invalid rows. Batch downloads contain the full row report."));
    card.append(invalid);
  }
  const details = element("details", "");
  details.append(element("summary", "", "View complete tool result"));
  details.append(element("pre", "", JSON.stringify(result, null, 2)));
  card.append(details);
  return card;
}

function usePrompt(text) {
  ui.input.value = text;
  ui.input.focus();
  ui.input.setSelectionRange(text.length, text.length);
}

function fileAction(label, file, prompt) {
  const button = element("button", "file-action", label);
  button.type = "button";
  button.setAttribute("aria-label", `${label}: ${file.name}`);
  button.addEventListener("click", () => usePrompt(prompt.replace("{file_id}", file.id)));
  return button;
}

function renderFiles(items) {
  files = items;
  ui.files.replaceChildren();
  byId("file-count").textContent = `${items.length} / 5`;
  if (!items.length) ui.files.append(element("li", "empty-state", "Your uploaded files will appear here."));
  for (const file of items) {
    const item = element("li", "file-item");
    item.append(element("div", "file-name", file.name));
    item.append(element("div", "file-meta", `${plural(file.rows, "row")} · ${plural(file.columns.length, "column")}${file.columns.includes("Label") ? " · labeled" : ""}`));
    const idRow = element("div", "file-id-row");
    const id = element("code", "file-id", file.id);
    id.title = file.id;
    const copy = element("button", "copy-file-id", "Copy");
    copy.type = "button";
    copy.title = "Copy file ID";
    copy.setAttribute("aria-label", `Copy file ID for ${file.name}`);
    let confirmationTimer;
    copy.addEventListener("click", async () => {
      clearTimeout(confirmationTimer);
      copy.textContent = "Copy";
      try {
        await navigator.clipboard.writeText(file.id);
        copy.textContent = "Copied!";
        notice(`File ID for ${file.name} copied.`, true);
        confirmationTimer = setTimeout(() => { copy.textContent = "Copy"; }, 1800);
      } catch (_) {
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(id);
        selection?.removeAllRanges();
        selection?.addRange(range);
        notice("Could not copy automatically. The file ID is selected; press Ctrl+C or ⌘C to copy, or use your device's copy menu.");
      }
    });
    idRow.append(element("span", "file-id-label", "ID:"), id, copy);
    item.append(idRow);
    // Each card asks about its own file, so a request never targets the wrong upload.
    const actions = element("div", "file-actions");
    actions.append(fileAction("Predict row 0", file, "Predict row 0 of file {file_id}."));
    actions.append(fileAction("Classify all", file, "Classify all rows in file {file_id}."));
    if (file.columns.includes("Label")) actions.append(fileAction("Evaluate", file, "Evaluate file {file_id} using its labels."));
    item.append(actions);
    ui.files.append(item);
  }
  for (const select of [ui.evaluation, ui.prediction]) {
    const selected = select.value;
    select.replaceChildren();
    select.append(new Option(items.length ? "Choose a file" : "Upload a CSV first", ""));
    for (const file of items) select.append(new Option(`${file.name} (${file.id.slice(0, 8)})`, file.id));
    if (items.some((file) => file.id === selected)) select.value = selected;
    else if (items.length === 1) select.value = items[0].id;
  }
}

function renderActivities(items) {
  ui.activities.replaceChildren();
  if (!items.length) ui.activities.append(element("p", "empty-state", "Tool calls will be recorded here."));
  items.forEach((entry, index) => {
    const status = statusOf(entry);
    const item = element("div", "activity-item");
    const heading = element("div", "activity-heading");
    heading.append(element("span", "activity-number", index + 1), element("code", "activity-tool", `${entry.tool}()`));
    heading.append(element("span", `activity-status ${status}`, STATUS_WORDS[status]));
    item.append(heading);
    const args = entry.arguments || {};
    const described = { ...args };
    for (const key of ["file_id", "evaluation_file_id", "prediction_file_id"]) {
      if (described[key]) described[key] = fileLabel(described[key], true);
    }
    if (Object.keys(described).length) item.append(element("p", "activity-arguments", Object.entries(described).map(([key, value]) => `${key.replaceAll("_", " ")}: ${value}`).join(" · ")));
    if (entry.reason || entry.error) item.append(element("p", "activity-reason", entry.reason || entry.error));
    ui.activities.append(item);
  });
}

async function refreshSession() {
  const state = await api("/api/session");
  csrfToken = state.csrf_token;
  renderFiles(state.files || []);
  renderActivities(state.activity || []);
  return state;
}

async function streamChat(text, onEvent) {
  const interrupted = "Progress was interrupted. Check your session results before sending again.";
  let response;
  try {
    response = await fetch("/api/chat/stream", {
      method: "POST", credentials: "same-origin",
      headers: { "Content-Type": "application/json", "X-CSRF-Token": csrfToken },
      body: JSON.stringify({ message: text }),
    });
  } catch (_) {
    throw new Error("The service could not be reached. Check your connection and try again.");
  }
  if (!response.ok) {
    let data;
    try { data = await response.json(); } catch (_) { throw new Error("The service returned an unexpected response. Check your session before trying again."); }
    throw new Error(data.error || "The request could not be completed.");
  }
  if (!response.body) throw new Error("Live progress is unavailable in this browser.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result;
  function consume(line) {
    if (!line.trim()) return;
    let event;
    try { event = JSON.parse(line); } catch (_) { throw new Error(interrupted); }
    if (event.type === "error") throw new Error(event.error);
    if (event.type === "result") result = event.response;
    else onEvent(event);
  }
  try {
    while (true) {
      let chunk;
      try { chunk = await reader.read(); } catch (_) { throw new Error(interrupted); }
      const { value, done } = chunk;
      buffer += decoder.decode(value, { stream: !done });
      let end;
      while ((end = buffer.indexOf("\n")) !== -1) {
        consume(buffer.slice(0, end));
        buffer = buffer.slice(end + 1);
      }
      if (done) break;
    }
    consume(buffer);
    if (!result) throw new Error(interrupted);
    return result;
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

async function sendChat(text) {
  if (busy || uploading) return;
  notice("");
  setBusy(true);
  const question = message("user", text);
  ui.input.value = "";
  const pending = message("assistant", "Sending your request…", true);
  const phase = pending.querySelector(".message-text");
  phase.setAttribute("role", "status");
  const elapsed = element("div", "progress-time", "Elapsed: 0s");
  elapsed.setAttribute("aria-hidden", "true");
  pending.append(elapsed);
  const started = Date.now();
  const timer = setInterval(() => { elapsed.textContent = `Elapsed: ${Math.floor((Date.now() - started) / 1000)}s`; }, 1000);
  const activity = [];
  try {
    const response = await streamChat(text, (event) => {
      if (event.type === "progress") phase.textContent = event.message;
      if (event.type === "tool" && Number.isInteger(event.index) && event.index >= 0 && event.index < 100) {
        activity[event.index] = event.activity;
        pending.querySelector(".tool-trail")?.remove();
        pending.append(toolTrail(activity.filter(Boolean)));
        const entry = event.activity;
        phase.textContent = `${entry.tool}() ${STATUS_WORDS[statusOf(entry)]}${entry.reason || entry.error ? `: ${entry.reason || entry.error}` : "…"}`;
      }
      ui.messages.scrollTop = ui.messages.scrollHeight;
    });
    pending.remove();
    const answer = message("assistant", response.reply || "The request completed.");
    answer.append(toolTrail(response.activity || []));
    for (const result of response.results || []) answer.append(resultCard(result));
    if (response.error) notice(response.reply || "The request could not be completed.");
    await refreshSession();
    // Show this turn from the question down, so a long answer is read from its start.
    ui.messages.scrollTop = Math.max(0, question.offsetTop - 12);
  } catch (error) {
    pending.remove();
    const answer = message("assistant", error.message || "The connection was interrupted. Check your session results before sending again.");
    if (activity.length) answer.append(toolTrail(activity.filter(Boolean)));
    notice(error.message);
  } finally {
    clearInterval(timer);
    setBusy(false);
    ui.input.focus({ preventScroll: true });
  }
}

byId("chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const text = ui.input.value.trim();
  if (text) sendChat(text);
});

ui.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    byId("chat-form").requestSubmit();
  }
});

function setUploadText(title, hint) {
  byId("upload-title").textContent = title;
  byId("upload-hint").textContent = hint;
}

// Uploads start as soon as files are chosen or dropped, one at a time.
async function uploadFiles(list) {
  const chosen = Array.from(list || []);
  if (!chosen.length || uploading || busy) return;
  uploading = true;
  setBusy(busy);
  ui.uploadZone.classList.add("busy");
  let prepared;
  let problem = "";
  try {
    for (const file of chosen) {
      if (!file.name.toLowerCase().endsWith(".csv")) { problem = `${file.name} is not a CSV file.`; continue; }
      if (file.size > MAX_UPLOAD_BYTES) { problem = `${file.name} is larger than 5 MiB.`; continue; }
      setUploadText(`Validating ${file.name}…`, "Checking type, size and columns");
      const data = new FormData();
      data.append("file", file);
      try {
        const response = await api("/api/upload", { method: "POST", body: data });
        prepared = response.file;
      } catch (error) {
        problem = `${file.name}: ${error.message}`;
        break;
      }
    }
    await refreshSession();
  } catch (error) {
    problem = error.message;
  } finally {
    uploading = false;
    setBusy(busy);
    ui.fileInput.value = "";
    ui.uploadZone.classList.remove("busy");
    setUploadText("Choose or drop CSV files", "Up to 5 MiB · 10,000 rows each");
  }
  if (prepared) usePrompt(prepared.rows === 1 ? `Predict row 0 of file ${prepared.id}.` : `Classify all rows in file ${prepared.id}.`);
  const ready = prepared ? `${prepared.name} uploaded. Review or edit the question, then click Send.` : "";
  if (problem) notice(`${ready} ${problem}`.trim());
  else notice(ready, true);
}

ui.fileInput.addEventListener("change", () => uploadFiles(ui.fileInput.files));
byId("upload-form").addEventListener("submit", (event) => event.preventDefault());

for (const type of ["dragenter", "dragover"]) {
  ui.uploadZone.addEventListener(type, (event) => {
    event.preventDefault();
    ui.uploadZone.classList.add("dragging");
  });
}
for (const type of ["dragleave", "drop"]) {
  ui.uploadZone.addEventListener(type, () => ui.uploadZone.classList.remove("dragging"));
}
ui.uploadZone.addEventListener("drop", (event) => {
  event.preventDefault();
  uploadFiles(event.dataTransfer?.files);
});
// A file dropped outside the upload area must not navigate away from the session.
for (const type of ["dragover", "drop"]) {
  window.addEventListener(type, (event) => {
    if (event.dataTransfer?.types?.includes("Files") && !ui.uploadZone.contains(event.target)) event.preventDefault();
  });
}

byId("conditional-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const threshold = Number(byId("threshold").value);
  const index = Number(byId("row-index").value);
  if (!ui.evaluation.value || !ui.prediction.value) return notice("Choose evaluation and prediction files.");
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1 || !Number.isInteger(index) || index < 0) return notice("Use an accuracy between 0 and 1 and a whole row number starting at 0.");
  usePrompt(`Evaluate file ${ui.evaluation.value}; only if accuracy >= ${threshold}, predict row ${index} of file ${ui.prediction.value}.`);
  notice("Conditional question prepared. Review or edit it, then click Send.", true);
});

// The welcome suggestions use the most recent upload; evaluation needs one with labels.
document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    const needsLabels = button.dataset.prompt.startsWith("Evaluate");
    const file = [...files].reverse().find((item) => !needsLabels || item.columns.includes("Label"));
    if (!file) return notice(needsLabels ? "Upload a CSV with a Label column first." : "Upload a feature CSV first.");
    usePrompt(button.dataset.prompt.replace("{file_id}", file.id));
  });
});

byId("reset-button").addEventListener("click", async () => {
  if (busy) return;
  setBusy(true);
  try {
    await api("/api/reset", { method: "POST" });
    await refreshSession();
    ui.messages.replaceChildren();
    message("assistant", "Session reset. Your files and results were removed. Upload a new CSV to begin.");
    notice("Session reset.", true);
  } catch (error) { notice(error.message); }
  finally { setBusy(false); }
});

async function initialize() {
  try {
    const state = await refreshSession();
    if (state.results?.length) {
      const restored = message("assistant", "Your recent tool results are available in this session.");
      for (const result of state.results) restored.append(resultCard(result));
    }
  } catch (error) { notice(error.message); }
  const badge = byId("model-status");
  try {
    const health = await api("/health");
    badge.classList.add("ready");
    badge.lastChild.textContent = `${health.selected_model || "Model"} ready`;
    if (!health.openai_configured) notice("The model is ready. An administrator must configure OPENAI_API_KEY before chat can run.");
  } catch (_) {
    badge.classList.add("error");
    badge.lastChild.textContent = "Model unavailable";
    notice("The production model is unavailable. Try again after it is configured.");
  }
}

// Wide screens size the workspace to the rest of the first screen (see style.css),
// so the chat and its input are visible without scrolling.
function fitWorkbench() {
  const top = document.querySelector(".workbench").getBoundingClientRect().top + window.scrollY;
  document.documentElement.style.setProperty("--workbench-top", `${Math.round(top)}px`);
}
let fitFrame;
window.addEventListener("resize", () => {
  cancelAnimationFrame(fitFrame);
  fitFrame = requestAnimationFrame(fitWorkbench);
});
fitWorkbench();
document.fonts?.ready.then(fitWorkbench);

initialize();
