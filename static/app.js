"use strict";

const byId = (id) => document.getElementById(id);
const ui = {
  messages: byId("messages"), notice: byId("notice"), files: byId("file-list"),
  activities: byId("activity-list"), input: byId("message-input"),
  evaluation: byId("evaluation-file"), prediction: byId("prediction-file"),
};
let csrfToken = "";
let files = [];
let busy = false;

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
  for (const id of ["send-button", "conditional-button", "reset-button"]) byId(id).disabled = value;
  byId("send-button").firstChild.textContent = value ? "Working " : "Send ";
}

function message(role, text, pending = false) {
  const welcome = ui.messages.querySelector(".welcome-card");
  if (welcome) welcome.remove();
  const wrapper = element("div", `message ${role}${pending ? " pending" : ""}`);
  wrapper.append(element("div", "message-label", role === "user" ? "You" : "Assistant"));
  wrapper.append(element("div", "message-text", text));
  ui.messages.append(wrapper);
  ui.messages.scrollTop = ui.messages.scrollHeight;
  return wrapper;
}

function resultCard(result) {
  const card = element("div", "result-card");
  const headings = {
    predict_single: "Single-record prediction", predict_batch: "Batch prediction",
    evaluate: "Labeled evaluation", evaluate_then_predict: "Conditional prediction",
  };
  const title = element("div", "result-heading");
  title.append(element("span", "", headings[result.tool] || "Model result"));
  if (/^[a-zA-Z0-9_-]+$/.test(result.download_id || "")) {
    const link = element("a", "", "Download results ↓");
    link.href = `/api/download/${encodeURIComponent(result.download_id)}`;
    title.append(link);
  }
  card.append(title);
  const grid = element("div", "result-grid");
  const known = ["row_index", "prediction", "label", "malware_probability", "auc", "accuracy", "evaluation_coverage", "total_count", "valid_count", "invalid_count", "evaluated_count", "missing_label_count", "invalid_label_count", "malware_count", "goodware_count"];
  for (const key of known) {
    if (result[key] === undefined) continue;
    const item = element("div", "");
    item.append(element("div", "metric-label", key.replaceAll("_", " ")));
    let value = result[key];
    if (value === null) value = "Unavailable";
    else if (typeof value === "number" && ["accuracy", "malware_probability", "evaluation_coverage"].includes(key)) value = `${(value * 100).toFixed(2)}%`;
    else if (typeof value === "number" && key === "auc") value = value.toFixed(6);
    item.append(element("div", "metric-value", value));
    grid.append(item);
  }
  card.append(grid);
  if (Array.isArray(result.confusion_matrix) && result.confusion_matrix.length === 2) {
    const table = element("table", "confusion-table");
    const caption = element("caption", "", "Confusion matrix · rows: actual · columns: predicted");
    table.append(caption);
    const head = element("tr", "");
    for (const text of ["", "Goodware", "Malware"]) head.append(element("th", "", text));
    table.append(head);
    result.confusion_matrix.forEach((values, index) => {
      const row = element("tr", "");
      row.append(element("th", "", index === 0 ? "Goodware" : "Malware"));
      for (const value of values) row.append(element("td", "", value));
      table.append(row);
    });
    card.append(table);
  }
  if (result.class_counts) card.append(element("p", "result-detail", `Evaluated labels: ${result.class_counts["0"]} goodware · ${result.class_counts["1"]} malware`));
  if (result.auc === null) card.append(element("p", "result-detail", "AUC needs both classes among valid labeled rows."));
  if (result.invalid_count) {
    card.append(element("p", "result-detail", `${result.invalid_count} invalid row(s). ${result.tool === "predict_batch" ? "Every row is retained in the downloadable CSV." : "Metrics cover only the valid labeled rows."}`));
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

function renderFiles(items) {
  files = items;
  ui.files.replaceChildren();
  byId("file-count").textContent = `${items.length} / 5`;
  if (!items.length) ui.files.append(element("li", "empty-state", "Your uploaded files will appear here."));
  for (const file of items) {
    const item = element("li", "file-item");
    item.append(element("div", "file-name", file.name));
    item.append(element("div", "file-meta", `${file.rows.toLocaleString()} rows · ${file.columns.length} columns`));
    const idRow = element("div", "file-id-row");
    const id = element("code", "file-id", file.id);
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
  for (const entry of items) {
    const item = element("div", "activity-item");
    item.append(element("span", "", String(entry.tool || "Tool").replaceAll("_", " ")));
    const status = ["success", "error", "skipped"].includes(entry.status) ? entry.status : "recorded";
    item.append(element("span", `activity-status ${status}`, status));
    if (entry.reason || entry.error) item.append(element("p", "", entry.reason || entry.error));
    if (entry.arguments) item.append(element("p", "", JSON.stringify(entry.arguments)));
    ui.activities.append(item);
  }
}

async function refreshSession() {
  const state = await api("/api/session");
  csrfToken = state.csrf_token;
  renderFiles(state.files || []);
  renderActivities(state.activity || []);
  return state;
}

async function sendChat(text) {
  if (busy) return;
  notice("");
  setBusy(true);
  message("user", text);
  ui.input.value = "";
  const pending = message("assistant", "Choosing tools and running the model…", true);
  try {
    const response = await api("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: text }),
    });
    pending.remove();
    const answer = message("assistant", response.reply || "The request completed.");
    for (const result of response.results || []) answer.append(resultCard(result));
    if (response.error) notice(response.reply || "The request could not be completed.");
    await refreshSession();
    ui.messages.scrollTop = ui.messages.scrollHeight;
  } catch (error) {
    pending.remove();
    message("assistant", error.message);
    notice(error.message);
  } finally {
    setBusy(false);
    ui.input.focus();
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

byId("file-input").addEventListener("change", () => {
  byId("selected-file").textContent = byId("file-input").files[0]?.name || "";
});

byId("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = byId("file-input").files[0];
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".csv")) return notice("Choose a CSV file.");
  if (file.size > 5 * 1024 * 1024) return notice("CSV upload exceeds 5 MiB.");
  const data = new FormData();
  data.append("file", file);
  byId("upload-button").disabled = true;
  byId("upload-button").textContent = "Validating CSV…";
  try {
    const response = await api("/api/upload", { method: "POST", body: data });
    await refreshSession();
    byId("upload-form").reset();
    byId("selected-file").textContent = "";
    notice(`${response.file.name} uploaded. Use its ID in your request or choose a quick action.`, true);
  } catch (error) { notice(error.message); }
  finally {
    byId("upload-button").disabled = false;
    byId("upload-button").textContent = "Upload CSV";
  }
});

byId("conditional-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const threshold = Number(byId("threshold").value);
  const index = Number(byId("row-index").value);
  if (!ui.evaluation.value || !ui.prediction.value) return notice("Choose evaluation and prediction files.");
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1 || !Number.isInteger(index) || index < 0) return notice("Use an accuracy between 0 and 1 and a whole row number starting at 0.");
  sendChat(`Evaluate file ${ui.evaluation.value}; only if accuracy >= ${threshold}, predict row ${index} of file ${ui.prediction.value}.`);
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    if (!files.length) return notice("Upload a feature CSV first.");
    ui.input.value = button.dataset.prompt.replace("{file_id}", files[0].id);
    ui.input.focus();
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

initialize();
