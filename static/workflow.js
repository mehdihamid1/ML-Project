"use strict";

// Draws the real steps of the session's last chat request from /api/session.
// Every value is inserted as text, never as HTML.
(function () {
  const root = document.getElementById("trace");
  const refresh = document.getElementById("trace-refresh");
  // The progress message the chat showed for this call, as a short title.
  function purpose(call) {
    const text = String(call.purpose || call.stage || "").replace(/^OpenAI is /, "").replace(/…$/, "");
    return text.charAt(0).toLowerCase() + text.slice(1);
  }

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function plural(count, word, many) {
    return `${count} ${count === 1 ? word : (many || `${word}s`)}`;
  }

  function step(lane, label, title, parts) {
    const item = element("li", lane === "llm" ? "flow-step llm" : "flow-step");
    item.append(element("span", `lane-chip lane-${lane}`, label));
    const body = element("div", "step-body");
    body.append(element("h3", "", title));
    for (const part of parts) {
      if (part) body.append(typeof part === "string" ? element("p", "", part) : part);
    }
    item.append(body);
    return item;
  }

  function quoted(prefix, text) {
    const line = element("p", "", prefix);
    line.append(element("q", "", text));
    return line;
  }

  function offered(call) {
    if (!call.tools.length) return "No tools offered: the AI may only pick a stored result.";
    if (call.tool_choice === "auto") return `Tools offered: ${call.tools.join(", ")}. The AI chooses.`;
    return `Tools offered: ${call.tools.join(", ")}. The AI must call ${call.tool_choice}.`;
  }

  function received(parts) {
    const list = element("ul", "sent-list");
    list.append(element("li", "", "The fixed instructions, the same for every call"));
    for (const part of parts || []) {
      if (part.part === "context") {
        const results = part.results || [];
        const summaries = results.length
          ? `; ${plural(results.length, "earlier result summary", "earlier result summaries")} (${results.join(", ")})`
          : "; no earlier results";
        list.append(element("li", "", `${plural(part.files, "registered file")}, IDs and names only${summaries}`));
      } else if (part.part === "history") {
        list.append(element("li", "", `${plural(part.messages, "earlier message")} from this conversation`));
      } else if (part.part === "brief") {
        const item = element("li", "", "Instructions for this request: ");
        item.append(element("q", "", part.text));
        list.append(item);
      } else if (part.part === "message") {
        const item = element("li", "", "Your message: ");
        item.append(element("q", "", part.text));
        list.append(item);
      } else if (part.part === "tool_call") {
        list.append(element("li", "", `Its previous tool call: ${part.name}(${part.arguments})`));
      } else if (part.part === "tool_output") {
        const item = element("li", "", "The tool result it read, a summary without CSV rows:");
        item.append(element("pre", "json", JSON.stringify(part.output, null, 2)));
        list.append(item);
      }
    }
    return list;
  }

  function returned(call) {
    const value = call.returned || {};
    if (value.type === "function_call") {
      const extra = value.count > 1 ? ` and ${value.count - 1} more, refused because tools run one at a time` : "";
      const ignored = call.tools.length ? "" : " No tool ran: this question allowed none.";
      return `Returned: a call to ${value.name}(${value.arguments})${extra}.${ignored}`;
    }
    if (value.type === "answer") {
      const count = Array.isArray(value.result_ids) ? value.result_ids.length : 0;
      return `Returned: an answer of kind “${value.kind}”, focus “${value.focus}”, naming ${plural(count, "stored result")}.`;
    }
    if (value.type === "error") return "Returned: nothing. The OpenAI call failed, so no unverified result was shown.";
    return "Returned: text that was not the required JSON, so Python did not use it.";
  }

  function arguments_(entry) {
    return Object.entries(entry.arguments || {}).map(([key, value]) => `${key}=${value}`).join(", ");
  }

  function toolStep(entry) {
    const call = `${entry.tool}(${arguments_(entry)})`;
    if (entry.status === "success") {
      return step("model", "ML model", `${call} finished`, ["The frozen LightGBM pipeline ran; the full result was stored with an ID."]);
    }
    const word = { skipped: "skipped", blocked: "blocked", error: "failed", started: "interrupted" }[entry.status] || "recorded";
    return step("server", "Server", `${call} ${word}`, [entry.reason || entry.error || ""]);
  }

  function render(trace) {
    root.replaceChildren();
    if (!trace) {
      root.append(element("p", "empty-state", "No request yet in this session. Send one on the Chat tab, then come back here."));
      return;
    }
    const calls = trace.llm_calls || [];
    const activity = trace.activity || [];
    root.append(element("p", "trace-summary",
      `${plural(calls.length, "OpenAI call")} · ${plural(activity.length, "tool step")} · AI model ${trace.model || "unknown"}`));
    const flow = element("ol", "flow trace-flow");
    flow.append(step("browser", "Browser", "You sent", [quoted("", trace.message || "")]));
    flow.append(calls.length
      ? step("server", "Server", "Checks passed", ["The request passed the checks that run before any AI call."])
      : step("server", "Server", "Answered before any AI call", ["A check stopped this request, so OpenAI was not called."]));
    let next = 0;
    for (const call of calls) {
      flow.append(step("llm", "OpenAI", `OpenAI call ${call.call}: ${purpose(call)}`, [
        offered(call), element("p", "received-label", "What it received:"), received(call.sent), returned(call),
      ]));
      const value = call.returned || {};
      // Tool steps that followed this call: the tool it asked for, or a skip the server recorded.
      while (next < activity.length) {
        const entry = activity[next];
        const asked = value.type === "function_call" && call.tools.length && entry.tool === value.name;
        const skipped = value.type === "answer" && entry.status === "skipped";
        if (!asked && !skipped) break;
        flow.append(toolStep(entry));
        next += 1;
        if (asked) break;
      }
    }
    for (; next < activity.length; next += 1) flow.append(toolStep(activity[next]));
    const reply = element("blockquote", "reply-text", trace.reply || "");
    flow.append(step("server", "Server", trace.error ? "Reply written by Python, reporting a problem" : "Reply written by Python", [reply]));
    root.append(flow);
  }

  async function load() {
    refresh.disabled = true;
    try {
      const response = await fetch("/api/session", { credentials: "same-origin", headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error("unavailable");
      render((await response.json()).last_trace);
    } catch (_) {
      root.replaceChildren(element("p", "empty-state", "Your last request could not be loaded. Use Refresh to try again."));
    } finally {
      refresh.disabled = false;
    }
  }

  refresh.addEventListener("click", load);
  load();
})();
