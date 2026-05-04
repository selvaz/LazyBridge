// Right-side panel: tabs for "Event" (currently selected event's
// payload) and "Store" (live key-value snapshot with a freshness
// highlight). Subscribes to state events, never owns state.

import { state, on } from "/static/state.js";
import { renderJSON } from "/static/json-tree.js";

let root, body;
let activeTab = "event";

export function initInspector() {
  root = document.getElementById("inspector");
  root.innerHTML = `
    <div class="tabs">
      <div class="tab active" data-tab="event">Event <span class="badge" id="evt-badge">0</span></div>
      <div class="tab" data-tab="store">Store <span class="badge" id="store-badge">0</span></div>
    </div>
    <div class="body" id="inspector-body">
      <div class="placeholder">Select an event from the timeline or click a node.</div>
    </div>`;
  body = document.getElementById("inspector-body");
  for (const tab of root.querySelectorAll(".tab")) {
    tab.addEventListener("click", () => switchTab(tab.dataset.tab));
  }

  on("eventArrived", () => {
    document.getElementById("evt-badge").textContent = state.events.length;
    if (state.selectedSeq === null && activeTab === "event") {
      // Live "follow tail" mode: show the most recent event
      renderEvent(state.events[state.events.length - 1]);
    }
  });
  on("select", (seq) => {
    if (activeTab === "event") renderEvent(state.byId.get(seq));
  });
  on("storeChanged", () => {
    document.getElementById("store-badge").textContent = Object.keys(state.store || {}).length;
    if (activeTab === "store") renderStore();
  });
  on("nodeClick", (n) => {
    activeTab = "event";
    syncTabUI();
    // Find the most recent event mentioning this node
    for (let i = state.events.length - 1; i >= 0; i--) {
      const ev = state.events[i];
      if (ev.agent_name === n.id || ev.name === n.name) {
        renderEvent(ev);
        return;
      }
    }
    body.innerHTML = `<div class="placeholder">No events for ${escapeHtml(n.name)} yet.</div>`;
  });
}

function switchTab(t) {
  activeTab = t;
  syncTabUI();
  if (t === "event") renderEvent(state.byId.get(state.selectedSeq) || state.events[state.events.length - 1]);
  else renderStore();
}

function syncTabUI() {
  for (const tab of root.querySelectorAll(".tab")) {
    tab.classList.toggle("active", tab.dataset.tab === activeTab);
  }
}

function renderEvent(ev) {
  if (!ev) {
    body.innerHTML = `<div class="placeholder">No events yet.</div>`;
    return;
  }
  const ts = ev.ts ? new Date(ev.ts * 1000).toLocaleTimeString() : "";
  const cls = ev.event_type && ev.event_type.startsWith("tool") ? "tool"
            : ev.event_type === "tool_error" ? "error"
            : "";
  body.innerHTML = `
    <div class="evt-head">
      <div class="row">
        <span class="type ${cls}">${escapeHtml(ev.event_type || "?")}</span>
        ${ev.agent_name ? `<span class="agent">${escapeHtml(ev.agent_name)}</span>` : ""}
        <span class="ts">#${ev._seq ?? ""} ${ts}</span>
      </div>
    </div>
    <div id="evt-payload"></div>`;
  const { _seq, event_type, session_id, ...rest } = ev;
  renderJSON(rest, document.getElementById("evt-payload"));
}

function renderStore() {
  const keys = Object.keys(state.store || {}).sort();
  if (!keys.length) {
    body.innerHTML = `<div class="placeholder">Store is empty.</div>`;
    return;
  }
  body.innerHTML = "";
  const now = performance.now();
  for (const k of keys) {
    const fresh = state.storeFresh.get(k);
    const isFresh = fresh && (now - fresh) < 2000;
    const div = document.createElement("div");
    div.className = "store-entry" + (isFresh ? " fresh" : "");
    const valStr = JSON.stringify(state.store[k], null, 2);
    div.innerHTML = `
      <div class="key">${escapeHtml(k)}</div>
      <div class="meta">${valStr.length} chars</div>
      <pre class="val">${escapeHtml(valStr.length > 4000 ? valStr.slice(0, 4000) + "…" : valStr)}</pre>`;
    body.appendChild(div);
  }
}

function escapeHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
