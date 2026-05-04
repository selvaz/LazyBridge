// Right-side panel: tabs for "Node" (agent card), "Event" (payload),
// and "Store" (live key-value snapshot).

import { state, on } from "/static/state.js";
import { renderJSON } from "/static/json-tree.js";

let root, body;
let activeTab = "event";

export function initInspector() {
  root = document.getElementById("inspector");
  root.innerHTML = `
    <div class="tabs">
      <div class="tab" data-tab="node">Node</div>
      <div class="tab active" data-tab="event">Event <span class="badge" id="evt-badge">0</span></div>
      <div class="tab" data-tab="store">Store <span class="badge" id="store-badge">0</span></div>
    </div>
    <div class="body" id="inspector-body">
      <div class="placeholder">Click a node or select an event from the timeline.</div>
    </div>`;
  body = document.getElementById("inspector-body");

  for (const tab of root.querySelectorAll(".tab")) {
    tab.addEventListener("click", () => switchTab(tab.dataset.tab));
  }

  on("eventArrived", () => {
    document.getElementById("evt-badge").textContent = state.events.length;
    if (state.selectedSeq === null && activeTab === "event") {
      renderEvent(state.events[state.events.length - 1]);
    }
  });
  on("select", (seq) => {
    if (activeTab === "event") renderEvent(state.byId.get(seq));
  });
  on("storeChanged", () => {
    document.getElementById("store-badge").textContent = Object.keys(state.store || {}).length;
    if (activeTab === "store") renderStore();
    // If node card is open, refresh its store section
    if (activeTab === "node" && state.selectedNode) {
      _refreshNodeTab(state.selectedNode);
    }
  });
  on("storeProvenanceChanged", () => {
    if (activeTab === "node" && state.selectedNode?.id === "__STORE__") renderStoreCard();
  });
  on("memoryChanged", () => {
    if (activeTab === "node" && state.selectedNode) {
      _refreshNodeTab(state.selectedNode);
    }
  });
  on("nodeClick", (n) => {
    switchTab("node");
    if (n.id === "__START__") renderStartCard();
    else if (n.id === "__END__") renderEndCard();
    else if (n.id === "__STORE__") renderStoreCard();
    else renderNodeCard(n);
  });
}

function _refreshNodeTab(node) {
  if (!node) return;
  if (node.id === "__START__") renderStartCard();
  else if (node.id === "__END__") renderEndCard();
  else if (node.id === "__STORE__") renderStoreCard();
  else renderNodeCard(node);
}

function switchTab(t) {
  activeTab = t;
  syncTabUI();
  if (t === "event") {
    renderEvent(state.byId.get(state.selectedSeq) || state.events[state.events.length - 1]);
  } else if (t === "store") {
    renderStore();
  } else if (t === "node") {
    if (!state.selectedNode) {
      body.innerHTML = `<div class="placeholder">Click any node on the graph.</div>`;
      return;
    }
    _refreshNodeTab(state.selectedNode);
  }
}

function syncTabUI() {
  for (const tab of root.querySelectorAll(".tab")) {
    tab.classList.toggle("active", tab.dataset.tab === activeTab);
  }
}

// ---- START / END node cards ----------------------------------------------

function renderStartCard() {
  const task = state.pipelineTask;
  const agents = [...state.agentTasks.entries()];
  body.innerHTML = `
    <div class="agent-card">
      <div class="ac-nameplate">
        <span class="ac-type-icon" style="color:var(--accent-store);font-size:22px">▶</span>
        <div>
          <div class="ac-name" style="color:var(--accent-store)">Pipeline START</div>
          <div class="ac-badges">
            <span class="ac-badge" style="color:var(--accent-store);border-color:rgba(80,250,123,.3)">entry point</span>
          </div>
        </div>
      </div>
      ${task ? `
      <div class="ac-section">
        <div class="ac-section-title">PIPELINE TASK</div>
        <div class="ac-task">${escapeHtml(task)}</div>
      </div>` : `<div class="ac-section"><div class="placeholder">No task recorded yet.</div></div>`}
      ${agents.length ? `
      <div class="ac-section">
        <div class="ac-section-title">AGENT TASKS (${agents.length})</div>
        ${agents.map(([a, t]) => `
          <div style="margin-bottom:8px">
            <div style="font-size:9px;color:var(--text-dim);margin-bottom:3px">${escapeHtml(a)}</div>
            <div class="ac-task">${escapeHtml(truncate(t, 200))}</div>
          </div>`).join("")}
      </div>` : ""}
    </div>`;
}

function renderEndCard() {
  const outputs = [...state.pipelineOutputs.entries()];
  const totalEvts  = state.events.length;
  const llmCalls   = state.events.filter(e => e.event_type === "model_response").length;
  const totalIn    = state.events.reduce((s, e) => s + (e.usage?.input_tokens  || 0), 0);
  const totalOut   = state.events.reduce((s, e) => s + (e.usage?.output_tokens || 0), 0);
  body.innerHTML = `
    <div class="agent-card">
      <div class="ac-nameplate">
        <span class="ac-type-icon" style="color:var(--accent-router);font-size:22px">■</span>
        <div>
          <div class="ac-name" style="color:var(--accent-router)">Pipeline END</div>
          <div class="ac-badges">
            <span class="ac-badge" style="color:var(--accent-router);border-color:rgba(255,179,64,.3)">exit point</span>
          </div>
        </div>
      </div>
      <div class="ac-section">
        <div class="ac-section-title">RUN STATS</div>
        <div class="ac-stats">
          <div class="ac-stat"><span class="ac-v">${totalEvts}</span><span class="ac-l">events</span></div>
          <div class="ac-stat"><span class="ac-v">${llmCalls}</span><span class="ac-l">LLM calls</span></div>
          ${totalIn  ? `<div class="ac-stat"><span class="ac-v">${fmtTok(totalIn)}</span><span class="ac-l">in-tok</span></div>`  : ""}
          ${totalOut ? `<div class="ac-stat"><span class="ac-v">${fmtTok(totalOut)}</span><span class="ac-l">out-tok</span></div>` : ""}
        </div>
      </div>
      ${outputs.length ? `
      <div class="ac-section">
        <div class="ac-section-title">AGENT OUTPUTS (${outputs.length})</div>
        ${outputs.map(([a, r]) => `
          <div style="margin-bottom:10px">
            <div style="font-size:9px;font-weight:700;color:var(--accent-router);margin-bottom:4px">${escapeHtml(a)}</div>
            <pre style="font-family:JetBrains Mono,monospace;font-size:10px;color:var(--text);white-space:pre-wrap;word-break:break-word;max-height:180px;overflow:auto;margin:0">${escapeHtml(String(r).slice(0, 1200))}</pre>
          </div>`).join("")}
      </div>` : `<div class="ac-section"><div class="placeholder">Pipeline not finished yet.</div></div>`}
    </div>`;
}

// ---- Store node card -----------------------------------------------------

function renderStoreCard() {
  const entries = Object.entries(state.store || {});
  body.innerHTML = `<div class="agent-card">
    <div class="ac-nameplate">
      <span class="ac-type-icon" style="color:var(--accent-store);font-size:22px">🗄</span>
      <div>
        <div class="ac-name" style="color:var(--accent-store)">Shared Store</div>
        <div class="ac-badges"><span class="ac-badge" style="color:var(--accent-store)">${entries.length} keys</span></div>
      </div>
    </div>
    ${entries.length ? `
    <div class="ac-section">
      <div class="ac-section-title">ENTRIES</div>
      ${entries.map(([k, v]) => {
        const prov = state.storeProvenance.get(k);
        const val  = typeof v === "object" && v?.value !== undefined ? v.value : v;
        const writer = prov?.agent || (typeof v === "object" ? v.agent : null);
        return `<div style="margin-bottom:10px;padding:8px;background:rgba(80,250,123,0.04);border-left:2px solid var(--accent-store);border-radius:0 6px 6px 0">
          <div style="font-size:9px;font-weight:700;color:var(--accent-store);margin-bottom:3px">${escapeHtml(k)}</div>
          ${writer ? `<div style="font-size:9px;color:var(--text-dim);margin-bottom:4px">written by ${escapeHtml(writer)}</div>` : ""}
          <pre style="font-size:10px;color:var(--text);white-space:pre-wrap;word-break:break-word;margin:0;max-height:100px;overflow:auto">${escapeHtml(String(val).slice(0, 400))}</pre>
        </div>`;
      }).join("")}
    </div>` : `<div class="ac-section"><div class="placeholder">Store is empty.</div></div>`}
  </div>`;
}

// ---- Agent card (D&D character sheet) ------------------------------------

function renderNodeCard(node) {
  // Derive tools from graph links
  const tools = node.tools || state.links
    .filter(l => {
      const src = typeof l.source === "object" ? l.source.id : l.source;
      return src === node.id && l.kind === "tool";
    })
    .map(l => {
      const tgt = typeof l.target === "object" ? l.target.id : l.target;
      return tgt.replace(/^tool:/, "");
    });

  // Context sources: nodes that send context INTO this node
  const ctxSources = state.links
    .filter(l => {
      const tgt = typeof l.target === "object" ? l.target.id : l.target;
      return tgt === node.id && (l.kind === "context" || l.kind === "router");
    })
    .map(l => {
      const src = typeof l.source === "object" ? l.source.id : l.source;
      const srcNode = state.nodes.find(n => n.id === src);
      return srcNode?.name || src;
    });

  // Store keys read by this agent (from store_read edges)
  const storeReads = state.links
    .filter(l => {
      const s = typeof l.source === "object" ? l.source.id : l.source;
      const tgt = typeof l.target === "object" ? l.target.id : l.target;
      return s === "__STORE__" && tgt === node.id;
    })
    .map(l => l.label || l.kind || "?");

  // Store contributions (entries where .agent matches this node's name)
  const storeContribs = Object.entries(state.store || {})
    .filter(([, v]) => v && typeof v === "object" && v.agent === node.name)
    .map(([k, v]) => ({ key: k, ts: v.ts }));

  // Also check provenance for keys written by this agent
  const provenanceWrites = [...state.storeProvenance.entries()]
    .filter(([, prov]) => prov.agent === node.name)
    .map(([k]) => k)
    .filter(k => !storeContribs.find(c => c.key === k));

  // Memory entries for this agent
  const memEntries = state.memoryEntries.get(node.name) || [];

  // Task assigned to this agent (from agent_start events)
  const task = state.agentTasks.get(node.name) || state.agentTasks.get(node.id);

  // Activity stats
  const evts     = state.events.filter(e => e.agent_name === node.name || e.agent_name === node.id);
  const llmCalls = evts.filter(e => e.event_type === "model_response").length;
  const toolCalls = evts.filter(e => e.event_type === "tool_call").length;
  const inputTok  = evts.reduce((s, e) => s + (e.usage?.input_tokens  || 0), 0);
  const outputTok = evts.reduce((s, e) => s + (e.usage?.output_tokens || 0), 0);

  const typeLabel  = node.type || "agent";
  const typeClass  = `type-${typeLabel}`;
  const isActive   = evts.length > 0;

  body.innerHTML = `
    <div class="agent-card">

      <div class="ac-nameplate">
        <span class="ac-type-icon ${typeClass}">${typeLabel === "tool" ? "⬡" : typeLabel === "router" ? "◆" : "▣"}</span>
        <div>
          <div class="ac-name">${escapeHtml(node.name)}</div>
          <div class="ac-badges">
            ${node.provider ? `<span class="ac-badge provider">${escapeHtml(node.provider)}</span>` : ""}
            ${node.model    ? `<span class="ac-badge model">${escapeHtml(truncate(node.model, 22))}</span>` : ""}
            <span class="ac-badge ${typeClass}">${typeLabel}</span>
            ${isActive ? `<span class="ac-badge live">● active</span>` : `<span class="ac-badge dormant">○ idle</span>`}
          </div>
        </div>
      </div>

      ${task ? `
      <div class="ac-section">
        <div class="ac-section-title">TASK</div>
        <div class="ac-task">${escapeHtml(truncate(task, 320))}</div>
      </div>` : ""}

      ${ctxSources.length ? `
      <div class="ac-section">
        <div class="ac-section-title">CONTEXT FROM</div>
        <div class="ac-ctx-sources">
          ${ctxSources.map(s => `<span class="ac-ctx-source">→ ${escapeHtml(s)}</span>`).join("")}
        </div>
      </div>` : ""}

      ${node.system || node.description ? `
      <div class="ac-section">
        <div class="ac-section-title">SYSTEM PROMPT</div>
        <div class="ac-system">"${escapeHtml((node.system || node.description || "").slice(0, 280))}"</div>
      </div>` : ""}

      ${tools.length ? `
      <div class="ac-section">
        <div class="ac-section-title">TOOLS (${tools.length})</div>
        <div class="ac-tools">
          ${tools.map(t => `<span class="ac-tool">⬡ ${escapeHtml(t)}</span>`).join("")}
        </div>
      </div>` : ""}

      ${isActive ? `
      <div class="ac-section">
        <div class="ac-section-title">ACTIVITY</div>
        <div class="ac-stats">
          <div class="ac-stat"><span class="ac-v">${llmCalls}</span><span class="ac-l">LLM calls</span></div>
          <div class="ac-stat"><span class="ac-v">${toolCalls}</span><span class="ac-l">tool calls</span></div>
          <div class="ac-stat"><span class="ac-v">${evts.length}</span><span class="ac-l">events</span></div>
          ${inputTok ? `<div class="ac-stat"><span class="ac-v">${fmtTok(inputTok)}</span><span class="ac-l">in-tok</span></div>` : ""}
          ${outputTok ? `<div class="ac-stat"><span class="ac-v">${fmtTok(outputTok)}</span><span class="ac-l">out-tok</span></div>` : ""}
        </div>
      </div>` : ""}

      ${storeContribs.length || provenanceWrites.length ? `
      <div class="ac-section">
        <div class="ac-section-title">STORE WRITES (${storeContribs.length + provenanceWrites.length})</div>
        <div class="ac-store-list">
          ${storeContribs.map(c => `
            <div class="ac-store-item" title="${escapeHtml(c.key)}">
              <span class="ac-store-dot" style="color:var(--accent-store)">●</span>
              <span class="ac-store-key">${escapeHtml(c.key)}</span>
              ${c.ts ? `<span class="ac-store-ts">${new Date(c.ts * 1000).toLocaleTimeString()}</span>` : ""}
            </div>`).join("")}
          ${provenanceWrites.map(k => `
            <div class="ac-store-item">
              <span class="ac-store-dot" style="color:var(--accent-store)">●</span>
              <span class="ac-store-key">${escapeHtml(k)}</span>
            </div>`).join("")}
        </div>
      </div>` : ""}

      ${storeReads.length ? `
      <div class="ac-section">
        <div class="ac-section-title">STORE READS</div>
        <div class="ac-store-list">
          ${storeReads.map(k => `
            <div class="ac-store-item">
              <span class="ac-store-dot" style="color:var(--accent-store);opacity:0.5">○</span>
              <span class="ac-store-key">${escapeHtml(k)}</span>
            </div>`).join("")}
        </div>
      </div>` : ""}

      ${memEntries.length ? `
      <div class="ac-section">
        <div class="ac-section-title">MEMORY (${memEntries.length})</div>
        ${memEntries.map(e => `
          <div style="margin-bottom:8px;padding:6px 8px;background:rgba(179,136,255,0.05);border-left:2px solid var(--accent-think, #b388ff);border-radius:0 4px 4px 0">
            <div style="font-size:9px;font-weight:700;color:var(--accent-think,#b388ff);margin-bottom:2px">${escapeHtml(e.key)}</div>
            <pre style="font-size:10px;color:var(--text);white-space:pre-wrap;word-break:break-word;margin:0;max-height:80px;overflow:auto">${escapeHtml(String(e.value ?? "").slice(0, 300))}</pre>
          </div>`).join("")}
      </div>` : ""}

      ${!isActive && !tools.length && !task ? `
      <div class="ac-section">
        <div class="placeholder" style="padding-top:12px">Agent registered — not yet active in this run.</div>
      </div>` : ""}

    </div>`;
}

// ---- Event tab -----------------------------------------------------------

function renderEvent(ev) {
  if (!ev) {
    body.innerHTML = `<div class="placeholder">No events yet.</div>`;
    return;
  }
  const ts  = ev.ts ? new Date(ev.ts * 1000).toLocaleTimeString() : "";
  const cls = (ev.event_type || "").startsWith("tool") ? "tool"
            : ev.event_type === "tool_error" ? "error" : "";
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

// ---- Store tab -----------------------------------------------------------

function renderStore() {
  const keys = Object.keys(state.store || {}).sort();
  if (!keys.length) {
    body.innerHTML = `<div class="placeholder">Store is empty.</div>`;
    return;
  }
  body.innerHTML = "";
  const now = performance.now();
  for (const k of keys) {
    const fresh   = state.storeFresh.get(k);
    const isFresh = fresh && (now - fresh) < 2000;
    const div     = document.createElement("div");
    div.className = "store-entry" + (isFresh ? " fresh" : "");
    const raw = state.store[k];
    const valStr = JSON.stringify(typeof raw === "object" && raw?.value !== undefined ? raw.value : raw, null, 2);
    const agent  = typeof raw === "object" && raw?.agent ? raw.agent : null;
    div.innerHTML = `
      <div class="key">${escapeHtml(k)}${agent ? ` <span style="opacity:.55;font-weight:400">by ${escapeHtml(agent)}</span>` : ""}</div>
      <div class="meta">${valStr.length} chars</div>
      <pre class="val">${escapeHtml(valStr.length > 4000 ? valStr.slice(0, 4000) + "…" : valStr)}</pre>`;
    body.appendChild(div);
  }
}

// ---- helpers -------------------------------------------------------------

function truncate(s, n) {
  if (!s) return "";
  return s.length <= n ? s : s.slice(0, n - 1) + "…";
}

function fmtTok(n) {
  return n >= 1000 ? (n / 1000).toFixed(1) + "k" : String(n);
}

function escapeHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
