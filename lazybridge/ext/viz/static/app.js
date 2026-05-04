// Entry point: bootstrap meta + graph + subscriptions. Loads D3 v7
// first (UMD bundle exposes window.d3), then hooks the panels into
// the SSE stream. Everything else lives in a focused module.

import { state, emit, on } from "/static/state.js";
import { getJSON } from "/static/auth.js";
import { initGraph, setGraph, zoomIn, zoomOut, zoomReset, getLayers, setVisibleLayers } from "/static/graph.js";
import { dispatch, refreshStore } from "/static/dispatch.js";
import { openStream } from "/static/sse.js";
import { initInspector } from "/static/inspector.js";
import { initTimeline } from "/static/timeline.js";

await loadD3();

// Zoom controls
document.getElementById("zoom-in")   ?.addEventListener("click", zoomIn);
document.getElementById("zoom-out")  ?.addEventListener("click", zoomOut);
document.getElementById("zoom-reset")?.addEventListener("click", zoomReset);
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  if (e.key === "+" || e.key === "=") zoomIn();
  if (e.key === "-" || e.key === "_") zoomOut();
  if (e.key === "0")                  zoomReset();
});

initGraph();
initInspector();
initTimeline();

const meta = await getJSON("/api/meta").catch(() => ({}));
state.meta = meta;
applyMeta(meta);
emit("metaLoaded");

const g = await getJSON("/api/graph").catch(() => ({ nodes: [], edges: [] }));
setGraph(buildNodes(g), buildLinks(g));
buildLevelFilter();

// Snapshot for late-joining tabs (replay-mode pre-fill happens here too)
const snap = await getJSON("/api/snapshot").catch(() => ({ events: [] }));
for (const ev of snap.events || []) dispatch(ev);

await refreshStore().catch(() => {});

document.getElementById("canvas-hint").classList.add("hidden");

openStream(dispatch, (s) => {
  if (s === "open") document.getElementById("canvas-hint").classList.add("hidden");
});

// ---- helpers ---------------------------------------------------------

async function loadD3() {
  if (window.d3) return;
  await new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = "https://d3js.org/d3.v7.min.js";
    s.onload = res;
    s.onerror = () => rej(new Error("Failed to load D3"));
    document.head.appendChild(s);
  });
}

function buildNodes(graph) {
  return (graph.nodes || []).map(n => ({
    id: n.id,
    name: n.name || n.id,
    type: n.type || "agent",
    provider: n.provider,
    model: n.model,
    system: n.system,
  }));
}

function buildLinks(graph) {
  return (graph.edges || []).map(e => ({
    source: e.from,
    target: e.to,
    kind: e.type || "tool",
    label: e.label,
  }));
}

function buildLevelFilter() {
  const container = document.getElementById("level-filter");
  if (!container) return;
  const layers = getLayers();
  if (layers.length <= 1) { container.style.display = "none"; return; }

  container.innerHTML = "";
  const active = new Set(layers); // all visible by default

  const allBtn = document.createElement("button");
  allBtn.className = "lf-btn lf-all active";
  allBtn.textContent = "ALL";
  allBtn.addEventListener("click", () => {
    active.clear(); layers.forEach(l => active.add(l));
    sync();
  });
  container.appendChild(allBtn);

  for (const l of layers) {
    const btn = document.createElement("button");
    btn.className = "lf-btn active";
    btn.dataset.layer = l;
    btn.textContent = `L${l}`;
    btn.addEventListener("click", () => {
      if (active.has(l) && active.size > 1) active.delete(l);
      else active.add(l);
      sync();
    });
    container.appendChild(btn);
  }

  function sync() {
    const allActive = active.size === layers.length;
    allBtn.classList.toggle("active", allActive);
    for (const btn of container.querySelectorAll("[data-layer]")) {
      btn.classList.toggle("active", active.has(Number(btn.dataset.layer)));
    }
    setVisibleLayers(allActive ? null : active);
  }
}

function applyMeta(m) {
  const root = document.getElementById("meta");
  if (!m) return;
  const sid = (m.session_id || "").slice(0, 8);
  root.innerHTML = `
    <span class="pill">mode<span class="v">${m.mode || "?"}</span></span>
    <span class="pill">session<span class="v">${sid}</span></span>
    <span class="pill" id="meta-evt">events<span class="v">0</span></span>`;
  on("eventArrived", () => {
    const span = document.querySelector("#meta-evt .v");
    if (span) span.textContent = state.events.length;
  });
}
