// Bottom timeline: colored tick per event, click to inspect.
// Play / Step controls are always visible — in live mode they navigate
// the already-received event buffer; in replay mode they also drive
// the server-side ReplayController.
//
// Keyboard: Space = play/pause, J/K or ←/→ = step back/forward.

import { state, on, selectEvent } from "/static/state.js";
import { postJSON } from "/static/auth.js";

let track, cursor;
let mode = "live";
let playing = false;  // local playback through received events
let playTimer = null;

export function initTimeline() {
  const wrap = document.getElementById("timeline-wrap");
  wrap.innerHTML = `
    <div class="tl-controls">
      <span class="counter" id="tl-counter">0 events</span>
      <span class="spacer"></span>
      <button data-act="prev" title="Previous event (K / ←)">‹</button>
      <button data-act="play" id="tl-play" title="Play through events (Space)">▶</button>
      <button data-act="next" title="Next event (J / →)">›</button>
      <select id="tl-speed" title="Playback speed">
        <option value="0.5">0.5×</option>
        <option value="1" selected>1×</option>
        <option value="2">2×</option>
        <option value="5">5×</option>
        <option value="10">10×</option>
      </select>
      <select id="tl-filter" title="Filter by event type">
        <option value="">all events</option>
        <option value="tool_call">tool calls</option>
        <option value="model_response">model responses</option>
        <option value="agent_start">agent start/finish</option>
      </select>
      <span id="tl-mode-badge" class="mode-badge">LIVE</span>
    </div>
    <div class="tl-track" id="tl-track">
      <div class="tl-cursor" id="tl-cursor" style="left:0"></div>
    </div>`;

  track  = document.getElementById("tl-track");
  cursor = document.getElementById("tl-cursor");

  for (const btn of wrap.querySelectorAll("button[data-act]")) {
    btn.addEventListener("click", () => onControl(btn.dataset.act, btn));
  }
  document.getElementById("tl-speed").addEventListener("change", (e) => {
    if (mode === "replay") {
      postJSON("/api/control/speed", { speed: parseFloat(e.target.value) }).catch(() => {});
    }
  });
  document.getElementById("tl-filter").addEventListener("change", redraw);

  on("metaLoaded", () => {
    mode = state.meta?.mode || "live";
    const badge = document.getElementById("tl-mode-badge");
    if (badge) { badge.textContent = mode.toUpperCase(); badge.className = `mode-badge ${mode}`; }
  });
  on("eventArrived", () => { redraw(); if (playing) advancePlay(); });
  on("select", redraw);

  // Keyboard shortcuts
  document.addEventListener("keydown", onKey);
}

function onKey(e) {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  if (e.code === "Space")   { e.preventDefault(); togglePlay(); }
  if (e.code === "KeyJ" || e.code === "ArrowRight") stepLocal(+1);
  if (e.code === "KeyK" || e.code === "ArrowLeft")  stepLocal(-1);
}

function onControl(act, btn) {
  if (act === "prev") return stepLocal(-1);
  if (act === "next") return stepLocal(+1);
  if (act === "play") togglePlay();
}

function togglePlay() {
  playing = !playing;
  const btn = document.getElementById("tl-play");
  btn.textContent = playing ? "❚❚" : "▶";
  btn.title = playing ? "Pause (Space)" : "Play (Space)";

  if (mode === "replay") {
    postJSON("/api/control/" + (playing ? "play" : "pause"), {}).catch(() => {});
  }

  if (playing) {
    schedulePlay();
  } else {
    clearTimeout(playTimer);
    playTimer = null;
  }
}

function schedulePlay() {
  if (!playing) return;
  const speed = parseFloat(document.getElementById("tl-speed")?.value || "1");
  const delay = Math.max(80, 400 / speed);
  playTimer = setTimeout(() => {
    if (!playing) return;
    const moved = stepLocal(+1);
    if (moved) schedulePlay();
    else { playing = false; const btn = document.getElementById("tl-play"); if (btn) { btn.textContent = "▶"; btn.title = "Play (Space)"; } }
  }, delay);
}

// Advance one step in the local buffer. Returns true if moved.
function stepLocal(dir) {
  if (!state.events.length) return false;
  const filter = document.getElementById("tl-filter")?.value || "";
  const shown = filter ? state.events.filter(e => (e.event_type || "").includes(filter)) : state.events;
  if (!shown.length) return false;
  let idx = shown.findIndex(e => e._seq === state.selectedSeq);
  if (idx === -1) idx = dir > 0 ? -1 : shown.length;
  const next = Math.max(0, Math.min(shown.length - 1, idx + dir));
  if (next === idx) return false;
  selectEvent(shown[next]._seq);
  return true;
}

function advancePlay() {
  if (!playing || mode !== "live") return;
  // In live mode, auto-select the latest event when tail-following
  if (state.selectedSeq == null) return;
  const last = state.events[state.events.length - 1];
  if (last && last._seq > state.selectedSeq) selectEvent(last._seq);
}

function redraw() {
  const filter = document.getElementById("tl-filter")?.value || "";
  const all    = state.events;
  const shown  = filter ? all.filter(e => (e.event_type || "").includes(filter)) : all;

  document.getElementById("tl-counter").textContent =
    shown.length === all.length
      ? `${all.length} event${all.length !== 1 ? "s" : ""}`
      : `${shown.length} / ${all.length}`;

  // Re-render ticks
  for (const t of [...track.querySelectorAll(".tl-tick")]) t.remove();
  if (!shown.length) { cursor.style.left = "0"; return; }

  shown.forEach((ev) => {
    const seq  = ev._seq;
    const xPct = ((seq - shown[0]._seq) / Math.max(1, shown[shown.length - 1]._seq - shown[0]._seq)) * 100;
    const tick = document.createElement("div");
    tick.className = `tl-tick t-${ev.event_type}`;
    if (seq === state.selectedSeq) tick.classList.add("selected");
    tick.style.left = `calc(${xPct}% - 1px)`;
    tick.title = `${ev.event_type}${ev.agent_name ? " · " + ev.agent_name : ""}`;
    tick.addEventListener("click", () => selectEvent(seq));
    track.appendChild(tick);
  });

  // Move cursor to selected
  const sel = state.selectedSeq;
  if (sel != null && shown.length) {
    const range = Math.max(1, shown[shown.length - 1]._seq - shown[0]._seq);
    const xPct  = ((sel - shown[0]._seq) / range) * 100;
    cursor.style.left = `${Math.max(0, Math.min(100, xPct))}%`;
  } else {
    cursor.style.left = "100%";
  }
}
