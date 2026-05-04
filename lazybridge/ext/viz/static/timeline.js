// Bottom-of-screen timeline: a colored tick per event, click to
// inspect, plus replay controls when the server signals replay
// mode in /api/meta. Live mode hides the play/pause/speed controls.

import { state, on, selectEvent } from "/static/state.js";
import { postJSON } from "/static/auth.js";

let track, cursor, controls;
let mode = "live";

export function initTimeline() {
  const wrap = document.getElementById("timeline-wrap");
  wrap.innerHTML = `
    <div class="tl-controls">
      <span class="counter" id="tl-counter">0 / 0</span>
      <span class="spacer"></span>
      <span id="tl-replay-controls" style="display:none">
        <button data-act="prev" title="Previous event">‹</button>
        <button data-act="play" id="tl-play">▶ play</button>
        <button data-act="next" title="Next event">›</button>
        <select id="tl-speed">
          <option value="1">1×</option>
          <option value="2">2×</option>
          <option value="5">5×</option>
          <option value="10">10×</option>
        </select>
      </span>
      <select id="tl-filter" title="Filter by event type">
        <option value="">all events</option>
        <option value="tool">tool calls only</option>
        <option value="model">model only</option>
      </select>
    </div>
    <div class="tl-track" id="tl-track">
      <div class="tl-cursor" id="tl-cursor" style="left:0"></div>
    </div>`;
  track = document.getElementById("tl-track");
  cursor = document.getElementById("tl-cursor");
  controls = document.getElementById("tl-replay-controls");

  for (const btn of wrap.querySelectorAll("button")) {
    btn.addEventListener("click", () => onControl(btn.dataset.act, btn));
  }
  document.getElementById("tl-speed").addEventListener("change", (e) => {
    postJSON("/api/control/speed", { speed: parseFloat(e.target.value) }).catch(() => {});
  });
  document.getElementById("tl-filter").addEventListener("change", redraw);

  on("metaLoaded", () => {
    mode = state.meta.mode || "live";
    controls.style.display = (mode === "replay") ? "" : "none";
  });
  on("eventArrived", redraw);
  on("select", redraw);
}

function onControl(act, btn) {
  if (act === "prev") return stepLocal(-1);
  if (act === "next") return stepLocal(+1);
  if (act === "play") {
    const playing = btn.dataset.playing === "1";
    btn.dataset.playing = playing ? "0" : "1";
    btn.textContent = playing ? "▶ play" : "❚❚ pause";
    postJSON("/api/control/" + (playing ? "pause" : "play"), {}).catch(() => {});
  }
}

function stepLocal(dir) {
  if (!state.events.length) return;
  let idx = state.events.findIndex(e => e._seq === state.selectedSeq);
  if (idx === -1) idx = state.events.length - 1;
  idx = Math.max(0, Math.min(state.events.length - 1, idx + dir));
  selectEvent(state.events[idx]._seq);
}

function redraw() {
  const filter = document.getElementById("tl-filter")?.value || "";
  const all = state.events;
  const shown = filter ? all.filter(e => (e.event_type || "").includes(filter)) : all;

  document.getElementById("tl-counter").textContent =
    `${shown.length} / ${all.length}`;

  // Re-render ticks (cheap; capped at 5000 in state)
  for (const t of [...track.querySelectorAll(".tl-tick")]) t.remove();
  if (!shown.length) {
    cursor.style.left = "0";
    return;
  }
  const w = track.clientWidth || 1;
  shown.forEach((ev) => {
    const seq = ev._seq;
    const xPct = ((seq - shown[0]._seq) / Math.max(1, shown[shown.length - 1]._seq - shown[0]._seq)) * 100;
    const tick = document.createElement("div");
    tick.className = `tl-tick t-${ev.event_type}`;
    if (seq === state.selectedSeq) tick.classList.add("selected");
    tick.style.left = `calc(${xPct}% - 1px)`;
    tick.title = `${ev.event_type}${ev.agent_name ? " · " + ev.agent_name : ""}`;
    tick.addEventListener("click", () => selectEvent(seq));
    track.appendChild(tick);
  });

  // Cursor on selected
  const sel = state.selectedSeq;
  if (sel != null && shown.length) {
    const range = Math.max(1, shown[shown.length - 1]._seq - shown[0]._seq);
    const xPct = ((sel - shown[0]._seq) / range) * 100;
    cursor.style.left = `${Math.max(0, Math.min(100, xPct))}%`;
  } else {
    cursor.style.left = "100%";
  }
}
