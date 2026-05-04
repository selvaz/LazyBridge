// Translates raw events from the SSE stream into graph animations
// and store updates. Also subscribes to "select" so that stepping
// through the timeline re-fires the visual gesture for that event.

import { state, on, pushEvent, emit } from "/static/state.js";
import { pulse, fireHitRing, setInFlight, flashError, ensureToolNode, refreshNodeLabels } from "/static/graph.js";
import { getJSON } from "/static/auth.js";

const COLOR = {
  agent:  "#00d4ff",
  tool:   "#ff3ec9",
  router: "#ffb340",
  store:  "#50fa7b",
  error:  "#ff5577",
  think:  "#b388ff",
};

// Called by the SSE stream for every incoming event
export function dispatch(ev) {
  pushEvent(ev);
  emit("eventArrived", ev);
  _gesture(ev);
}

// Called when the user navigates the timeline (play / step / click)
on("select", (seq) => {
  const ev = state.byId.get(seq);
  if (ev) _gesture(ev, { replay: true });
});

// Visual gesture for a single event. In replay mode we still fire
// setInFlight(false) on model_request so we don't leave ghosts.
function _gesture(ev, { replay = false } = {}) {
  const t     = ev.event_type;
  const agent = ev.agent_name;

  switch (t) {
    case "agent_start":
      if (agent) fireHitRing(agent, COLOR.store);
      if (agent && ev.task) {
        state.agentTasks.set(agent, ev.task);
        if (!state.pipelineTask) state.pipelineTask = ev.task;
        refreshNodeLabels();
      }
      break;

    case "agent_finish":
      if (agent) fireHitRing(agent, ev.error ? COLOR.error : COLOR.store);
      if (ev.error && agent) flashError(agent);
      if (agent && (ev.result != null)) state.pipelineOutputs.set(agent, ev.result);
      break;

    case "model_request":
      if (agent) setInFlight(agent, !replay); // don't leave in-flight on replay
      if (agent && replay) fireHitRing(agent, COLOR.think);
      break;

    case "model_response":
      if (agent) {
        setInFlight(agent, false);
        fireHitRing(agent, COLOR.agent);
      }
      break;

    case "loop_step":
      if (agent) fireHitRing(agent, COLOR.think);
      break;

    case "tool_call": {
      const name = ev.name;
      if (!name || !agent) break;
      const toolId = ensureToolNode(name, agent);
      pulse(agent, toolId, COLOR.tool);
      break;
    }

    case "tool_result": {
      const name = ev.name;
      if (!name || !agent) break;
      const toolId = `tool:${name}`;
      pulse(toolId, agent, COLOR.tool);
      refreshStore();
      break;
    }

    case "tool_error": {
      const name = ev.name;
      if (name) flashError(`tool:${name}`);
      if (agent) flashError(agent);
      break;
    }

    case "hil_decision":
      if (agent) fireHitRing(agent, COLOR.router);
      break;
  }
}

let storePending = false;
export async function refreshStore() {
  if (storePending) return;
  storePending = true;
  try {
    const data   = await getJSON("/api/store");
    const prev   = state.store || {};
    state.store  = data || {};
    for (const k of Object.keys(state.store)) {
      if (JSON.stringify(prev[k]) !== JSON.stringify(state.store[k])) {
        state.storeFresh.set(k, performance.now());
      }
    }
    emit("storeChanged");
  } finally {
    storePending = false;
  }
}
