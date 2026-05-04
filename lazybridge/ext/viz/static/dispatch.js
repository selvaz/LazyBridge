// Translates raw events from the SSE stream into graph animations
// and store updates. The mapping from event_type -> visual gesture
// lives here so the rest of the UI stays presentation-only.

import { state, pushEvent, emit } from "/static/state.js";
import { pulse, fireHitRing, setInFlight, flashError, ensureToolNode } from "/static/graph.js";
import { getJSON } from "/static/auth.js";

const COLOR = {
  agent:  "#00d4ff",
  tool:   "#ff3ec9",
  router: "#ffb340",
  store:  "#50fa7b",
  error:  "#ff5577",
  think:  "#b388ff",
};

export function dispatch(ev) {
  pushEvent(ev);
  emit("eventArrived", ev);
  const t = ev.event_type;
  const agent = ev.agent_name;
  switch (t) {
    case "agent_start":
      if (agent) fireHitRing(agent, COLOR.store);
      break;
    case "agent_finish":
      if (agent) fireHitRing(agent, ev.error ? COLOR.error : COLOR.store);
      if (ev.error && agent) flashError(agent);
      break;
    case "model_request":
      if (agent) setInFlight(agent, true);
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
      // Refresh store — tool calls often write to it
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
async function refreshStore() {
  if (storePending) return;
  storePending = true;
  try {
    const data = await getJSON("/api/store");
    storePending = false;
    const prev = state.store || {};
    state.store = data || {};
    for (const k of Object.keys(state.store)) {
      if (JSON.stringify(prev[k]) !== JSON.stringify(state.store[k])) {
        state.storeFresh.set(k, performance.now());
      }
    }
    emit("storeChanged");
  } catch (e) {
    storePending = false;
  }
}
export { refreshStore };
