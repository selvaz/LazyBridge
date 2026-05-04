// Translates raw events from the SSE stream into graph animations
// and store updates. Also subscribes to "select" so that stepping
// through the timeline re-fires the visual gesture for that event.

import { state, on, pushEvent, emit } from "/static/state.js";
import {
  pulse, flashLink, fireHitRing, setInFlight, flashError,
  ensureToolNode, ensureStoreEdge, refreshNodeLabels,
} from "/static/graph.js";
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
    case "agent_start": {
      if (agent) fireHitRing(agent, COLOR.store);
      if (ev.session_id) state.sessionInfo.session_id = ev.session_id;
      if (agent && ev.task) {
        state.agentTasks.set(agent, ev.task);
        if (!state.pipelineTask) state.pipelineTask = ev.task;
        refreshNodeLabels();
      }
      // Pulse along incoming context edges — shows context arriving at this agent
      if (agent && !replay) {
        const dstNode = state.nodes.find(n => n.name === agent || n.id === agent);
        if (dstNode) {
          const inEdges = state.links.filter(l => {
            const tId = typeof l.target === "object" ? l.target.id : l.target;
            const k = l.kind || "";
            return tId === dstNode.id && (k === "context" || k === "router");
          });
          for (const edge of inEdges) {
            const srcId = typeof edge.source === "object" ? edge.source.id : edge.source;
            if (!srcId.startsWith("__")) pulse(srcId, dstNode.id, COLOR.store);
          }
        }
      }
      break;
    }

    case "agent_finish": {
      if (agent) fireHitRing(agent, ev.error ? COLOR.error : COLOR.store);
      if (ev.error && agent) flashError(agent);
      if (agent && (ev.result != null)) state.pipelineOutputs.set(agent, ev.result);
      // Pulse along every outgoing context/router edge
      if (agent && !ev.error) {
        const srcNode = state.nodes.find(n => n.name === agent || n.id === agent);
        if (srcNode) {
          const outEdges = state.links.filter(l => {
            const s = typeof l.source === "object" ? l.source.id : l.source;
            const k = l.kind || "";
            return s === srcNode.id && (k === "context" || k === "router" || k === "flow");
          });
          for (const edge of outEdges) {
            const dstId = typeof edge.target === "object" ? edge.target.id : edge.target;
            if (!dstId.startsWith("__")) pulse(srcNode.id, dstId, COLOR.agent);
          }
        }
      }
      break;
    }

    case "model_request":
      if (agent) setInFlight(agent, !replay);
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

    case "store_write": {
      const key = ev.key, val = ev.value;
      if (agent && key) {
        state.storeProvenance.set(key, { agent, ts: ev.ts || performance.now() / 1000 });
        ensureStoreEdge(agent, "__STORE__", "store_write");
        pulse(agent, "__STORE__", COLOR.store);
        emit("storeProvenanceChanged");
      }
      refreshStore();
      break;
    }

    case "store_read": {
      const key = ev.key;
      if (agent && key) {
        ensureStoreEdge("__STORE__", agent, "store_read");
        pulse("__STORE__", agent, COLOR.store);
        // Highlight the write edge from the original writer too
        const prov = state.storeProvenance.get(key);
        if (prov && prov.agent !== agent) {
          flashLink(prov.agent, "__STORE__");
        }
      }
      break;
    }

    case "memory_write": {
      if (agent && ev.key) {
        if (!state.memoryEntries.has(agent)) state.memoryEntries.set(agent, []);
        const entries = state.memoryEntries.get(agent);
        const idx = entries.findIndex(e => e.key === ev.key);
        const entry = { key: ev.key, value: ev.value, ts: ev.ts };
        if (idx >= 0) entries[idx] = entry; else entries.push(entry);
        fireHitRing(agent, COLOR.think);
        emit("memoryChanged");
      }
      break;
    }

    case "memory_read": {
      if (agent) fireHitRing(agent, COLOR.think);
      break;
    }
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
