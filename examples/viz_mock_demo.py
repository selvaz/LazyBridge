"""Pipeline Visualizer — mock demo (no LLM calls needed).

Simulates a 3-agent research pipeline with two tools and a shared
blackboard Store so you can see every visual effect: node pulses,
edge animations, inspector payloads, store writes, and the timeline
— without burning API credits.

Run::

    python examples/viz_mock_demo.py
"""

from __future__ import annotations

import threading
import time

from lazybridge import Session
from lazybridge.session import EventType
from lazybridge.graph.schema import EdgeType
from lazybridge.ext.viz import Visualizer

# ---------------------------------------------------------------------------
# Fake store — a plain dict the viz serves via /api/store
# ---------------------------------------------------------------------------

_store: dict = {}


def _store_provider() -> dict:
    return dict(_store)


def _store_write(key: str, value: str, agent: str) -> None:
    _store[key] = {"value": value, "agent": agent, "ts": time.time()}


# ---------------------------------------------------------------------------
# Pipeline metadata
# ---------------------------------------------------------------------------

AGENTS = [
    {"name": "researcher", "provider": "anthropic", "model": "claude-haiku-4-5",
     "system": "Find facts. Cite sources."},
    {"name": "analyst",    "provider": "anthropic", "model": "claude-haiku-4-5",
     "system": "Summarise findings into key metrics."},
    {"name": "writer",     "provider": "anthropic", "model": "claude-haiku-4-5",
     "system": "Write a short executive brief."},
]

TOOLS = {
    "researcher": [
        {"name": "web_search",
         "args": {"query": "fusion energy 2026 ITER progress"},
         "result": "ITER achieved Q=1.2 in January 2026 — net energy gain confirmed."},
        {"name": "web_search",
         "args": {"query": "private fusion startups funding 2026"},
         "result": "Helion, Commonwealth Fusion, TAE raised Series D. Helion targeting 2028 ignition."},
    ],
    "analyst": [
        {"name": "summarise",
         "args": {"text": "ITER Q=1.2 breakeven + private sector surge..."},
         "result": "Scientific breakeven achieved. 3 companies targeting 2028-2035 pilot plants."},
    ],
}

FINAL_BRIEF = (
    "Fusion energy crossed a landmark threshold in early 2026 when ITER achieved "
    "energy gain Q=1.2, proving net energy output at scale. The private sector "
    "responded swiftly: Helion, CFS, and TAE each closed major funding rounds. "
    "Pilot commercial plants are expected 2028-2035."
)


# ---------------------------------------------------------------------------
# Fake pipeline event emitter
# ---------------------------------------------------------------------------

def _run_mock_pipeline(sess: Session) -> None:
    run = "mock-run-001"
    task = "Brief me on the state of fusion energy in 2026."

    def p(s: float) -> None:
        time.sleep(s)

    # ── researcher ──────────────────────────────────────────────────────────
    sess.emit(EventType.AGENT_START, {"agent_name": "researcher", "task": task}, run_id=run)
    p(0.5)

    for i, tool in enumerate(TOOLS["researcher"]):
        sess.emit(EventType.MODEL_REQUEST, {
            "agent_name": "researcher",
            "messages": [{"role": "user", "content": task}],
            "model": "claude-haiku-4-5", "step": i + 1,
        }, run_id=run)
        p(0.7)
        sess.emit(EventType.MODEL_RESPONSE, {
            "agent_name": "researcher",
            "content": f"Searching for: {tool['args']['query']}",
            "model": "claude-haiku-4-5",
            "usage": {"input_tokens": 120, "output_tokens": 28}, "step": i + 1,
        }, run_id=run)
        p(0.2)
        sess.emit(EventType.TOOL_CALL, {
            "agent_name": "researcher", "name": tool["name"], "arguments": tool["args"],
        }, run_id=run)
        p(0.6)
        sess.emit(EventType.TOOL_RESULT, {
            "agent_name": "researcher", "name": tool["name"], "result": tool["result"],
        }, run_id=run)
        # Write intermediate findings to store
        _store_write(f"research_hit_{i+1}", tool["result"], "researcher")
        p(0.3)

    # Final researcher response
    sess.emit(EventType.MODEL_REQUEST, {
        "agent_name": "researcher",
        "messages": [{"role": "user", "content": task}],
        "model": "claude-haiku-4-5", "step": 3,
    }, run_id=run)
    p(0.9)
    researcher_output = (
        "ITER achieved Q=1.2 (Jan 2026). Private sector: Helion, CFS, TAE "
        "all targeting 2028-2035 ignition/pilot plants."
    )
    sess.emit(EventType.MODEL_RESPONSE, {
        "agent_name": "researcher", "content": researcher_output,
        "model": "claude-haiku-4-5",
        "usage": {"input_tokens": 310, "output_tokens": 55}, "step": 3,
    }, run_id=run)
    _store_write("researcher_findings", researcher_output, "researcher")
    sess.emit(EventType.AGENT_FINISH, {
        "agent_name": "researcher", "result": researcher_output,
    }, run_id=run)
    p(0.4)

    # ── analyst ─────────────────────────────────────────────────────────────
    sess.emit(EventType.AGENT_START, {
        "agent_name": "analyst", "task": researcher_output,
    }, run_id=run)
    p(0.5)

    tool = TOOLS["analyst"][0]
    sess.emit(EventType.MODEL_REQUEST, {
        "agent_name": "analyst",
        "messages": [{"role": "user", "content": researcher_output}],
        "model": "claude-haiku-4-5", "step": 1,
    }, run_id=run)
    p(0.6)
    sess.emit(EventType.MODEL_RESPONSE, {
        "agent_name": "analyst", "content": "Let me summarise these findings.",
        "model": "claude-haiku-4-5",
        "usage": {"input_tokens": 180, "output_tokens": 18}, "step": 1,
    }, run_id=run)
    p(0.2)
    sess.emit(EventType.TOOL_CALL, {
        "agent_name": "analyst", "name": tool["name"], "arguments": tool["args"],
    }, run_id=run)
    p(0.5)
    sess.emit(EventType.TOOL_RESULT, {
        "agent_name": "analyst", "name": tool["name"], "result": tool["result"],
    }, run_id=run)
    p(0.3)

    sess.emit(EventType.MODEL_REQUEST, {
        "agent_name": "analyst",
        "messages": [{"role": "user", "content": researcher_output}],
        "model": "claude-haiku-4-5", "step": 2,
    }, run_id=run)
    p(0.8)
    analyst_output = tool["result"]
    sess.emit(EventType.MODEL_RESPONSE, {
        "agent_name": "analyst", "content": analyst_output,
        "model": "claude-haiku-4-5",
        "usage": {"input_tokens": 240, "output_tokens": 44}, "step": 2,
    }, run_id=run)
    _store_write("analyst_summary", analyst_output, "analyst")
    _store_write("metrics", {"breakeven": True, "companies": 3, "timeline": "2028-2035"}, "analyst")
    sess.emit(EventType.AGENT_FINISH, {
        "agent_name": "analyst", "result": analyst_output,
    }, run_id=run)
    p(0.4)

    # ── writer ───────────────────────────────────────────────────────────────
    sess.emit(EventType.AGENT_START, {
        "agent_name": "writer", "task": analyst_output,
    }, run_id=run)
    p(0.5)
    sess.emit(EventType.MODEL_REQUEST, {
        "agent_name": "writer",
        "messages": [{"role": "user", "content": analyst_output}],
        "model": "claude-haiku-4-5", "step": 1,
    }, run_id=run)
    p(1.1)
    sess.emit(EventType.MODEL_RESPONSE, {
        "agent_name": "writer", "content": FINAL_BRIEF,
        "model": "claude-haiku-4-5",
        "usage": {"input_tokens": 195, "output_tokens": 82}, "step": 1,
    }, run_id=run)
    _store_write("final_brief", FINAL_BRIEF, "writer")
    sess.emit(EventType.AGENT_FINISH, {
        "agent_name": "writer", "result": FINAL_BRIEF,
    }, run_id=run)

    print("[mock] pipeline complete — store has", len(_store), "entries")
    print("[viz]  watching... press Ctrl+C to stop")


# ---------------------------------------------------------------------------
# Graph registration
# ---------------------------------------------------------------------------

def _build_graph(sess: Session) -> None:
    """Register mock agents + their tools so the full graph is visible
    before any events are emitted — same as real Agent registration."""
    g = sess.graph

    class _MockAgent:
        def __init__(self, meta: dict, tools: dict | None = None) -> None:
            self.name           = meta["name"]
            self._provider_name = meta["provider"]
            self._model_name    = meta["model"]
            self.engine         = None
            self.description    = meta.get("system", "")
            # _tool_map triggers tool-node pre-registration in add_agent()
            self._tool_map      = tools or {}

    def _stub(name): return type(name, (), {"__name__": name})()

    g.add_agent(_MockAgent(AGENTS[0], {"web_search": _stub("web_search")}))
    g.add_agent(_MockAgent(AGENTS[1], {"summarise":  _stub("summarise")}))
    g.add_agent(_MockAgent(AGENTS[2]))

    g.add_edge("researcher", "analyst", label="findings", kind=EdgeType.CONTEXT)
    g.add_edge("analyst",    "writer",  label="summary",  kind=EdgeType.CONTEXT)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    sess = Session(db="examples/viz_mock.db", console=False)
    _build_graph(sess)

    # Inject custom store provider so the viz serves our mock store
    viz = Visualizer.__new__(Visualizer)
    viz._session  = sess
    viz._store    = None
    from lazybridge.ext.viz.exporter import EventHub, HubExporter
    from lazybridge.ext.viz.server import VizServer
    viz._hub      = EventHub()
    viz._exporter = HubExporter(viz._hub)
    viz._mode     = "live"
    viz._replay   = None
    sess.add_exporter(viz._exporter)
    viz._server   = VizServer(
        viz._hub,
        graph_provider=lambda: sess.graph.to_dict(),
        store_provider=_store_provider,
        meta_provider=lambda: {"mode": "live", "session_id": sess.session_id},
        host="127.0.0.1",
        port=0,
    )
    viz._auto_open = True
    viz._opened    = False

    viz.start()

    import webbrowser
    webbrowser.open(viz._server.url, new=2)
    print(f"[viz] open -> {viz._server.url}")

    # Give the browser 2 s to connect before events start firing
    time.sleep(2.0)

    t = threading.Thread(target=_run_mock_pipeline, args=(sess,), daemon=True)
    t.start()

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        viz.stop()
        sess.close()


if __name__ == "__main__":
    main()
