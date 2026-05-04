"""Pipeline Visualizer — mock demo (no LLM calls).

Pipeline topology:
  researcher_a (web_search, news_api)  ──┐
                                          ├──► merger (combine, deduplicate) ──► writer (format_text)
  researcher_b (database_query)         ──┘
       └── nested: sub_researcher (fact_check, verify_url)

Features demonstrated:
  • Parallel agents (researcher_a and researcher_b run concurrently)
  • Nested agent (sub_researcher called as tool by researcher_b)
  • Multiple tools per agent
  • Live store writes (visible in Store tab + Node card)
  • Click any node → D&D character sheet with tools, stats, store entries
  • Drag to pin nodes, double-click to unpin
  • Play/step through events (Space / J / K)

Run::

    python examples/viz_mock_demo.py
"""

from __future__ import annotations

import threading
import time
import webbrowser

from lazybridge import Session
from lazybridge.session import EventType
from lazybridge.graph.schema import EdgeType
from lazybridge.ext.viz.exporter import EventHub, HubExporter
from lazybridge.ext.viz.server import VizServer

# ---------------------------------------------------------------------------
# Mock store — plain dict served via /api/store
# ---------------------------------------------------------------------------

_store: dict = {}


def _store_provider() -> dict:
    return dict(_store)


def _write(key: str, value, agent: str) -> None:
    _store[key] = {"value": value, "agent": agent, "ts": time.time()}


# ---------------------------------------------------------------------------
# Agent definitions (metadata only — no real engines)
# ---------------------------------------------------------------------------

AGENTS = [
    {"name": "researcher_a", "provider": "anthropic", "model": "claude-haiku-4-5",
     "system": "Search the web for recent news. Cite sources precisely.",
     "tools": ["web_search", "news_api"]},
    {"name": "researcher_b", "provider": "anthropic", "model": "claude-haiku-4-5",
     "system": "Query internal databases and verify facts via sub-agents.",
     "tools": ["database_query"]},
    {"name": "sub_researcher", "provider": "anthropic", "model": "claude-haiku-4-5",
     "system": "Verify individual claims. Return confidence scores.",
     "tools": ["fact_check", "verify_url"]},
    {"name": "merger", "provider": "anthropic", "model": "claude-haiku-4-5",
     "system": "Combine research from multiple sources, removing duplicates.",
     "tools": ["combine_results", "deduplicate"]},
    {"name": "writer", "provider": "anthropic", "model": "claude-haiku-4-5",
     "system": "Write polished executive briefs from structured data.",
     "tools": ["format_text"]},
]

# ---------------------------------------------------------------------------
# Mock pipeline event sequence
# ---------------------------------------------------------------------------

def _emit_parallel_research(sess: Session, run: str) -> None:
    """Parallel researcher_a and researcher_b (interleaved events)."""
    task = "Brief me on fusion energy breakthroughs in 2026."

    def p(s): time.sleep(s)

    # Both agents start roughly simultaneously
    sess.emit(EventType.AGENT_START, {"agent_name": "researcher_a", "task": task}, run_id=run)
    p(0.15)
    sess.emit(EventType.AGENT_START, {"agent_name": "researcher_b", "task": task}, run_id=run)
    p(0.3)

    # Both send MODEL_REQUEST
    sess.emit(EventType.MODEL_REQUEST, {
        "agent_name": "researcher_a", "model": "claude-haiku-4-5", "step": 1,
        "messages": [{"role": "user", "content": task}],
    }, run_id=run)
    p(0.1)
    sess.emit(EventType.MODEL_REQUEST, {
        "agent_name": "researcher_b", "model": "claude-haiku-4-5", "step": 1,
        "messages": [{"role": "user", "content": task}],
    }, run_id=run)
    p(0.6)

    # researcher_a responds first → web_search
    sess.emit(EventType.MODEL_RESPONSE, {
        "agent_name": "researcher_a", "model": "claude-haiku-4-5", "step": 1,
        "content": "Searching news for fusion energy...",
        "usage": {"input_tokens": 115, "output_tokens": 22},
    }, run_id=run)
    p(0.1)
    sess.emit(EventType.TOOL_CALL, {
        "agent_name": "researcher_a", "name": "web_search",
        "arguments": {"query": "ITER fusion energy Q>1 2026"},
    }, run_id=run)
    p(0.4)

    # researcher_b responds → database_query (still in parallel)
    sess.emit(EventType.MODEL_RESPONSE, {
        "agent_name": "researcher_b", "model": "claude-haiku-4-5", "step": 1,
        "content": "Querying internal DB for fusion R&D entries...",
        "usage": {"input_tokens": 118, "output_tokens": 24},
    }, run_id=run)
    p(0.1)
    sess.emit(EventType.TOOL_CALL, {
        "agent_name": "researcher_b", "name": "database_query",
        "arguments": {"table": "fusion_projects", "filter": "year=2026"},
    }, run_id=run)
    p(0.5)

    # researcher_a tool_result → news_api
    sess.emit(EventType.TOOL_RESULT, {
        "agent_name": "researcher_a", "name": "web_search",
        "result": "ITER Q=1.2 confirmed Jan 2026. Helion Series D closed at $2.8B.",
    }, run_id=run)
    p(0.1)
    sess.emit(EventType.MODEL_REQUEST, {
        "agent_name": "researcher_a", "model": "claude-haiku-4-5", "step": 2,
        "messages": [{"role": "user", "content": task}],
    }, run_id=run)
    p(0.5)
    sess.emit(EventType.MODEL_RESPONSE, {
        "agent_name": "researcher_a", "model": "claude-haiku-4-5", "step": 2,
        "content": "Getting latest headlines from news_api...",
        "usage": {"input_tokens": 220, "output_tokens": 18},
    }, run_id=run)
    sess.emit(EventType.TOOL_CALL, {
        "agent_name": "researcher_a", "name": "news_api",
        "arguments": {"topic": "fusion energy", "days": 30},
    }, run_id=run)

    # researcher_b tool_result → calls sub_researcher (nested agent)
    p(0.3)
    sess.emit(EventType.TOOL_RESULT, {
        "agent_name": "researcher_b", "name": "database_query",
        "result": "47 active fusion projects, 6 reached ignition threshold in 2026.",
    }, run_id=run)
    p(0.2)
    sess.emit(EventType.MODEL_REQUEST, {
        "agent_name": "researcher_b", "model": "claude-haiku-4-5", "step": 2,
        "messages": [{"role": "user", "content": task}],
    }, run_id=run)
    p(0.55)
    sess.emit(EventType.MODEL_RESPONSE, {
        "agent_name": "researcher_b", "model": "claude-haiku-4-5", "step": 2,
        "content": "Delegating fact verification to sub_researcher...",
        "usage": {"input_tokens": 195, "output_tokens": 20},
    }, run_id=run)
    # researcher_b calls sub_researcher as a tool
    sess.emit(EventType.TOOL_CALL, {
        "agent_name": "researcher_b", "name": "sub_researcher",
        "arguments": {"claim": "ITER achieved Q=1.2 in January 2026"},
    }, run_id=run)

    # --- sub_researcher runs (nested) ---
    p(0.2)
    sess.emit(EventType.AGENT_START, {
        "agent_name": "sub_researcher",
        "task": "Verify: ITER achieved Q=1.2 in January 2026",
    }, run_id=run)
    p(0.3)
    sess.emit(EventType.MODEL_REQUEST, {
        "agent_name": "sub_researcher", "model": "claude-haiku-4-5", "step": 1,
        "messages": [{"role": "user", "content": "Verify: ITER Q=1.2 Jan 2026"}],
    }, run_id=run)
    p(0.6)
    sess.emit(EventType.MODEL_RESPONSE, {
        "agent_name": "sub_researcher", "model": "claude-haiku-4-5", "step": 1,
        "content": "Running fact_check tool...",
        "usage": {"input_tokens": 88, "output_tokens": 14},
    }, run_id=run)
    sess.emit(EventType.TOOL_CALL, {
        "agent_name": "sub_researcher", "name": "fact_check",
        "arguments": {"claim": "ITER Q=1.2 January 2026", "sources": ["nature.com", "iter.org"]},
    }, run_id=run)
    p(0.5)
    sess.emit(EventType.TOOL_RESULT, {
        "agent_name": "sub_researcher", "name": "fact_check",
        "result": "VERIFIED (confidence 0.97). ITER press release Jan 14 2026 confirms Q=1.2.",
    }, run_id=run)
    p(0.15)
    sess.emit(EventType.TOOL_CALL, {
        "agent_name": "sub_researcher", "name": "verify_url",
        "arguments": {"url": "https://iter.org/news/2026-01-14-q12"},
    }, run_id=run)
    p(0.35)
    sess.emit(EventType.TOOL_RESULT, {
        "agent_name": "sub_researcher", "name": "verify_url",
        "result": "URL valid. Content matches claim. Confidence: 0.99.",
    }, run_id=run)
    p(0.2)
    sess.emit(EventType.MODEL_REQUEST, {
        "agent_name": "sub_researcher", "model": "claude-haiku-4-5", "step": 2,
        "messages": [{"role": "user", "content": "Verify: ITER Q=1.2 Jan 2026"}],
    }, run_id=run)
    p(0.55)
    sub_result = "VERIFIED: ITER Q=1.2 confirmed (confidence 0.98). Two independent sources."
    sess.emit(EventType.MODEL_RESPONSE, {
        "agent_name": "sub_researcher", "model": "claude-haiku-4-5", "step": 2,
        "content": sub_result,
        "usage": {"input_tokens": 152, "output_tokens": 28},
    }, run_id=run)
    _write("verification_result", sub_result, "sub_researcher")
    sess.emit(EventType.AGENT_FINISH, {
        "agent_name": "sub_researcher", "result": sub_result,
    }, run_id=run)

    # researcher_b gets sub_researcher result
    p(0.2)
    sess.emit(EventType.TOOL_RESULT, {
        "agent_name": "researcher_b", "name": "sub_researcher",
        "result": sub_result,
    }, run_id=run)

    # researcher_a news_api result arrives
    p(0.1)
    sess.emit(EventType.TOOL_RESULT, {
        "agent_name": "researcher_a", "name": "news_api",
        "result": "30 articles found. Top: 'Fusion Era Begins' (Nature, Jan 15), 'CFS magnet record' (Science, Feb 3).",
    }, run_id=run)
    p(0.25)

    # Both finalize
    sess.emit(EventType.MODEL_REQUEST, {
        "agent_name": "researcher_a", "model": "claude-haiku-4-5", "step": 3,
    }, run_id=run)
    sess.emit(EventType.MODEL_REQUEST, {
        "agent_name": "researcher_b", "model": "claude-haiku-4-5", "step": 3,
    }, run_id=run)
    p(0.7)

    ra_out = ("ITER Q=1.2 confirmed Jan 2026. Helion raised $2.8B. "
              "CFS set magnet field record. 30+ news articles in 30 days.")
    rb_out = ("DB: 47 active projects, 6 reached ignition. "
              "Verification: ITER claim confirmed 0.98 confidence. "
              "Private funding +340% YoY.")

    sess.emit(EventType.MODEL_RESPONSE, {
        "agent_name": "researcher_a", "model": "claude-haiku-4-5", "step": 3,
        "content": ra_out, "usage": {"input_tokens": 288, "output_tokens": 52},
    }, run_id=run)
    _write("research_a_findings", ra_out, "researcher_a")
    sess.emit(EventType.AGENT_FINISH, {"agent_name": "researcher_a", "result": ra_out}, run_id=run)
    p(0.1)
    sess.emit(EventType.MODEL_RESPONSE, {
        "agent_name": "researcher_b", "model": "claude-haiku-4-5", "step": 3,
        "content": rb_out, "usage": {"input_tokens": 305, "output_tokens": 58},
    }, run_id=run)
    _write("research_b_findings", rb_out, "researcher_b")
    sess.emit(EventType.AGENT_FINISH, {"agent_name": "researcher_b", "result": rb_out}, run_id=run)

    return ra_out, rb_out


def _emit_merger(sess: Session, run: str, ra: str, rb: str) -> str:
    p = lambda s: time.sleep(s)
    combined = f"{ra} | {rb}"
    sess.emit(EventType.AGENT_START, {"agent_name": "merger", "task": combined}, run_id=run)
    p(0.4)
    sess.emit(EventType.MODEL_REQUEST, {
        "agent_name": "merger", "model": "claude-haiku-4-5", "step": 1,
        "messages": [{"role": "user", "content": combined}],
    }, run_id=run)
    p(0.65)
    sess.emit(EventType.MODEL_RESPONSE, {
        "agent_name": "merger", "model": "claude-haiku-4-5", "step": 1,
        "content": "Combining and deduplicating...",
        "usage": {"input_tokens": 245, "output_tokens": 19},
    }, run_id=run)
    sess.emit(EventType.TOOL_CALL, {
        "agent_name": "merger", "name": "combine_results",
        "arguments": {"sources": ["researcher_a", "researcher_b"]},
    }, run_id=run)
    p(0.45)
    sess.emit(EventType.TOOL_RESULT, {
        "agent_name": "merger", "name": "combine_results",
        "result": "Merged 9 unique facts from 2 sources.",
    }, run_id=run)
    sess.emit(EventType.TOOL_CALL, {
        "agent_name": "merger", "name": "deduplicate",
        "arguments": {"threshold": 0.85},
    }, run_id=run)
    p(0.35)
    sess.emit(EventType.TOOL_RESULT, {
        "agent_name": "merger", "name": "deduplicate",
        "result": "Removed 2 duplicate entries. 7 unique facts remain.",
    }, run_id=run)
    p(0.3)
    sess.emit(EventType.MODEL_REQUEST, {
        "agent_name": "merger", "model": "claude-haiku-4-5", "step": 2,
    }, run_id=run)
    p(0.6)
    merger_out = ("7 verified facts: ITER Q=1.2 (verified 0.98), 47 active projects, "
                  "6 ignition events, Helion $2.8B, CFS magnet record, Nature+Science coverage, "
                  "private funding +340% YoY.")
    sess.emit(EventType.MODEL_RESPONSE, {
        "agent_name": "merger", "model": "claude-haiku-4-5", "step": 2,
        "content": merger_out, "usage": {"input_tokens": 320, "output_tokens": 61},
    }, run_id=run)
    _write("merged_facts", merger_out, "merger")
    sess.emit(EventType.AGENT_FINISH, {"agent_name": "merger", "result": merger_out}, run_id=run)
    return merger_out


def _emit_writer(sess: Session, run: str, facts: str) -> None:
    p = lambda s: time.sleep(s)
    sess.emit(EventType.AGENT_START, {"agent_name": "writer", "task": facts}, run_id=run)
    p(0.4)
    sess.emit(EventType.MODEL_REQUEST, {
        "agent_name": "writer", "model": "claude-haiku-4-5", "step": 1,
        "messages": [{"role": "user", "content": facts}],
    }, run_id=run)
    p(1.0)
    final = (
        "EXECUTIVE BRIEF — Fusion Energy 2026\n\n"
        "2026 marks the beginning of the fusion era. ITER achieved energy gain Q=1.2 "
        "in January, confirmed with 0.98 confidence from two independent sources. "
        "Among 47 tracked projects, 6 reached ignition threshold this year. "
        "Private investment surged +340% YoY: Helion raised $2.8B, Commonwealth Fusion "
        "set a new magnet field record. Pilot commercial plants are projected for 2028–2035."
    )
    sess.emit(EventType.MODEL_RESPONSE, {
        "agent_name": "writer", "model": "claude-haiku-4-5", "step": 1,
        "content": final, "usage": {"input_tokens": 198, "output_tokens": 95},
    }, run_id=run)
    sess.emit(EventType.TOOL_CALL, {
        "agent_name": "writer", "name": "format_text",
        "arguments": {"style": "executive_brief", "max_words": 120},
    }, run_id=run)
    p(0.4)
    sess.emit(EventType.TOOL_RESULT, {
        "agent_name": "writer", "name": "format_text",
        "result": "Formatted. Word count: 89. Readability score: A.",
    }, run_id=run)
    _write("final_brief", final, "writer")
    sess.emit(EventType.AGENT_FINISH, {"agent_name": "writer", "result": final}, run_id=run)


def _run_pipeline(sess: Session) -> None:
    run = "mock-run-001"
    print("[mock] pipeline starting...")
    ra, rb = _emit_parallel_research(sess, run)
    merged = _emit_merger(sess, run, ra, rb)
    _emit_writer(sess, run, merged)
    print(f"[mock] pipeline complete — store has {len(_store)} entries")
    print("[viz]  step through events with Space / J / K")
    print("[viz]  click any node to see its character sheet")
    print("[viz]  press Ctrl+C to stop")


# ---------------------------------------------------------------------------
# Graph registration
# ---------------------------------------------------------------------------

def _build_graph(sess):
    g = sess.graph

    class _M:
        def __init__(self, meta):
            self.name           = meta["name"]
            self._provider_name = meta["provider"]
            self._model_name    = meta["model"]
            self.engine         = None
            self.system         = meta.get("system", "")
            self.description    = self.system
            # Fake _tool_map so add_agent() pre-registers tool nodes
            self._tool_map      = {t: object() for t in meta.get("tools", [])}

    for meta in AGENTS:
        g.add_agent(_M(meta))

    # Pipeline edges (parallel merge pattern + nested call)
    g.add_edge("researcher_a",  "merger",       label="findings_a",  kind=EdgeType.CONTEXT)
    g.add_edge("researcher_b",  "merger",       label="findings_b",  kind=EdgeType.CONTEXT)
    g.add_edge("researcher_b",  "sub_researcher", label="verify",     kind=EdgeType.TOOL)
    g.add_edge("merger",        "writer",       label="merged_facts", kind=EdgeType.CONTEXT)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    sess = Session(db="examples/viz_mock.db", console=False)
    _build_graph(sess)

    hub      = EventHub()
    exporter = HubExporter(hub)
    sess.add_exporter(exporter)

    server = VizServer(
        hub,
        graph_provider=lambda: sess.graph.to_dict(),
        store_provider=_store_provider,
        meta_provider=lambda: {"mode": "live", "session_id": sess.session_id},
        host="127.0.0.1",
        port=0,
    )
    server.start()
    url = server.url
    webbrowser.open(url, new=2)
    print(f"[viz] open -> {url}")

    # Let the browser connect before events start
    time.sleep(2.2)

    t = threading.Thread(target=_run_pipeline, args=(sess,), daemon=True)
    t.start()

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        sess.close()


if __name__ == "__main__":
    main()
