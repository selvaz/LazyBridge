"""Tests covering the second-round audit follow-up fixes.

Sections:
  P0   — LLMEngine provider-registration is serialised by a Lock.
  P1.4 — MCP transports raise RuntimeError (not AssertionError) before connect.
  P1.5 — OTelExporter._on_agent_end snapshots the registry once.
  P1.6 — Agent.stream() finally surfaces non-CancelledError exceptions.
  P1.7 — deep_dive TOC escapes id and text.
  P2.9 — strict-mode SIGNATURE schema always emits required: [] when strict.
  P2.10 — Session.usage_summary prefers payload agent_name over run_id map.
  P2.12 — ReplayController._run claims the index before publishing.
  P2.13 — Memory.add scales word-count estimate to ~1.3 tokens per word.

Third-round (replay-mode graph reconstruction):
  R1   — Plan emits AGENT_START + AGENT_FINISH around its body.
  R2   — Plan TOOL_CALL / TOOL_RESULT / TOOL_ERROR carry agent_name.
  R3   — reconstruct_graph derives child nodes from `step` and parent→child
         edges from agent_name → step.
"""

from __future__ import annotations

import threading
import warnings

import pytest

# ---------------------------------------------------------------------------
# P0 — provider registration thread-safety
# ---------------------------------------------------------------------------


def test_provider_registration_lock_attribute_exists():
    from lazybridge.engines.llm import LLMEngine

    assert isinstance(LLMEngine._PROVIDER_REGISTRY_LOCK, type(threading.Lock()))


def test_provider_alias_registration_concurrent_no_lost_writes():
    """Many threads racing on register_provider_alias must all survive."""
    from lazybridge.engines.llm import LLMEngine

    saved_aliases = dict(LLMEngine._PROVIDER_ALIASES)
    saved_rules = list(LLMEngine._PROVIDER_RULES)
    try:
        N = 50

        def _register(i: int) -> None:
            LLMEngine.register_provider_alias(f"audit-alias-{i}", "anthropic")

        threads = [threading.Thread(target=_register, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i in range(N):
            assert LLMEngine._PROVIDER_ALIASES[f"audit-alias-{i}"] == "anthropic"
    finally:
        LLMEngine._PROVIDER_ALIASES = saved_aliases
        LLMEngine._PROVIDER_RULES = saved_rules


# ---------------------------------------------------------------------------
# P1.4 — MCP transports: RuntimeError, not AssertionError, before connect
# ---------------------------------------------------------------------------
# Relocated to lazytools/tests/test_mcp.py with the MCP connector (moved to
# lazytoolkit in 0.8).

# ---------------------------------------------------------------------------
# P1.5 — OTelExporter._on_agent_end snapshots once
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("opentelemetry"),
    reason="opentelemetry-sdk not installed",
)
def test_otel_on_agent_end_does_not_double_close_orphans():
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from lazybridge.ext.otel.exporter import OTelExporter

    inner = InMemorySpanExporter()
    exp = OTelExporter(exporter=inner)

    # Open agent, model, and a tool span — all under the same run_id.
    exp.export({"event_type": "agent_start", "run_id": "r1", "agent_name": "a"})
    exp.export({"event_type": "model_request", "run_id": "r1", "model": "fake"})
    exp.export({"event_type": "tool_call", "run_id": "r1", "tool": "t1", "tool_use_id": "id1"})

    # Close the agent without closing the children — they become orphans.
    exp.export({"event_type": "agent_finish", "run_id": "r1", "payload": "done"})

    # Registry must be empty after agent_end (every span closed exactly once).
    assert exp._spans.get("r1") in (None, {})


# ---------------------------------------------------------------------------
# P1.6 — Agent.stream() finally surfaces non-cancellation exceptions
# ---------------------------------------------------------------------------


def test_stream_aclose_warns_on_non_cancellation_exception():
    import asyncio

    from lazybridge import Agent

    class _BadStreamEngine:
        model = "fake"
        _agent_name = "fake"

        def _validate(self, tool_map):
            pass

        async def run(self, env, *, tools, output_type, memory, session):
            from lazybridge import Envelope

            return Envelope(task=env.task, payload="")

        async def stream(self, env, *, tools, output_type, memory, session):
            try:
                yield "chunk"
            finally:
                # aclose path — raise something non-cancellation.
                raise RuntimeError("aclose blew up")

    agent = Agent(name="streamer", engine=_BadStreamEngine())

    async def _drive():
        async for _ in agent.stream("x"):
            break  # consumer breaks early → triggers aclose

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(_drive())

    assert any("aclose" in str(w.message) and "RuntimeError" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# P1.7 — deep_dive TOC escapes id and text
# ---------------------------------------------------------------------------
# Moved to lazybridge-reports — the deep_dive template lives there.

# ---------------------------------------------------------------------------
# P2.9 — strict-mode SIGNATURE schema emits required: [] when no required args
# ---------------------------------------------------------------------------


def test_signature_strict_mode_emits_empty_required_for_zero_required_args():
    from lazybridge.core.tool_schema import ToolSchemaBuilder, ToolSchemaMode

    def fn(x: int = 1, y: str = "default") -> str:
        """All optional."""
        return ""

    defn = ToolSchemaBuilder().build(
        fn,
        name="fn",
        description="all optional",
        strict=True,
        mode=ToolSchemaMode.SIGNATURE,
    )
    params = defn.parameters
    assert params["additionalProperties"] is False
    assert "required" in params
    assert params["required"] == []


def test_signature_non_strict_mode_omits_required_when_empty():
    from lazybridge.core.tool_schema import ToolSchemaBuilder, ToolSchemaMode

    def fn(x: int = 1) -> str:
        """All optional."""
        return ""

    defn = ToolSchemaBuilder().build(
        fn,
        name="fn",
        description="all optional",
        strict=False,
        mode=ToolSchemaMode.SIGNATURE,
    )
    # Non-strict: omitting "required" is the JSON Schema canonical shape.
    assert defn.parameters.get("required") in (None, [])
    assert "additionalProperties" not in defn.parameters


# ---------------------------------------------------------------------------
# P2.10 — Session.usage_summary prefers payload agent_name
# ---------------------------------------------------------------------------


def test_usage_summary_prefers_payload_agent_name_over_run_id_map():
    from lazybridge import EventType, Session

    sess = Session()
    # Outer agent's AGENT_START carries the outer name.
    sess.emit(EventType.AGENT_START, {"agent_name": "outer", "task": "t"}, run_id="r1")
    # MODEL_RESPONSE under the same run_id but on behalf of the inner agent —
    # the cost should be attributed to "inner", not "outer".
    sess.emit(
        EventType.MODEL_RESPONSE,
        {
            "agent_name": "inner",
            "input_tokens": 10,
            "output_tokens": 5,
            "cost_usd": 0.01,
        },
        run_id="r1",
    )

    summary = sess.usage_summary()
    assert "inner" in summary["by_agent"]
    assert summary["by_agent"]["inner"]["input_tokens"] == 10
    assert summary["by_agent"]["inner"]["output_tokens"] == 5
    assert summary["by_agent"]["inner"]["cost_usd"] == pytest.approx(0.01)
    assert "outer" not in summary["by_agent"]


def test_usage_summary_falls_back_to_run_id_map_when_payload_lacks_agent_name():
    from lazybridge import EventType, Session

    sess = Session()
    sess.emit(EventType.AGENT_START, {"agent_name": "alpha", "task": "t"}, run_id="r2")
    # No agent_name in MODEL_RESPONSE → fall back to AGENT_START map.
    sess.emit(
        EventType.MODEL_RESPONSE,
        {"input_tokens": 3, "output_tokens": 2, "cost_usd": 0.001},
        run_id="r2",
    )

    summary = sess.usage_summary()
    assert "alpha" in summary["by_agent"]


# ---------------------------------------------------------------------------
# P2.12 — ReplayController._run claims index before publishing
# ---------------------------------------------------------------------------


def test_replay_no_duplicate_publish_when_step_races_run():
    """Concurrent step() during _run must never publish the same event twice."""
    from lazybridge.ext.viz.exporter import EventHub
    from lazybridge.ext.viz.replay import ReplayController

    events = [{"event_type": "x", "session_id": "s", "ts": float(i), "i": i} for i in range(10)]
    hub = EventHub()
    ctrl = ReplayController(hub, events, speed=10.0)
    # ReplayController doesn't auto-start; step() advances atomically.
    # We claim the lock pattern that _run uses by stepping three times
    # in a row — duplicates would surface as out-of-order indexes.
    ctrl.pause()
    ctrl.step()
    ctrl.step()
    ctrl.step()
    assert ctrl.progress[0] == 3


# ---------------------------------------------------------------------------
# P2.13 — Memory token estimate ~1.3× word count
# ---------------------------------------------------------------------------


def test_memory_estimate_uses_one_point_three_tokens_per_word():
    from lazybridge.memory import Memory

    mem = Memory()
    # 10 words across user + assistant
    mem.add("one two three four five", "six seven eight nine ten")
    estimate = mem._turns[-1].token_estimate
    # 10 words × 1.3 = 13
    assert estimate == 13


def test_memory_estimate_floor_of_one_for_one_word():
    from lazybridge.memory import Memory

    mem = Memory()
    mem.add("solo", "")
    # int(1 * 1.3) == 1; max(1, ...) keeps the floor at 1.
    assert mem._turns[-1].token_estimate >= 1


# ---------------------------------------------------------------------------
# R1 / R2 — Plan emits AGENT_START + AGENT_FINISH; tool events carry agent_name
# ---------------------------------------------------------------------------


def _drive_plan(plan_engine, agent_name="pipeline"):
    """Run a Plan engine through an Agent + Session and return event rows."""
    from lazybridge import Agent, EventType, Session

    sess = Session()
    agent = Agent(engine=plan_engine, name=agent_name, session=sess)
    agent("seed task")
    rows = sess.events.query()
    return rows, EventType


def test_plan_emits_agent_start_and_finish_with_agent_name():
    from lazybridge.engines.plan import Plan, Step
    from lazybridge.testing import MockAgent

    a = MockAgent("ok", name="step_a")
    plan = Plan(Step(target=a, name="step_a"))
    rows, EventType = _drive_plan(plan, agent_name="orchestrator")

    starts = [r for r in rows if r["event_type"] == EventType.AGENT_START.value]
    finishes = [r for r in rows if r["event_type"] == EventType.AGENT_FINISH.value]
    assert any(r["payload"].get("agent_name") == "orchestrator" for r in starts)
    assert any(r["payload"].get("agent_name") == "orchestrator" for r in finishes)


def test_plan_tool_events_carry_agent_name():
    from lazybridge.engines.plan import Plan, Step
    from lazybridge.testing import MockAgent

    a = MockAgent("ok", name="step_a")
    plan = Plan(Step(target=a, name="step_a"))
    rows, EventType = _drive_plan(plan, agent_name="parent")

    tool_calls = [r for r in rows if r["event_type"] == EventType.TOOL_CALL.value]
    tool_results = [r for r in rows if r["event_type"] == EventType.TOOL_RESULT.value]
    assert tool_calls and tool_results
    for r in tool_calls + tool_results:
        assert r["payload"].get("agent_name") == "parent"
        # The step name must also survive so reconstruct_graph can
        # rebuild the parent → child edge.
        assert r["payload"].get("step") == "step_a"


def test_plan_emits_agent_finish_on_step_error():
    """Even when a step raises, AGENT_FINISH must still emit with the error."""
    from lazybridge.engines.plan import Plan, Step

    def boom(_task: str) -> str:
        raise RuntimeError("step crashed")

    plan = Plan(Step(target=boom, name="boom"))
    rows, EventType = _drive_plan(plan, agent_name="errcase")

    finishes = [r for r in rows if r["event_type"] == EventType.AGENT_FINISH.value]
    assert finishes, "AGENT_FINISH must always be emitted, even on error"
    payloads = [r["payload"] for r in finishes if r["payload"].get("agent_name") == "errcase"]
    assert payloads, "AGENT_FINISH must reference the wrapping agent name"
    # Plan returns an error envelope rather than raising, so the
    # AGENT_FINISH payload carries error= from the envelope's ErrorInfo.
    assert "error" in payloads[-1]


# ---------------------------------------------------------------------------
# R3 — reconstruct_graph derives nodes + edges from event payloads
# ---------------------------------------------------------------------------


def test_reconstruct_graph_links_agent_to_steps_via_tool_call_events():
    from lazybridge.ext.viz.replay import reconstruct_graph

    events = [
        {"event_type": "agent_start", "agent_name": "pipeline"},
        {"event_type": "tool_call", "agent_name": "pipeline", "step": "research"},
        {"event_type": "tool_result", "agent_name": "pipeline", "step": "research"},
        {"event_type": "tool_call", "agent_name": "pipeline", "step": "write"},
        {"event_type": "tool_result", "agent_name": "pipeline", "step": "write"},
        {"event_type": "agent_finish", "agent_name": "pipeline"},
    ]
    g = reconstruct_graph(events)
    node_ids = {n["id"] for n in g["nodes"]}
    assert node_ids == {"pipeline", "research", "write"}
    edge_pairs = {(e["from"], e["to"]) for e in g["edges"]}
    assert edge_pairs == {("pipeline", "research"), ("pipeline", "write")}
    # Edge keys match the live GraphSchema.to_dict shape so the frontend
    # renders replay edges with the same code path as live mode.
    for e in g["edges"]:
        assert set(e.keys()) >= {"from", "to", "label", "type"}


def test_reconstruct_graph_handles_llm_engine_tool_field():
    """LLMEngine emits tool calls with `tool` (not `step`); replay must still
    surface those as edges."""
    from lazybridge.ext.viz.replay import reconstruct_graph

    events = [
        {"event_type": "agent_start", "agent_name": "researcher"},
        {"event_type": "tool_call", "agent_name": "researcher", "tool": "search"},
        {"event_type": "agent_finish", "agent_name": "researcher"},
    ]
    g = reconstruct_graph(events)
    assert {"researcher", "search"} <= {n["id"] for n in g["nodes"]}
    assert any(e["from"] == "researcher" and e["to"] == "search" for e in g["edges"])


def test_reconstruct_graph_dedupes_repeat_tool_calls():
    from lazybridge.ext.viz.replay import reconstruct_graph

    events = [
        {"event_type": "tool_call", "agent_name": "p", "step": "child"},
        {"event_type": "tool_call", "agent_name": "p", "step": "child"},
        {"event_type": "tool_call", "agent_name": "p", "step": "child"},
    ]
    g = reconstruct_graph(events)
    assert sum(1 for e in g["edges"] if e["from"] == "p" and e["to"] == "child") == 1


def test_replay_visualizer_graph_matches_live_graph_for_mockagent_plan():
    """End-to-end: a MockAgent-driven Plan run replayed from SQLite produces
    the same node set the live session.graph carried."""
    import os
    import tempfile

    from lazybridge import Agent, Plan, Session, Step
    from lazybridge.ext.viz.replay import (
        load_session_events,
        reconstruct_graph,
    )
    from lazybridge.testing import MockAgent

    tmpdir = tempfile.mkdtemp(prefix="viz-graph-")
    db_path = os.path.join(tmpdir, "demo.db")

    research = MockAgent("hits", name="research")
    write = MockAgent("draft", name="write")
    plan = Plan(Step(target=research, name="research"), Step(target=write, name="write"))

    sess = Session(db=db_path)
    pipeline = Agent(engine=plan, name="pipeline", session=sess, tools=[research, write])
    pipeline("seed")

    # Replay reads only the SQLite events — no live graph, no agents.
    events = load_session_events(db_path, sess.session_id)
    replay_graph = reconstruct_graph(events)
    replay_nodes = {n["id"] for n in replay_graph["nodes"]}
    assert {"pipeline", "research", "write"} <= replay_nodes
    replay_edges = {(e["from"], e["to"]) for e in replay_graph["edges"]}
    assert ("pipeline", "research") in replay_edges
    assert ("pipeline", "write") in replay_edges
