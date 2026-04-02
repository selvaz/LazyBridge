"""Intensive live tests — multi-agent, memory, context, tracking, nested pipelines.

All tests use the LazyBridgeFramework public API exclusively (LazyAgent,
LazySession, LazyContext, LazyStore, Memory, LazyTool).  No raw SDK calls.

Cheapest capable model used throughout to minimise API costs.
Run with: pytest -m live tests/live/test_multi_agent_live.py -v
"""

from __future__ import annotations

import pytest

from lazybridge import (
    LazyAgent,
    LazyContext,
    LazySession,
    LazyStore,
    LazyTool,
    Memory,
    TrackLevel,
)
from tests.live.helpers import require_live_provider

# Haiku is the cheapest model capable enough for these integration scenarios.
_MODEL = "claude-haiku-4-5-20251001"
_PROVIDER = "anthropic"


def _agent(**kwargs) -> LazyAgent:
    require_live_provider(_PROVIDER)
    return LazyAgent(_PROVIDER, model=_MODEL, verbose=True, **kwargs)


# ── T.MA.01 — Memory: agent recalls earlier turns ─────────────────────────────

@pytest.mark.live
def test_memory_multi_turn():
    """Agent must use Memory to carry context across three turns."""
    agent = _agent(name="memory_agent")
    mem = Memory()

    agent.chat("My name is Marco and I like Python.", memory=mem)
    agent.chat("I also have a dog named Rocky.", memory=mem)
    resp = agent.chat("What is my name and my dog's name?", memory=mem)

    content = resp.content.lower()
    assert "marco" in content, f"Name not recalled: {resp.content!r}"
    assert "rocky" in content, f"Dog name not recalled: {resp.content!r}"


# ── T.MA.02 — Memory: history length grows correctly ─────────────────────────

@pytest.mark.live
def test_memory_history_length():
    """Memory must accumulate exactly N user+assistant pairs after N calls."""
    agent = _agent(name="counter_agent")
    mem = Memory()

    for i in range(3):
        agent.chat(f"Turn {i + 1}: say only 'OK'.", memory=mem)

    # Each chat() appends one user + one assistant message → 3×2 = 6
    assert len(mem) == 6, f"Expected 6 messages, got {len(mem)}: {mem.history}"


# ── T.MA.03 — LazyContext.from_agent: B sees A's last output ──────────────────

@pytest.mark.live
def test_context_from_agent():
    """Agent B must receive Agent A's output via LazyContext.from_agent."""
    a = _agent(name="producer")
    b = _agent(name="consumer")

    a.chat("Output exactly this sentence and nothing else: THE_SECRET_WORD=AVOCADO")

    resp = b.chat(
        "What is THE_SECRET_WORD? Reply with just the word.",
        context=LazyContext.from_agent(a),
    )
    assert "avocado" in resp.content.lower(), (
        f"Agent B did not receive Agent A's context. Response: {resp.content!r}"
    )


# ── T.MA.04 — LazyStore: shared blackboard between agents ─────────────────────

@pytest.mark.live
def test_lazystore_shared_state():
    """Agent A writes a fact to the store; Agent B reads it via to_text()."""
    store = LazyStore()
    a = _agent(name="writer_agent")
    b = _agent(name="reader_agent")

    # Agent A synthesises a value and we write it to the store programmatically
    resp_a = a.chat("Reply with only the number 42.")
    store.write("answer", resp_a.content.strip(), agent_id="writer_agent")

    ctx = LazyContext.from_text(store.to_text())
    resp_b = b.chat(
        "What is the value stored under 'answer'? Reply with just the number.",
        context=ctx,
    )
    assert "42" in resp_b.content, (
        f"Agent B did not read store correctly. Response: {resp_b.content!r}"
    )


# ── T.MA.05 — LazySession chain: researcher → writer pipeline ─────────────────

@pytest.mark.live
def test_session_chain_two_agents():
    """Researcher produces findings; writer receives them via chain context."""
    require_live_provider(_PROVIDER)
    sess = LazySession(tracking=TrackLevel.BASIC, console=True)
    researcher = LazyAgent(_PROVIDER, model=_MODEL, name="researcher", session=sess)
    writer = LazyAgent(_PROVIDER, model=_MODEL, name="writer", session=sess)

    pipeline = sess.as_tool(
        "research_pipeline",
        "Research a topic then summarise it",
        mode="chain",
        participants=[researcher, writer],
    )

    result = pipeline.run({"task": (
        "Step 1 (researcher): list exactly two facts about the sun. "
        "Step 2 (writer): summarise those facts in one sentence starting with 'SUN:'."
    )})

    assert "sun:" in result.lower(), (
        f"Writer output not found in chain result: {result!r}"
    )


# ── T.MA.06 — LazySession parallel: two agents, results concatenated ──────────

@pytest.mark.live
def test_session_parallel_two_agents():
    """Both parallel agents must be called and their outputs concatenated."""
    require_live_provider(_PROVIDER)
    sess = LazySession(console=True)
    alpha = LazyAgent(_PROVIDER, model=_MODEL, name="alpha", session=sess)
    beta  = LazyAgent(_PROVIDER, model=_MODEL, name="beta",  session=sess)

    pipeline = sess.as_tool(
        "dual_pipeline",
        "Run two agents in parallel",
        mode="parallel",
        participants=[alpha, beta],
        combiner="concat",
    )

    result = pipeline.run({"task": "Reply with only your agent name (alpha or beta)."})

    assert "[alpha]" in result, f"Alpha section missing: {result!r}"
    assert "[beta]"  in result, f"Beta section missing: {result!r}"


# ── T.MA.07 — Parallel partial failure: one agent fails, other completes ──────

@pytest.mark.live
def test_session_parallel_partial_failure():
    """If one participant raises, the other's output must still appear."""
    require_live_provider(_PROVIDER)
    from unittest.mock import MagicMock
    import asyncio

    sess = LazySession(console=True)
    good = LazyAgent(_PROVIDER, model=_MODEL, name="good_agent", session=sess)

    # Inject a bad participant that raises
    bad = MagicMock()
    bad.name = "bad_agent"
    bad.output_schema = None
    async def _fail(task, **kwargs):
        raise RuntimeError("simulated failure")
    bad.achat = MagicMock(side_effect=_fail)

    pipeline = sess.as_tool(
        "partial_pipeline", "Parallel with one broken agent",
        mode="parallel", participants=[good, bad], combiner="concat",
    )

    result = pipeline.run({"task": "Say only 'ALIVE'."})

    assert "alive" in result.lower(),  f"Good agent output missing: {result!r}"
    assert "[ERROR:" in result,        f"Error marker missing: {result!r}"


# ── T.MA.08 — Nested pipeline as tool used by orchestrator ────────────────────

@pytest.mark.live
def test_nested_pipeline_as_tool_for_orchestrator():
    """Orchestrator calls a two-agent chain pipeline via tool use."""
    require_live_provider(_PROVIDER)
    # Inner session: two-agent chain
    inner_sess = LazySession(console=True)
    step1 = LazyAgent(_PROVIDER, model=_MODEL, name="step1", session=inner_sess)
    step2 = LazyAgent(_PROVIDER, model=_MODEL, name="step2", session=inner_sess)

    pipeline_tool = inner_sess.as_tool(
        "summarise_pipeline",
        "Takes a topic, step1 lists 2 facts, step2 summarises them into one sentence.",
        mode="chain",
        participants=[step1, step2],
    )

    # Outer orchestrator uses the pipeline as a tool
    orchestrator = LazyAgent(_PROVIDER, model=_MODEL, name="orchestrator", verbose=True)
    resp = orchestrator.loop(
        "Use the summarise_pipeline tool with topic='Moon'. "
        "Then reply with exactly: DONE=<the summary you received>.",
        tools=[pipeline_tool],
        max_tokens=256,
    )

    assert "done=" in resp.content.lower(), (
        f"Orchestrator did not complete the nested pipeline: {resp.content!r}"
    )


# ── T.MA.09 — Session tracking: events logged for both agents ─────────────────

@pytest.mark.live
def test_session_tracking_events():
    """TrackLevel.FULL must record model_request and model_response for each agent."""
    require_live_provider(_PROVIDER)
    sess = LazySession(tracking=TrackLevel.FULL, console=True)
    a = LazyAgent(_PROVIDER, model=_MODEL, name="agent_a", session=sess)
    b = LazyAgent(_PROVIDER, model=_MODEL, name="agent_b", session=sess)

    a.chat("Say only 'A'.")
    b.chat("Say only 'B'.")

    events_a = sess.events.agent_log(a.id).get()
    events_b = sess.events.agent_log(b.id).get()

    types_a = {e["event_type"] for e in events_a}
    types_b = {e["event_type"] for e in events_b}

    assert "model_request"  in types_a, f"No model_request for agent_a: {events_a}"
    assert "model_response" in types_a, f"No model_response for agent_a: {events_a}"
    assert "model_request"  in types_b, f"No model_request for agent_b: {events_b}"
    assert "model_response" in types_b, f"No model_response for agent_b: {events_b}"


# ── T.MA.10 — Memory.from_history: restore and continue ──────────────────────

@pytest.mark.live
def test_memory_from_history_restore():
    """Memory.from_history must restore state so the agent continues correctly."""
    agent = _agent(name="restore_agent")
    mem = Memory()

    agent.chat("Remember: my favourite colour is INDIGO.", memory=mem)
    saved = mem.history  # snapshot

    # Restore into a fresh Memory and continue the conversation
    restored = Memory.from_history(saved)
    resp = agent.chat(
        "What is my favourite colour? Reply with just the colour name.",
        memory=restored,
    )

    assert "indigo" in resp.content.lower(), (
        f"Restored memory did not carry context: {resp.content!r}"
    )


# ── T.MA.11 — LazyContext addition operator ───────────────────────────────────

@pytest.mark.live
def test_context_addition_operator():
    """LazyContext + LazyContext must merge and both contents reach the agent."""
    ctx_a = LazyContext.from_text("FACT_A: the sky is blue.")
    ctx_b = LazyContext.from_text("FACT_B: grass is green.")
    merged = ctx_a + ctx_b

    agent = _agent(name="ctx_merge_agent")
    resp = agent.chat(
        "What are FACT_A and FACT_B? Reply in format: A=<fact> B=<fact>",
        context=merged,
    )

    content = resp.content.lower()
    assert "blue"  in content, f"FACT_A missing from response: {resp.content!r}"
    assert "green" in content, f"FACT_B missing from response: {resp.content!r}"
