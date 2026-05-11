"""Progressive live tests — each rung adds one layer of the stack.

Run with::

    pytest -m live -s tests/live/

Open browser during Rung 7::

    pytest -m live -s tests/live/test_pipeline_live.py::test_rung7_viz --viz

Override model (e.g. DeepSeek)::

    LB_LIVE_MODEL=deepseek/deepseek-chat pytest -m live -s tests/live/

All tests are excluded from the default suite (``addopts = "-m 'not live ...'"``
in pyproject.toml) so they never run in CI unless explicitly opted in.
"""

from __future__ import annotations

import time

import pytest
from pydantic import BaseModel

from lazybridge import (
    Agent,
    LLMEngine,
    Plan,
    Session,
    Step,
    Tool,
    from_parallel_all,
    from_prev,
)
from lazybridge.ext.viz import Visualizer

# ---------------------------------------------------------------------------
# Rung 1 — single agent, bare call
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_rung1_single_agent(model: str, sess: Session) -> None:
    """Provider auth, executor path, Envelope metadata roll-up."""
    agent = Agent(
        engine=LLMEngine(model, system="You are a terse assistant. Follow instructions exactly."),
        session=sess,
        name="rung1",
    )
    env = agent("Reply with exactly the word: PONG")

    assert "PONG" in env.text().upper()
    assert env.metadata.input_tokens > 0
    assert env.metadata.output_tokens > 0
    assert env.metadata.cost_usd >= 0
    assert env.metadata.latency_ms > 0


# ---------------------------------------------------------------------------
# Rung 2 — agent with a deterministic tool
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_rung2_tool_call(model: str, sess: Session) -> None:
    """Tool schema generation, dispatch, result injection."""

    def multiply(a: int, b: int) -> int:
        """Multiply two integers."""
        return a * b

    agent = Agent(
        engine=LLMEngine(model, system="Use tools when asked. Reply concisely."),
        tools=[Tool.wrap(multiply, name="multiply")],
        session=sess,
        name="rung2",
    )
    env = agent("What is 6 multiplied by 7? Use the multiply tool.")

    assert "42" in env.text()

    events = sess.events.query()
    event_types = [e["event_type"] for e in events]
    assert "tool_call" in event_types
    assert "tool_result" in event_types


# ---------------------------------------------------------------------------
# Rung 3 — structured output
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_rung3_structured_output(model: str, sess: Session) -> None:
    """Structured output parsing, Pydantic coercion."""

    class Coords(BaseModel):
        x: int
        y: int

    agent = Agent(
        engine=LLMEngine(model, system="Return only the requested JSON fields."),
        output=Coords,
        session=sess,
        name="rung3",
    )
    env = agent("Return x=4 y=9.")

    assert isinstance(env.payload, Coords), f"Expected Coords, got {type(env.payload)}"
    assert env.payload.x == 4
    assert env.payload.y == 9


# ---------------------------------------------------------------------------
# Rung 4 — sequential Plan with Session event tracking
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_rung4_sequential_plan(model_capable: str, sess: Session) -> None:
    """Plan compilation, step execution, from_prev sentinel, event emission."""
    fetch = Agent(
        engine=LLMEngine(model_capable, system="Reply only with: DATA:42"),
        name="fetch",
        session=sess,
    )
    analyse = Agent(
        engine=LLMEngine(
            model_capable,
            system="Extract the integer after 'DATA:' and reply with only that integer.",
        ),
        name="analyse",
        session=sess,
    )

    pipeline = Agent(
        engine=Plan(
            Step("fetch"),
            Step("analyse", task=from_prev),
        ),
        tools=[fetch, analyse],
        session=sess,
        name="pipeline",
    )
    env = pipeline("run")

    assert "42" in env.text(), f"Expected '42' in output, got: {env.text()!r}"

    events = sess.events.query()
    event_types = [e["event_type"] for e in events]
    assert "agent_start" in event_types
    assert "agent_finish" in event_types
    assert len(events) >= 4


# ---------------------------------------------------------------------------
# Rung 5 — parallel plan steps
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_rung5_parallel_steps(model_capable: str, sess: Session) -> None:
    """Parallel step dispatch, from_parallel_all aggregation, fan-in."""
    ra = Agent(
        engine=LLMEngine(model_capable, system="Reply only with: RESULT_A"),
        name="ra",
        session=sess,
    )
    rb = Agent(
        engine=LLMEngine(model_capable, system="Reply only with: RESULT_B"),
        name="rb",
        session=sess,
    )
    merger = Agent(
        engine=LLMEngine(
            model_capable,
            system="Combine all inputs into one comma-separated line.",
        ),
        name="merger",
        session=sess,
    )

    pipeline = Agent(
        engine=Plan(
            Step("ra", parallel=True),
            Step("rb", parallel=True),
            Step("merger", task=from_parallel_all("ra")),
        ),
        tools=[ra, rb, merger],
        session=sess,
        name="par_pipeline",
    )
    env = pipeline("run")

    assert env.text(), "merger returned empty output"
    assert env.metadata.cost_usd >= 0


# ---------------------------------------------------------------------------
# Rung 6 — agent as tool (nested agent, metadata roll-up)
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_rung6_agent_as_tool(model: str, sess: Session) -> None:
    """Agent-as-tool wrapping, nested envelope cost/token roll-up."""
    translator = Agent(
        engine=LLMEngine(
            model,
            system="Translate the given text to French. Reply with only the translation.",
        ),
        name="translator",
        session=sess,
    )
    outer = Agent(
        engine=LLMEngine(model, system="Use the translator tool when asked to translate."),
        tools=[translator],
        session=sess,
        name="outer",
    )
    env = outer("Translate 'hello' to French using the translator tool.")

    assert any(w in env.text().lower() for w in ["bonjour", "salut", "allô"]), (
        f"Expected French greeting, got: {env.text()!r}"
    )
    assert env.metadata.input_tokens > 0


# ---------------------------------------------------------------------------
# Rung 7 — full pipeline + Session tracking + Visualizer
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_rung7_viz(model_capable: str, sess: Session, viz_open: bool, tmp_path) -> None:
    """Visualizer lifecycle, SSE server, graph population, event log, replay."""

    def web_search(query: str) -> str:
        """Search the web and return a short summary."""
        time.sleep(0.2)
        return f"[stub] top result for '{query}': Python is a popular language."

    researcher = Agent(
        engine=LLMEngine(model_capable, system="Find one key fact. Use the web_search tool."),
        tools=[Tool.wrap(web_search, name="web_search")],
        name="researcher",
        session=sess,
    )
    writer = Agent(
        engine=LLMEngine(model_capable, system="Write one sentence summarising the findings."),
        name="writer",
        session=sess,
    )
    chain = Agent.chain(researcher, writer)

    with Visualizer(sess, auto_open=viz_open) as viz:
        print(f"\n[viz] {viz.url}")
        env = chain("What is Python used for?")
        print(f"[viz] result: {env.text()[:120]}")

    assert env.text(), "writer returned empty output"

    events = sess.events.query()
    assert len(events) > 4, (
        f"Expected >4 events, got {len(events)}: {[e['event_type'] for e in events]}"
    )
    event_types = {e["event_type"] for e in events}
    assert "agent_start" in event_types
    assert "agent_finish" in event_types

    # Verify replay constructor accepts the recorded DB
    db_path = str(tmp_path / "live.db")
    viz_replay = Visualizer.replay(db_path, auto_open=False)
    assert viz_replay is not None
