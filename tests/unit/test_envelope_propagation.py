"""Envelope propagation through tool calls.

Before this change, Agent-as-tool flattened the inner Agent's full
Envelope to ``str`` at the ``as_tool()`` boundary — so a nested Agent's
tokens / cost / error channel were lost to the outer caller.  The fix
makes ``as_tool()`` return the inner Envelope verbatim, and teaches
engines (LLMEngine, Plan) to aggregate nested metadata.

What's verified here:

* ``Envelope.__str__`` falls through to ``.text()`` — prevents
  accidental ``"Envelope(task=…)"`` garbage leaking into LLM tool
  results or REPL output.
* ``Tool.returns_envelope`` flag is set automatically by
  ``agent.as_tool()`` and by ``wrap_tool(agent)``.
* An Envelope-returning tool's metadata accumulates in the outer
  Envelope's ``nested_input_tokens`` / ``nested_output_tokens`` /
  ``nested_cost_usd`` buckets in LLMEngine (simulated model loop).
* Plan step that invokes an agent-as-tool preserves the inner
  Envelope's metadata and error state across the step boundary.
* Transitive aggregation: two levels of nesting add up correctly.
"""

from __future__ import annotations

import pytest

from lazybridge import (
    Agent,
    Envelope,
    Plan,
    Step,
    Tool,
)
from lazybridge.envelope import EnvelopeMetadata

# ---------------------------------------------------------------------------
# Envelope.__str__ safety net
# ---------------------------------------------------------------------------


def test_envelope_str_falls_through_to_text():
    """``str(envelope)`` must yield the same string as ``envelope.text()``.

    This is the safety net that lets legacy callers which did
    ``str(tool.run_sync(...))`` keep working after ``Agent.as_tool``
    started returning an Envelope instead of a flat string.
    """
    env = Envelope(task="hi", payload="world")
    assert str(env) == env.text() == "world"


def test_envelope_str_on_pydantic_payload():
    """Pydantic payloads serialise as JSON through __str__ too."""
    from pydantic import BaseModel

    class M(BaseModel):
        x: int

    env = Envelope(payload=M(x=42))
    assert str(env) == '{"x":42}'


# ---------------------------------------------------------------------------
# Tool.returns_envelope flag is opt-in and set by as_tool
# ---------------------------------------------------------------------------


def test_plain_tool_defaults_returns_envelope_false():
    """Function-wrapping tools do NOT return Envelope — the contract
    for tool authors stays "normal Python function".
    """
    def echo(x: str) -> str:
        """Return x."""
        return x

    tool = Tool(echo)
    assert tool.returns_envelope is False


def test_agent_as_tool_sets_returns_envelope_true():
    """Agent.as_tool opts into the Envelope-preservation path."""
    a = Agent("claude-opus-4-7", name="fake")
    assert a.as_tool().returns_envelope is True


def test_wrap_tool_on_agent_sets_returns_envelope_true():
    """Passing an Agent into tools=[...] goes through wrap_tool, which
    must set the flag automatically so engines preserve the Envelope.
    """
    from lazybridge.tools import wrap_tool

    a = Agent("claude-opus-4-7", name="nested")
    t = wrap_tool(a)
    assert t.returns_envelope is True


# ---------------------------------------------------------------------------
# LLMEngine — nested metadata aggregation (simulated with a fake engine)
# ---------------------------------------------------------------------------


class _FakeInnerEngine:
    """Fake engine that reports fixed token usage + cost on every run.

    Lets us exercise the nested-aggregation path without real provider
    calls.
    """

    def __init__(self, in_tok: int, out_tok: int, cost: float, payload: str) -> None:
        self._in, self._out, self._cost, self._payload = in_tok, out_tok, cost, payload

    async def run(self, env, *, tools, output_type, memory, session):
        return Envelope(
            task=env.task,
            payload=self._payload,
            metadata=EnvelopeMetadata(
                input_tokens=self._in,
                output_tokens=self._out,
                cost_usd=self._cost,
            ),
        )

    async def stream(self, *a, **kw):  # pragma: no cover
        if False:
            yield ""


class _RelayOuterEngine:
    """Minimal outer engine that calls its single tool and surfaces nested
    usage into the outer Envelope's ``nested_*`` buckets.

    Mirrors the accounting the real LLMEngine does in its tool-dispatch
    branch, so the test stays independent of live provider APIs.
    """

    async def run(self, env, *, tools, output_type, memory, session):
        assert len(tools) == 1
        raw = await tools[0].run(task=env.task or "")
        nested_in = nested_out = 0
        nested_cost = 0.0
        if isinstance(raw, Envelope):
            m = raw.metadata
            nested_in = m.input_tokens + m.nested_input_tokens
            nested_out = m.output_tokens + m.nested_output_tokens
            nested_cost = m.cost_usd + m.nested_cost_usd
            payload = raw.text()
            error = raw.error
        else:
            payload = str(raw)
            error = None
        return Envelope(
            task=env.task,
            payload=payload,
            metadata=EnvelopeMetadata(
                input_tokens=100, output_tokens=50, cost_usd=0.002,
                nested_input_tokens=nested_in,
                nested_output_tokens=nested_out,
                nested_cost_usd=nested_cost,
            ),
            error=error,
        )

    async def stream(self, *a, **kw):  # pragma: no cover
        if False:
            yield ""


def test_nested_envelope_propagates_single_level():
    inner = Agent(engine=_FakeInnerEngine(in_tok=30, out_tok=20, cost=0.001,
                                          payload="inner-result"),
                  name="inner")
    outer = Agent(engine=_RelayOuterEngine(), tools=[inner], name="outer")

    env = outer("do X")
    assert env.text() == "inner-result"

    # Outer's own tokens unchanged.
    assert env.metadata.input_tokens == 100
    assert env.metadata.output_tokens == 50
    assert env.metadata.cost_usd == pytest.approx(0.002)

    # Inner's tokens live in nested_*.
    assert env.metadata.nested_input_tokens == 30
    assert env.metadata.nested_output_tokens == 20
    assert env.metadata.nested_cost_usd == pytest.approx(0.001)


def test_nested_envelope_propagates_two_levels_transitively():
    """Deeply-nested agents: A → B → C.

    C's metadata must surface all the way up to A's nested_* buckets.
    The ``_RelayOuterEngine`` aggregates
    ``nested = inner.input + inner.nested_input`` so the chain is
    transitive.
    """
    leaf = Agent(engine=_FakeInnerEngine(in_tok=10, out_tok=5, cost=0.0003,
                                         payload="leaf"),
                 name="leaf")
    middle = Agent(engine=_RelayOuterEngine(), tools=[leaf], name="middle")
    root = Agent(engine=_RelayOuterEngine(), tools=[middle], name="root")

    env = root("top task")
    assert env.text() == "leaf"

    # middle's own tokens (100/50/0.002) + leaf's (10/5/0.0003) accumulate
    # into root's nested_* buckets.
    assert env.metadata.nested_input_tokens == 100 + 10
    assert env.metadata.nested_output_tokens == 50 + 5
    assert env.metadata.nested_cost_usd == pytest.approx(0.002 + 0.0003)


def test_nested_error_surfaces_through_outer_envelope():
    """When the inner agent errored (``Envelope.error`` populated), the
    outer Envelope must carry that error forward rather than silently
    flattening to text.
    """
    class _ErroringEngine:
        async def run(self, env, *, tools, output_type, memory, session):
            return Envelope.error_envelope(RuntimeError("inner boom"))

        async def stream(self, *a, **kw):  # pragma: no cover
            if False:
                yield ""

    inner = Agent(engine=_ErroringEngine(), name="inner")
    outer = Agent(engine=_RelayOuterEngine(), tools=[inner], name="outer")

    env = outer("go")
    assert not env.ok
    assert env.error.type == "RuntimeError"
    assert "inner boom" in env.error.message


# ---------------------------------------------------------------------------
# Plan — step boundary preserves the inner Envelope
# ---------------------------------------------------------------------------


def test_plan_preserves_inner_envelope_metadata_at_step_boundary():
    """A Plan step whose target is an agent-as-tool must not flatten
    the inner Envelope's metadata when producing its step result.
    """
    inner = Agent(engine=_FakeInnerEngine(in_tok=42, out_tok=11, cost=0.0005,
                                          payload="plan-step-out"),
                  name="worker")

    inner_tool = inner.as_tool("worker_tool")

    plan = Plan(Step(target="worker_tool", name="run_worker"))
    runner = Agent.from_engine(plan, tools=[inner_tool])

    env = runner("start")
    # The Plan's final Envelope reflects the inner agent's metadata
    # directly (plain Plan, no outer LLM loop).
    assert env.text() == "plan-step-out"
    assert env.metadata.input_tokens == 42
    assert env.metadata.output_tokens == 11
    assert env.metadata.cost_usd == pytest.approx(0.0005)
