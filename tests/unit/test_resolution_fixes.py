"""Tests covering every fix in the resolution plan.

Sections:
  E9  — Plan: checkpoint before durable store write
  S1  — Memory.add auto-estimates tokens from word count
  S5  — OTelExporter uses BatchSpanProcessor by default
  X1  — OTelExporter does not clobber the global OTel provider
  S6  — Session warns on every exporter failure (see also test_audit_fixes.py)
  A5  — Agent: fallback= cycle detected at construction time
  C6  — tool_schema: TypedDict and NamedTuple annotations → object schema
  A2  — stream() writes to store on completion
"""

from __future__ import annotations

import asyncio
import typing
from typing import NamedTuple

import pytest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _FakeEngine:
    model = "fake"
    _agent_name = "fake"

    def _validate(self, tool_map):
        pass

    async def run(self, env, *, tools, output_type, memory, session):
        from lazybridge import Envelope

        return Envelope(task=env.task, payload=env.task or "")

    async def stream(self, env, *, tools, output_type, memory, session):
        yield env.task or ""


# ---------------------------------------------------------------------------
# E9 — Plan: checkpoint before durable store write
# ---------------------------------------------------------------------------


def test_plan_sequential_checkpoint_before_store_write():
    """In the sequential path the step-end checkpoint must be saved before
    the durable store write, so a crash between the two is safe on resume."""
    from lazybridge import Agent
    from lazybridge.engines.plan import Plan, Step
    from lazybridge.store import Store
    from lazybridge.testing import MockAgent

    ops: list[str] = []

    class _TrackingStore(Store):
        def write(self, key, value, *, agent_id=None):
            ops.append(f"store.write:{key}")
            super().write(key, value, agent_id=agent_id)

        def compare_and_swap(self, key, expected, new):
            if key and "__plan_checkpoint__" in str(key):
                ops.append("checkpoint")
            return super().compare_and_swap(key, expected, new)

    tracking_store = _TrackingStore()
    a = MockAgent("result", name="s1")
    plan = Plan(
        Step(target=a, name="s1", writes="out"),
        store=tracking_store,
        checkpoint_key="test_ckpt_seq",
    )
    Agent(engine=plan, name="p")("task")

    store_write_idx = next((i for i, o in enumerate(ops) if o == "store.write:out"), None)
    assert store_write_idx is not None, f"store.write:out never happened; ops={ops}"

    checkpoints_after_write = [i for i, o in enumerate(ops) if o == "checkpoint" and i > store_write_idx]
    assert not checkpoints_after_write, f"Found checkpoint(s) AFTER store.write — wrong order; ops={ops}"


def test_plan_parallel_checkpoint_before_store_write():
    """Same ordering guarantee for the parallel-band path."""
    from lazybridge import Agent
    from lazybridge.engines.plan import Plan, Step
    from lazybridge.store import Store
    from lazybridge.testing import MockAgent

    ops: list[str] = []

    class _TrackingStore(Store):
        def write(self, key, value, *, agent_id=None):
            ops.append(f"store.write:{key}")
            super().write(key, value, agent_id=agent_id)

        def compare_and_swap(self, key, expected, new):
            if key and "__plan_checkpoint__" in str(key):
                ops.append("checkpoint")
            return super().compare_and_swap(key, expected, new)

    tracking_store = _TrackingStore()
    a1 = MockAgent("r1", name="p1")
    a2 = MockAgent("r2", name="p2")
    plan = Plan(
        Step(target=a1, name="p1", writes="o1", parallel=True),
        Step(target=a2, name="p2", writes="o2", parallel=True),
        store=tracking_store,
        checkpoint_key="test_ckpt_par",
    )
    Agent(engine=plan, name="pp")("task")

    for write_key in ("o1", "o2"):
        store_write_idx = next((i for i, o in enumerate(ops) if o == f"store.write:{write_key}"), None)
        if store_write_idx is None:
            continue  # step may not have written if output was None
        checkpoints_after = [i for i, o in enumerate(ops) if o == "checkpoint" and i > store_write_idx]
        assert not checkpoints_after, f"Found checkpoint after store.write:{write_key}; ops={ops}"


# ---------------------------------------------------------------------------
# S1 — Memory.add auto-estimates tokens from word count
# ---------------------------------------------------------------------------


def test_memory_add_auto_estimates_tokens_triggers_compression():
    """Memory(strategy='auto', max_tokens=5) must trigger compression once
    enough turns accumulate — without any explicit tokens= argument."""
    from lazybridge.memory import Memory

    compressed = []

    def _fake_summarizer(text: str) -> str:
        compressed.append(text)
        return "[summary]"

    mem = Memory(strategy="auto", max_tokens=5, summarizer=_fake_summarizer)
    # Add 12 turns with 6-word content each → total ≈ 72 words >> max_tokens=5
    # Compression fires after > 10 turns once the budget is exceeded.
    for _ in range(12):
        mem.add("hello world foo bar baz", "a")
    assert compressed, "Auto compression should trigger once turns > 10 and tokens > max"


def test_memory_add_explicit_tokens_stored_as_is():
    """When tokens= is explicitly provided, the word-count estimation must be
    skipped and the caller's value used verbatim."""
    from lazybridge.memory import Memory

    mem = Memory()
    mem.add("one two three four five", "", tokens=42)
    # 5 words → word-count estimate would be 6 (incl. empty assistant " ")
    # but explicit tokens=42 must be stored
    assert mem._turns[-1].token_estimate == 42, "Explicit tokens=42 must not be overridden by word-count estimate"


def test_memory_add_no_estimation_for_empty_content():
    """Empty strings must not inflate the token estimate."""
    from lazybridge.memory import Memory

    compressed = []

    def _fake_summarizer(text: str) -> str:
        compressed.append(text)
        return "[summary]"

    mem = Memory(strategy="auto", max_tokens=1, summarizer=_fake_summarizer)
    for _ in range(12):
        mem.add("", "")
    assert not compressed, "Empty turns must not trigger compression via word-count estimation"


# ---------------------------------------------------------------------------
# S5 — OTelExporter uses BatchSpanProcessor by default
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("opentelemetry"),
    reason="opentelemetry-sdk not installed",
)
def test_otel_exporter_default_uses_batch_span_processor():
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from lazybridge.ext.otel.exporter import OTelExporter

    inner = InMemorySpanExporter()
    exp = OTelExporter(exporter=inner)

    processors = exp._provider._active_span_processor._span_processors
    assert any(isinstance(p, BatchSpanProcessor) for p in processors), f"Expected BatchSpanProcessor; got {processors}"


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("opentelemetry"),
    reason="opentelemetry-sdk not installed",
)
def test_otel_exporter_batch_false_uses_simple_span_processor():
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from lazybridge.ext.otel.exporter import OTelExporter

    inner = InMemorySpanExporter()
    exp = OTelExporter(exporter=inner, batch=False)

    processors = exp._provider._active_span_processor._span_processors
    assert any(isinstance(p, SimpleSpanProcessor) for p in processors), (
        f"Expected SimpleSpanProcessor; got {processors}"
    )


# ---------------------------------------------------------------------------
# X1 — OTelExporter does not clobber the global OTel provider
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("opentelemetry"),
    reason="opentelemetry-sdk not installed",
)
def test_otel_exporter_does_not_set_global_provider():
    """Creating an OTelExporter must not install itself as the global
    OpenTelemetry trace provider."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from lazybridge.ext.otel.exporter import OTelExporter

    before = trace.get_tracer_provider()
    inner = InMemorySpanExporter()
    exp = OTelExporter(exporter=inner)
    after = trace.get_tracer_provider()

    assert before is after, "OTelExporter must not clobber the global OTel provider."
    assert exp._provider is not after, "instance-scoped provider must differ from global"


# ---------------------------------------------------------------------------
# A5 — Agent: fallback= cycle detected at construction time
# ---------------------------------------------------------------------------


def test_fallback_cycle_detected_when_chain_already_loops():
    """If `a.fallback = b` is injected after construction (forming a loop),
    the next agent constructed with `fallback=b` must detect the cycle."""
    from lazybridge import Agent

    a = Agent(name="a", engine=_FakeEngine())
    b = Agent(name="b", engine=_FakeEngine(), fallback=a)
    # Inject a cycle: a → b → a → ...
    a.fallback = b
    # Constructing c with fallback=b now walks: c→b→a→b (cycle)
    with pytest.raises(ValueError, match="cycle"):
        Agent(name="c", engine=_FakeEngine(), fallback=b)


def test_fallback_linear_chain_is_fine():
    """a → b → c (no cycle) must not raise."""
    from lazybridge import Agent

    c = Agent(name="c", engine=_FakeEngine())
    b = Agent(name="b", engine=_FakeEngine(), fallback=c)
    Agent(name="a", engine=_FakeEngine(), fallback=b)


def test_fallback_none_is_fine():
    """No fallback at all must not raise."""
    from lazybridge import Agent

    Agent(name="solo", engine=_FakeEngine())


# ---------------------------------------------------------------------------
# C6 — tool_schema: TypedDict and NamedTuple annotations → object schema
# ---------------------------------------------------------------------------


class _MovieQuery(typing.TypedDict):
    title: str
    year: int


class _Point(NamedTuple):
    x: float
    y: float


def test_typeddict_annotation_produces_object_schema():
    from lazybridge.core.tool_schema import _annotation_to_schema

    schema = _annotation_to_schema(_MovieQuery)
    assert schema["type"] == "object"
    assert "title" in schema["properties"]
    assert "year" in schema["properties"]
    assert schema["properties"]["title"]["type"] == "string"
    assert schema["properties"]["year"]["type"] == "integer"
    assert "additionalProperties" in schema


def test_namedtuple_annotation_produces_object_schema():
    from lazybridge.core.tool_schema import _annotation_to_schema

    schema = _annotation_to_schema(_Point)
    assert schema["type"] == "object"
    assert "x" in schema["properties"]
    assert "y" in schema["properties"]
    assert schema["properties"]["x"]["type"] == "number"
    assert schema["properties"]["y"]["type"] == "number"
    assert set(schema["required"]) == {"x", "y"}


def test_typeddict_used_as_function_param_schema():
    """TypedDict parameter inside a tool function produces object schema, not string."""
    from lazybridge import tool

    def search(query: _MovieQuery) -> str:
        """Search for a movie."""
        return ""

    t = tool(search, name="search")
    defn = t.definition()
    query_schema = defn.parameters.get("properties", {}).get("query", {})
    assert query_schema.get("type") == "object", f"Expected object schema for TypedDict param; got {query_schema}"


def test_namedtuple_used_as_function_param_schema():
    """NamedTuple parameter inside a tool function produces object schema, not string."""
    from lazybridge import tool

    def plot(point: _Point) -> str:
        """Plot a point."""
        return ""

    t = tool(plot, name="plot")
    defn = t.definition()
    point_schema = defn.parameters.get("properties", {}).get("point", {})
    assert point_schema.get("type") == "object", f"Expected object schema for NamedTuple param; got {point_schema}"


# ---------------------------------------------------------------------------
# A2 — stream() writes to store on completion
# ---------------------------------------------------------------------------


def test_stream_writes_to_store_on_completion():
    """After consuming all chunks from agent.stream(), the store must
    contain the concatenated output under the agent's key."""
    from lazybridge import Agent
    from lazybridge.sentinels import _AGENT_OUTPUT_KEY_PREFIX
    from lazybridge.store import Store

    store = Store()
    agent = Agent(name="writer", engine=_FakeEngine(), store=store)

    async def _run():
        chunks = []
        async for chunk in agent.stream("hello world"):
            chunks.append(chunk)
        return "".join(chunks)

    result = asyncio.run(_run())
    assert result == "hello world"

    stored = store.read(_AGENT_OUTPUT_KEY_PREFIX + "writer")
    assert stored == "hello world", f"store should hold streamed output; got {stored!r}"


def test_stream_does_not_write_to_store_on_early_break():
    """If the consumer breaks before reading all chunks the store must NOT
    be written (partial output is worse than no output)."""
    from lazybridge import Agent
    from lazybridge.sentinels import _AGENT_OUTPUT_KEY_PREFIX
    from lazybridge.store import Store

    store = Store()
    agent = Agent(name="partial", engine=_FakeEngine(), store=store)

    async def _run():
        async for _chunk in agent.stream("alpha beta gamma"):
            break  # early exit

    asyncio.run(_run())
    stored = store.read(_AGENT_OUTPUT_KEY_PREFIX + "partial")
    assert stored is None, f"store must not be written on early break; got {stored!r}"
