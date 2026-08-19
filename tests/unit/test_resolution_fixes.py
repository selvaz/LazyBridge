"""Tests covering every fix in the resolution plan.

Sections:
  E9  — Plan: checkpoint before durable store write
  S1  — Memory.add auto-estimates tokens from word count
  S5  — OTelExporter uses BatchSpanProcessor by default
  X1  — OTelExporter does not clobber the global OTel provider
  S6  — Session warns on every exporter failure (see also test_audit_fixes.py)
  A5  — Agent: fallback= cycle detected at construction time
  C6  — tool_schema: TypedDict and NamedTuple annotations → object schema
  C7  — tool_schema: dict[str, Any] is an open object; strict=True rejects it
  A2  — stream() writes to store on completion
"""

from __future__ import annotations

import asyncio
import typing
from typing import NamedTuple

import pytest
from pydantic import BaseModel as _BaseModel
from pydantic import ConfigDict as _ConfigDict

#: See the same constant in ``test_audit_followup.py``: ``opentelemetry.sdk``
#: is a separate distribution from the ``opentelemetry`` API namespace, and
#: another extra can supply the API alone. Short-circuits because
#: ``find_spec("opentelemetry.sdk")`` raises when the parent is absent.
_NO_OTEL_SDK = __import__("importlib").util.find_spec("opentelemetry") is None or (
    __import__("importlib").util.find_spec("opentelemetry.sdk") is None
)

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
    _NO_OTEL_SDK,
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
    _NO_OTEL_SDK,
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
    _NO_OTEL_SDK,
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


# Module-level (not function-local): this file has `from __future__ import
# annotations`, so a function-local class referenced by a nested function's
# annotation can't be resolved by typing.get_type_hints (it's not in
# func.__globals__) -- that failure is silently swallowed by
# _build_signature_mode, which falls back to a bare string schema instead of
# raising, defeating these tests' purpose entirely.
class _OpenInner(_BaseModel):
    x: int


class _ClosedOuterWithOpenInner(_BaseModel):
    model_config = _ConfigDict(extra="forbid")
    inner: _OpenInner
    y: str


class _ClosedInner(_BaseModel):
    model_config = _ConfigDict(extra="forbid")
    x: int


class _ClosedOuterWithClosedInner(_BaseModel):
    model_config = _ConfigDict(extra="forbid")
    inner: _ClosedInner
    y: str


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


# ---------------------------------------------------------------------------
# C7 — tool_schema: dict[str, Any] is an open object; strict=True must reject it
# ---------------------------------------------------------------------------


def test_dict_str_any_produces_open_object_schema_non_strict():
    """Non-strict: dict[str, Any] stays a bare, permissive object schema."""
    from lazybridge.core.tool_schema import _annotation_to_schema

    schema = _annotation_to_schema(dict[str, typing.Any])
    assert schema == {"type": "object"}


def test_dict_str_any_param_non_strict_builds_fine():
    from lazybridge import tool

    def configure(payload: dict[str, typing.Any]) -> str:
        """Accept an arbitrary config blob."""
        return ""

    t = tool(configure, name="configure")
    defn = t.definition()
    payload_schema = defn.parameters["properties"]["payload"]
    assert payload_schema == {"type": "object"}


def test_dict_str_any_param_strict_raises():
    """strict=True can't express an arbitrary-keyed dict as a closed schema --
    must fail loudly at build time, not silently forward an invalid schema
    that OpenAI's strict validator would reject at call time."""
    from lazybridge import tool
    from lazybridge.core.tool_schema import ToolSchemaBuildError

    def configure(payload: dict[str, typing.Any]) -> str:
        """Accept an arbitrary config blob."""
        return ""

    t = tool(configure, name="configure", strict=True)
    with pytest.raises(ToolSchemaBuildError, match="payload"):
        t.definition()


def test_bare_dict_param_strict_raises():
    """Same as dict[str, Any]: an unsubscripted dict is equally open."""
    from lazybridge import tool
    from lazybridge.core.tool_schema import ToolSchemaBuildError

    def configure(payload: dict) -> str:
        """Accept an arbitrary config blob."""
        return ""

    t = tool(configure, name="configure", strict=True)
    with pytest.raises(ToolSchemaBuildError, match="payload"):
        t.definition()


def test_dict_str_int_param_non_strict_builds_fine():
    from lazybridge import tool

    def configure(counts: dict[str, int]) -> str:
        """Accept named counts."""
        return ""

    t = tool(configure, name="configure")
    defn = t.definition()
    counts_schema = defn.parameters["properties"]["counts"]
    assert counts_schema == {"type": "object", "additionalProperties": {"type": "integer"}}


def test_bare_dict_produces_object_schema_not_string():
    """Regression: a bare (unsubscripted) dict annotation has no __origin__
    and was silently falling through to the string fallback."""
    from lazybridge.core.tool_schema import _annotation_to_schema

    assert _annotation_to_schema(dict) == {"type": "object"}


def test_dict_str_int_param_strict_raises():
    """A value-typed dict (dict[str, int]) is JUST AS open as dict[str, Any]
    under OpenAI strict mode: strict mode requires additionalProperties to be
    the literal `false`, not a value-type schema -- the *set of keys* is what
    must be closed, and a dict's keys are never enumerable. Only a
    TypedDict/pydantic model (fixed, named keys) can satisfy strict mode."""
    from lazybridge import tool
    from lazybridge.core.tool_schema import ToolSchemaBuildError

    def configure(counts: dict[str, int]) -> str:
        """Accept named counts."""
        return ""

    t = tool(configure, name="configure", strict=True)
    with pytest.raises(ToolSchemaBuildError, match="counts"):
        t.definition()


def test_typeddict_param_strict_builds_fine():
    """A TypedDict is already closed (additionalProperties: False) and must
    not be flagged as an open object under strict=True."""
    from lazybridge import tool

    def search(query: _MovieQuery) -> str:
        """Search for a movie."""
        return ""

    t = tool(search, name="search", strict=True)
    defn = t.definition()  # must not raise
    assert defn.parameters["properties"]["query"]["additionalProperties"] is False


def test_open_object_in_pydantic_defs_is_caught_under_strict():
    """A pydantic BaseModel param can itself be closed (extra="forbid") while
    a NESTED model it references is not -- pydantic emits that nested model
    under $defs with only a $ref left inline in properties, so a scan that
    only walks properties/items/anyOf never sees it. Must still be caught."""
    from lazybridge import tool
    from lazybridge.core.tool_schema import ToolSchemaBuildError

    def configure(payload: _ClosedOuterWithOpenInner) -> str:
        """Accept a nested config."""
        return ""

    t = tool(configure, name="configure", strict=True)
    with pytest.raises(ToolSchemaBuildError, match="payload"):
        t.definition()


def test_closed_nested_pydantic_model_strict_builds_fine():
    """The same shape, but the nested model is ALSO closed -- must not raise."""
    from lazybridge import tool

    def configure(payload: _ClosedOuterWithClosedInner) -> str:
        """Accept a nested config."""
        return ""

    t = tool(configure, name="configure", strict=True)
    t.definition()  # must not raise


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
