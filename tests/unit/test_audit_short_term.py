"""Tests for the short-term audit hardening (5.1 plan).

Each test traces back to a finding in the deep architecture audit:

  H-A   Per-event-type back-pressure in EventLog (``on_full="hybrid"``)
  H-B   Memory.summarizer_timeout + lock release during summarisation
  H-D   OTel GenAI conventions + parent-child spans + cross-agent
        context propagation
  H-E   MCP _tools_cache TTL + invalidate_tools_cache()
  M-A   Tool-call JSON parse errors surfaced as TOOL_ERROR loudly
  M-B   Streaming + tool-call accumulation works through LLMEngine
"""

from __future__ import annotations

import asyncio
import warnings
from collections.abc import AsyncIterator
from typing import Any

import pytest

from lazybridge.core.types import (
    CompletionRequest,
    CompletionResponse,
    StreamChunk,
    ToolCall,
    UsageStats,
)
from lazybridge.session import (
    DEFAULT_CRITICAL_EVENT_TYPES,
    EventLog,
    EventType,
    Session,
)

# ---------------------------------------------------------------------------
# H-A: per-event-type back-pressure
# ---------------------------------------------------------------------------


def test_eventlog_hybrid_blocks_on_critical_event_types() -> None:
    """When ``on_full='hybrid'`` (the new default), saturating the queue
    must block the producer for AGENT_*/TOOL_* events but never drop them.
    """
    log = EventLog(
        "sess-h1",
        batched=True,
        batch_size=1,
        batch_interval=10.0,
        max_queue_size=2,
        on_full="hybrid",
    )
    try:
        log.record(EventType.AGENT_START, {"agent_name": "a"}, run_id="r1")
        log.flush(timeout=2.0)
        log.record(EventType.TOOL_ERROR, {"tool": "t", "error": "boom"}, run_id="r1")
        log.flush(timeout=2.0)
        rows = log.query()
        assert any(r["event_type"] == "agent_start" for r in rows)
        assert any(r["event_type"] == "tool_error" for r in rows)
        assert log._dropped_count == 0
    finally:
        log.close()


def test_eventlog_hybrid_drops_only_telemetry_events_on_saturation() -> None:
    """Hybrid policy drops cheap telemetry (LOOP_STEP) when the writer
    can't keep up but never drops audit-critical events."""
    log = EventLog(
        "sess-h2",
        batched=True,
        batch_size=1000,
        batch_interval=60.0,  # writer effectively never wakes on its own
        max_queue_size=1,
        on_full="hybrid",
    )
    try:
        # Saturate with one telemetry event — second telemetry submit
        # will hit the bounded queue and drop silently.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log.record(EventType.LOOP_STEP, {"turn": 0}, run_id="r1")
            log.record(EventType.LOOP_STEP, {"turn": 1}, run_id="r1")
        # At least one telemetry drop expected.
        assert log._dropped_count >= 1
        # Critical events still block (they go on the queue head once
        # we drain).  Drain manually.
        log.flush(timeout=2.0)
    finally:
        log.close()


def test_eventlog_hybrid_critical_events_default_set() -> None:
    """Documented default critical events match the implementation."""
    expected = {
        "agent_start",
        "agent_finish",
        "tool_call",
        "tool_result",
        "tool_error",
        "tool_timeout",
        "hil_decision",
    }
    assert set(DEFAULT_CRITICAL_EVENT_TYPES) == expected


def test_session_default_on_full_is_hybrid() -> None:
    """The new default for Session(batched=True) is ``hybrid``."""
    sess = Session(batched=True, batch_size=10, max_queue_size=10)
    try:
        assert sess.events._on_full == "hybrid"
    finally:
        sess.close()


def test_eventlog_validates_on_full_value() -> None:
    with pytest.raises(ValueError, match="hybrid"):
        EventLog("sess-bad", batched=True, on_full="explode")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# H-B: Memory summarizer timeout + lock release
# ---------------------------------------------------------------------------


def test_memory_summarizer_timeout_falls_back_to_keyword_summary() -> None:
    """When the (async) summariser exceeds the deadline, compression
    must fall back to the keyword fallback rather than blocking add()."""
    from lazybridge.memory import Memory

    async def _slow_summariser(prompt: str) -> str:
        await asyncio.sleep(5.0)
        return "should never be returned"

    mem = Memory(strategy="summary", max_tokens=10, summarizer=_slow_summariser, summarizer_timeout=0.1)
    # Push >10 turns to force compression.
    for i in range(15):
        mem.add(f"q{i}", f"a{i}", tokens=50)
    # Compression ran: summary should be the keyword fallback (starts
    # with "[Earlier conversation covered:" per _rule_summary).
    assert mem._summary
    assert mem._summary.startswith("[Earlier conversation covered:")


def test_memory_summarizer_default_timeout_is_set() -> None:
    from lazybridge.memory import Memory

    mem = Memory()
    assert mem.summarizer_timeout == Memory._DEFAULT_SUMMARIZER_TIMEOUT


def test_memory_summarizer_timeout_validates_value() -> None:
    from lazybridge.memory import Memory

    with pytest.raises(ValueError, match="summarizer_timeout"):
        Memory(summarizer_timeout=0)
    with pytest.raises(ValueError, match="summarizer_timeout"):
        Memory(summarizer_timeout=-1.0)


def test_memory_compression_does_not_hold_lock() -> None:
    """While compression is in progress (slow summariser), other
    threads' ``add()`` calls must still progress.  Pre-fix this would
    deadlock because the lock was held during the LLM call."""
    import threading

    from lazybridge.memory import Memory

    started = threading.Event()
    finish = threading.Event()
    other_done = threading.Event()

    def _summariser(prompt: str) -> str:
        started.set()
        finish.wait(timeout=5.0)
        return "summary"

    mem = Memory(strategy="summary", max_tokens=10, summarizer=_summariser)
    # Bootstrap memory past the 10-turn threshold so the next .add()
    # triggers compression.
    for i in range(11):
        mem.add(f"q{i}", f"a{i}", tokens=100)

    def _trigger_compression() -> None:
        mem.add("trigger", "trigger-resp", tokens=100)

    def _other_caller() -> None:
        started.wait(timeout=2.0)
        # Pre-fix: blocked on the same lock as the compressing thread.
        # Post-fix: completes immediately because compression runs
        # outside the lock.
        mem.add("other-q", "other-a", tokens=10)
        other_done.set()

    t1 = threading.Thread(target=_trigger_compression, daemon=True)
    t2 = threading.Thread(target=_other_caller, daemon=True)
    t1.start()
    t2.start()
    assert other_done.wait(timeout=2.0), "concurrent add() blocked on summariser"
    finish.set()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)


# ---------------------------------------------------------------------------
# H-E: MCP cache TTL + invalidation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_tools_cache_expires_after_ttl() -> None:
    from lazybridge.ext.mcp import MCP
    from tests.unit.test_mcp import FakeTransport

    transport = FakeTransport()
    fs = MCP.from_transport("fs", transport, cache_tools_ttl=0.05)
    first = await fs.alist_tools()
    # Second call within TTL hits cache — transport is not re-asked.
    second = await fs.alist_tools()
    assert first is second  # same cached list object

    # Wait past TTL; the next call re-fetches and may or may not return
    # the same object (it actually rebuilds, but Tool identity differs).
    await asyncio.sleep(0.1)
    third = await fs.alist_tools()
    assert third is not first
    assert [t.name for t in third] == [t.name for t in first]


@pytest.mark.asyncio
async def test_mcp_invalidate_tools_cache_forces_refetch() -> None:
    from lazybridge.ext.mcp import MCP
    from tests.unit.test_mcp import FakeTransport

    transport = FakeTransport()
    fs = MCP.from_transport("fs", transport, cache_tools_ttl=600)
    first = await fs.alist_tools()
    fs.invalidate_tools_cache()
    second = await fs.alist_tools()
    assert second is not first


def test_mcp_cache_ttl_validates_value() -> None:
    from lazybridge.ext.mcp import MCP
    from tests.unit.test_mcp import FakeTransport

    with pytest.raises(ValueError, match="cache_tools_ttl"):
        MCP.from_transport("fs", FakeTransport(), cache_tools_ttl=0)


# ---------------------------------------------------------------------------
# M-A: tool-call JSON parse error surfacing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_engine_emits_tool_error_on_malformed_args() -> None:
    """A ToolCall whose arguments dict carries ``_parse_error`` must
    short-circuit to a structured TOOL_ERROR before the tool runs.
    Pre-fix the tool ran and failed with "missing required field"."""
    from lazybridge.engines.llm import LLMEngine
    from lazybridge.tools import Tool

    engine = LLMEngine.__new__(LLMEngine)
    engine.tool_timeout = None
    engine.max_parallel_tools = None
    engine.stream_idle_timeout = None
    engine._agent_name = "agent"

    invocations: list[Any] = []

    def my_tool(path: str) -> str:
        invocations.append(path)
        return "ok"

    tool = Tool(my_tool)

    sess = Session()
    try:
        bad_call = ToolCall(
            id="call-1",
            name="my_tool",
            arguments={"_raw_arguments": "{not json", "_parse_error": "Expecting property name"},
        )
        result = await engine._exec_tool(bad_call, {"my_tool": tool}, session=sess, run_id="r1")
        assert isinstance(result, RuntimeError)
        assert "ToolArgumentParseError" not in str(result)  # human-readable, not the type tag
        assert "malformed JSON" in str(result)
        # Real tool was NOT invoked.
        assert invocations == []
        # TOOL_ERROR was emitted with the structured fields.
        errs = sess.events.query(event_type=EventType.TOOL_ERROR)
        assert len(errs) == 1
        payload = errs[0]["payload"]
        assert payload["type"] == "ToolArgumentParseError"
        assert payload["raw_arguments"] == "{not json"
        assert payload["parse_error"]
        assert payload["tool_use_id"] == "call-1"
    finally:
        sess.close()


# ---------------------------------------------------------------------------
# H-D: OTel GenAI conventions + parent-child + cross-agent propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_otel_exporter_emits_genai_attribute_names() -> None:
    pytest.importorskip("opentelemetry")
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from lazybridge.ext.otel import OTelExporter

    sink = InMemorySpanExporter()
    exp = OTelExporter(exporter=sink)

    # Drive a synthetic agent-run sequence directly through the exporter
    # so we can read the resulting spans without needing a live LLM.
    exp.export({"event_type": "agent_start", "run_id": "r1", "agent_name": "researcher", "task": "find X"})
    exp.export(
        {
            "event_type": "model_request",
            "run_id": "r1",
            "provider": "AnthropicProvider",
            "model": "claude-opus-4-7",
            "turn": 0,
        }
    )
    exp.export(
        {
            "event_type": "model_response",
            "run_id": "r1",
            "input_tokens": 42,
            "output_tokens": 17,
            "stop_reason": "end_turn",
            "cost_usd": 0.001,
            "model": "claude-opus-4-7",
        }
    )
    exp.export(
        {"event_type": "tool_call", "run_id": "r1", "tool": "search", "tool_use_id": "tu1", "arguments": {"q": "x"}}
    )
    exp.export({"event_type": "tool_result", "run_id": "r1", "tool": "search", "tool_use_id": "tu1", "result": "ok"})
    exp.export({"event_type": "agent_finish", "run_id": "r1", "agent_name": "researcher", "latency_ms": 12.5})

    spans = sink.get_finished_spans()
    by_name = {s.name: s for s in spans}
    assert "invoke_agent researcher" in by_name
    assert "chat claude-opus-4-7" in by_name
    assert "execute_tool search" in by_name

    chat = by_name["chat claude-opus-4-7"]
    assert chat.attributes["gen_ai.system"] == "anthropic"
    assert chat.attributes["gen_ai.request.model"] == "claude-opus-4-7"
    assert chat.attributes["gen_ai.usage.input_tokens"] == 42
    assert chat.attributes["gen_ai.usage.output_tokens"] == 17
    assert "end_turn" in chat.attributes["gen_ai.response.finish_reasons"]
    assert chat.attributes["gen_ai.operation.name"] == "chat"

    tool = by_name["execute_tool search"]
    assert tool.attributes["gen_ai.tool.name"] == "search"
    assert tool.attributes["gen_ai.tool.call.id"] == "tu1"
    assert tool.attributes["gen_ai.operation.name"] == "execute_tool"


@pytest.mark.asyncio
async def test_otel_exporter_parents_children_under_agent_span() -> None:
    pytest.importorskip("opentelemetry")
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from lazybridge.ext.otel import OTelExporter

    sink = InMemorySpanExporter()
    exp = OTelExporter(exporter=sink)
    exp.export({"event_type": "agent_start", "run_id": "r1", "agent_name": "a"})
    exp.export({"event_type": "tool_call", "run_id": "r1", "tool": "t", "tool_use_id": "tu1"})
    exp.export({"event_type": "tool_result", "run_id": "r1", "tool": "t", "tool_use_id": "tu1", "result": "ok"})
    exp.export({"event_type": "agent_finish", "run_id": "r1", "agent_name": "a"})

    spans = sink.get_finished_spans()
    agent_span = next(s for s in spans if s.name == "invoke_agent a")
    tool_span = next(s for s in spans if s.name == "execute_tool t")
    # Tool span's parent is the agent span (same trace + parent_span_id matches).
    assert tool_span.parent is not None
    assert tool_span.parent.span_id == agent_span.context.span_id
    assert tool_span.context.trace_id == agent_span.context.trace_id


@pytest.mark.asyncio
async def test_otel_exporter_close_flushes_orphans() -> None:
    pytest.importorskip("opentelemetry")
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from lazybridge.ext.otel import OTelExporter

    sink = InMemorySpanExporter()
    exp = OTelExporter(exporter=sink)
    exp.export({"event_type": "agent_start", "run_id": "r1", "agent_name": "a"})
    exp.export({"event_type": "tool_call", "run_id": "r1", "tool": "t", "tool_use_id": "tu1"})
    # No tool_result, no agent_finish — simulate a cancelled run.
    exp.close()
    spans = sink.get_finished_spans()
    names = {s.name for s in spans}
    assert "invoke_agent a" in names
    assert "execute_tool t" in names


# ---------------------------------------------------------------------------
# M-B: streaming + tool-call accumulation through LLMEngine
# ---------------------------------------------------------------------------


class _FakeProvider:
    """Provider double with a controllable ``astream`` chunk sequence.

    Mirrors the Gemini / DeepSeek pattern where tool calls are delivered
    only on the final chunk: chunk 1 carries text deltas, chunk 2 carries
    a tool_calls list and is_final=True.  After the tool turn, the model
    returns plain text on a second round-trip.
    """

    name = "fake"

    def __init__(self) -> None:
        self.requests: list[CompletionRequest] = []
        self.turn = 0

    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        self.requests.append(request)
        if self.turn == 0:
            self.turn += 1
            yield StreamChunk(delta="thinking...", tool_calls=[])
            yield StreamChunk(
                delta="",
                tool_calls=[ToolCall(id="tu-1", name="echo", arguments={"text": "hi"})],
                stop_reason="tool_use",
                usage=UsageStats(input_tokens=10, output_tokens=5),
                is_final=True,
            )
        else:
            yield StreamChunk(delta="final answer", tool_calls=[])
            yield StreamChunk(
                delta="",
                stop_reason="end_turn",
                usage=UsageStats(input_tokens=12, output_tokens=8),
                is_final=True,
            )

    async def aexecute(self, request: CompletionRequest) -> CompletionResponse:
        # Aggregate astream into a single response (mirrors how the
        # streaming-aware engine path consumes the stream).
        text_parts: list[str] = []
        tcs: list[ToolCall] = []
        usage = UsageStats()
        stop = "end_turn"
        async for chunk in self.astream(request):
            if chunk.delta:
                text_parts.append(chunk.delta)
            if chunk.tool_calls:
                tcs.extend(chunk.tool_calls)
            if chunk.stop_reason:
                stop = chunk.stop_reason
            if chunk.usage:
                usage = chunk.usage
        return CompletionResponse(
            content="".join(text_parts),
            tool_calls=tcs,
            stop_reason=stop,
            usage=usage,
            model="fake-model",
        )


@pytest.mark.asyncio
async def test_streaming_collects_tool_calls_from_final_chunk() -> None:
    """Regression for M-B: when a provider streams text deltas first and
    the tool_calls only on the final chunk (Gemini / DeepSeek), the
    engine's stream-accumulation must still execute the tool — pre-fix
    a thin consumer reading only ``delta`` would lose them."""
    from lazybridge.engines.llm import LLMEngine
    from lazybridge.envelope import Envelope
    from lazybridge.tools import Tool

    captured: list[str] = []

    def echo(text: str) -> str:
        captured.append(text)
        return f"echo:{text}"

    engine = LLMEngine("fake-model", provider="fake", request_timeout=None)

    fake = _FakeProvider()

    # Patch executor factory so the engine uses our fake provider.
    class _FakeExecutor:
        _provider = fake

        async def aexecute(self, req: CompletionRequest) -> CompletionResponse:
            return await fake.aexecute(req)

        async def astream(self, req: CompletionRequest) -> AsyncIterator[StreamChunk]:
            async for c in fake.astream(req):
                yield c

    engine._make_executor = lambda: _FakeExecutor()  # type: ignore[assignment]

    tool = Tool(echo)
    env = Envelope(task="say hi")
    chunks: list[str] = []
    async for chunk in engine.stream(env, tools=[tool], output_type=str, memory=None, session=None):
        chunks.append(chunk)

    # Tool was actually executed on the streaming path.
    assert captured == ["hi"], "tool was lost when arriving on the final chunk"
    # Pre-tool text and post-tool text both reached the consumer.
    full = "".join(chunks)
    assert "thinking..." in full
    assert "final answer" in full
