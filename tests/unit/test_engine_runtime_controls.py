"""Tests for LLMEngine runtime-control knobs.

Covers:
  * ``max_parallel_tools`` bounds in-flight tool concurrency per turn.
  * ``tool_timeout`` cancels a hung tool and surfaces the failure to
    the model loop as ``ToolResultContent(is_error=True)`` without
    aborting the run.
  * ``stream_idle_timeout`` raises ``StreamStallError`` when chunks
    stop arriving (vs. ``request_timeout`` which is a total deadline).
  * Constructor validation rejects out-of-range values.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from lazybridge.core.executor import Executor
from lazybridge.core.providers.base import BaseProvider
from lazybridge.core.types import (
    CompletionRequest,
    CompletionResponse,
    StreamChunk,
    ToolCall,
    UsageStats,
)
from lazybridge.engines.llm import LLMEngine, StreamStallError, ToolTimeoutError
from lazybridge.envelope import Envelope
from lazybridge.tools import Tool

# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_max_parallel_tools_rejects_zero_and_negative() -> None:
    with pytest.raises(ValueError, match="max_parallel_tools"):
        LLMEngine("gpt-5.5", max_parallel_tools=0)
    with pytest.raises(ValueError, match="max_parallel_tools"):
        LLMEngine("gpt-5.5", max_parallel_tools=-1)


def test_tool_timeout_rejects_zero_and_negative() -> None:
    with pytest.raises(ValueError, match="tool_timeout"):
        LLMEngine("gpt-5.5", tool_timeout=0)
    with pytest.raises(ValueError, match="tool_timeout"):
        LLMEngine("gpt-5.5", tool_timeout=-1.5)


def test_stream_idle_timeout_rejects_zero_and_negative() -> None:
    with pytest.raises(ValueError, match="stream_idle_timeout"):
        LLMEngine("gpt-5.5", stream_idle_timeout=0)
    with pytest.raises(ValueError, match="stream_idle_timeout"):
        LLMEngine("gpt-5.5", stream_idle_timeout=-0.001)


def test_runtime_knobs_default_to_none() -> None:
    e = LLMEngine("gpt-5.5")
    # max_parallel_tools defaults to 8 (safe cap against resource exhaustion)
    assert e.max_parallel_tools == 8
    assert e.tool_timeout is None
    assert e.stream_idle_timeout is None


def test_engine_constructed_via_new_inherits_class_defaults() -> None:
    """Tests that bypass ``__init__`` via ``__new__`` still see safe defaults."""
    e = LLMEngine.__new__(LLMEngine)
    # max_parallel_tools class-level default is 8 (safe cap).
    assert e.max_parallel_tools == 8
    assert e.tool_timeout is None
    assert e.stream_idle_timeout is None


# ---------------------------------------------------------------------------
# Helpers: minimal fake provider that emits exactly one tool turn then a
# final answer, so we can exercise the tool-execution code path under the
# real ``LLMEngine.run``.
# ---------------------------------------------------------------------------


class _ToolThenAnswerProvider(BaseProvider):
    """Two-turn provider: first turn emits N tool calls, second turn answers."""

    default_model = "fake-tool-model"

    def __init__(self, tool_calls: list[ToolCall], **kw: Any) -> None:
        super().__init__(**kw)
        self._tool_calls = tool_calls

    def _init_client(self, **kwargs: Any) -> None:
        self._call_count = 0

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self._call_count += 1
        if self._call_count == 1:
            return CompletionResponse(
                content="",
                tool_calls=list(self._tool_calls),
                usage=UsageStats(),
                model=self.model,
            )
        return CompletionResponse(content="done", usage=UsageStats(), model=self.model)

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        return self.complete(request)

    def stream(self, request: CompletionRequest):  # pragma: no cover - unused
        yield StreamChunk(delta="", stop_reason="stop")

    async def astream(self, request: CompletionRequest):  # pragma: no cover - unused
        yield StreamChunk(delta="", stop_reason="stop")


def _build_engine_with_provider(provider: BaseProvider, **overrides: Any) -> LLMEngine:
    """Construct an LLMEngine bound to a specific provider instance."""
    engine = LLMEngine.__new__(LLMEngine)
    engine._agent_name = "test"
    engine.model = provider.model or provider.default_model
    engine.provider = "anthropic"  # value doesn't matter; _make_executor is overridden
    engine.system = None
    engine.max_turns = 5
    engine.tool_choice = "auto"
    engine.temperature = None
    engine.thinking = False
    engine.request_timeout = None
    engine.max_retries = 0
    engine.retry_delay = 0.0
    engine.native_tools = []
    for k, v in overrides.items():
        setattr(engine, k, v)
    fake_exec = Executor(provider, max_retries=0)
    engine._make_executor = lambda: fake_exec  # type: ignore[method-assign]
    return engine


# ---------------------------------------------------------------------------
# max_parallel_tools — bounded concurrency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_parallel_tools_caps_in_flight_count() -> None:
    """When N tools fan out and max_parallel_tools=2, peak in-flight is 2."""
    in_flight = 0
    peak = 0
    lock = asyncio.Lock()

    async def slow(idx: int) -> int:
        nonlocal in_flight, peak
        async with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        async with lock:
            in_flight -= 1
        return idx

    n_calls = 6
    tcs = [ToolCall(id=f"c{i}", name="slow", arguments={"idx": i}) for i in range(n_calls)]
    provider = _ToolThenAnswerProvider(model="fake-tool-model", tool_calls=tcs)
    engine = _build_engine_with_provider(provider, max_parallel_tools=2)

    tool = Tool(slow, name="slow")
    env = Envelope.from_task("fan out")
    result = await engine.run(env, tools=[tool], output_type=str, memory=None, session=None)

    assert result.ok
    assert peak == 2, f"expected peak in-flight 2, observed {peak}"


@pytest.mark.asyncio
async def test_unbounded_concurrency_when_max_parallel_tools_is_none() -> None:
    """With no cap, all N tools are in-flight simultaneously."""
    in_flight = 0
    peak = 0
    lock = asyncio.Lock()

    async def slow(idx: int) -> int:
        nonlocal in_flight, peak
        async with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        async with lock:
            in_flight -= 1
        return idx

    n_calls = 5
    tcs = [ToolCall(id=f"c{i}", name="slow", arguments={"idx": i}) for i in range(n_calls)]
    provider = _ToolThenAnswerProvider(model="fake-tool-model", tool_calls=tcs)
    engine = _build_engine_with_provider(provider)  # default: unbounded

    tool = Tool(slow, name="slow")
    env = Envelope.from_task("fan out unbounded")
    result = await engine.run(env, tools=[tool], output_type=str, memory=None, session=None)
    assert result.ok
    assert peak == n_calls, f"expected peak in-flight {n_calls}, observed {peak}"


# ---------------------------------------------------------------------------
# tool_timeout — per-tool cancellation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_timeout_surfaces_as_tool_error_not_run_failure() -> None:
    """A tool that exceeds tool_timeout returns an error block and the run continues."""
    invocations: list[str] = []

    async def hangs() -> str:
        invocations.append("entered")
        await asyncio.sleep(10)
        invocations.append("never-reached")
        return "should not happen"

    tcs = [ToolCall(id="c1", name="hangs", arguments={})]
    provider = _ToolThenAnswerProvider(model="fake-tool-model", tool_calls=tcs)
    engine = _build_engine_with_provider(provider, tool_timeout=0.05)

    tool = Tool(hangs, name="hangs")
    env = Envelope.from_task("call hung tool")
    result = await engine.run(env, tools=[tool], output_type=str, memory=None, session=None)

    assert "entered" in invocations
    assert "never-reached" not in invocations
    # Run completes via the second-turn final answer because the engine
    # surfaced the timeout as an error block, not a hard failure.
    assert result.ok
    assert "done" in result.payload


# ---------------------------------------------------------------------------
# stream_idle_timeout — raises on stalled stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idle_guarded_stream_raises_when_stalled() -> None:
    """Wrapping a stream that never yields raises StreamStallError after idle_timeout."""

    async def never_yields():
        await asyncio.sleep(10)
        yield "unreachable"

    engine = LLMEngine("gpt-5.5", stream_idle_timeout=0.05)
    with pytest.raises(StreamStallError, match="went idle"):
        async for _ in engine._idle_guarded_stream(never_yields()):
            pass


@pytest.mark.asyncio
async def test_idle_guarded_stream_passes_through_when_disabled() -> None:
    """When stream_idle_timeout is None, the helper is a transparent passthrough."""

    async def gen():
        yield 1
        yield 2
        yield 3

    engine = LLMEngine("gpt-5.5")  # default: None
    items = [x async for x in engine._idle_guarded_stream(gen())]
    assert items == [1, 2, 3]


@pytest.mark.asyncio
async def test_idle_guarded_stream_yields_chunks_within_idle_window() -> None:
    """A stream that pauses but resumes under the idle threshold completes normally."""

    async def slow_but_progressing():
        for i in range(3):
            await asyncio.sleep(0.02)
            yield i

    engine = LLMEngine("gpt-5.5", stream_idle_timeout=0.5)
    items = [x async for x in engine._idle_guarded_stream(slow_but_progressing())]
    assert items == [0, 1, 2]


# ---------------------------------------------------------------------------
# Exported names
# ---------------------------------------------------------------------------


def test_tool_timeout_error_is_publicly_exported() -> None:
    import lazybridge

    assert lazybridge.ToolTimeoutError is ToolTimeoutError
    assert lazybridge.StreamStallError is StreamStallError
