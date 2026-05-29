"""Tests for ``conclude`` — non-local exit across nested agent boundaries.

A ``conclude(msg)`` raised by any tool must unwind the whole call chain and
surface as ``Envelope(payload=msg)`` from the originating top-level
``Agent.run`` — never get swallowed by ``gather`` into a plain tool result.
"""

from __future__ import annotations

from typing import Any

import pytest

from lazybridge import Agent, ConcludeSignal, conclude
from lazybridge.core.executor import Executor
from lazybridge.core.providers.base import BaseProvider
from lazybridge.core.types import (
    CompletionRequest,
    CompletionResponse,
    StreamChunk,
    ToolCall,
    UsageStats,
)
from lazybridge.engines.llm import LLMEngine
from lazybridge.envelope import Envelope
from lazybridge.tools import Tool


class _CallsProvider(BaseProvider):
    """Turn 1 emits ``tool_calls``; later turns answer ``final``."""

    default_model = "fake-tool-model"

    def __init__(self, tool_calls: list[ToolCall], final: str = "fallback", **kw: Any) -> None:
        super().__init__(**kw)
        self._tool_calls = tool_calls
        self._final = final

    def _init_client(self, **kwargs: Any) -> None:
        self._n = 0

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self._n += 1
        if self._n == 1:
            return CompletionResponse(
                content="", tool_calls=list(self._tool_calls), usage=UsageStats(), model=self.model
            )
        return CompletionResponse(content=self._final, usage=UsageStats(), model=self.model)

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        return self.complete(request)

    def stream(self, request: CompletionRequest):  # pragma: no cover
        yield StreamChunk(delta="", stop_reason="stop")

    async def astream(self, request: CompletionRequest):  # pragma: no cover
        yield StreamChunk(delta="", stop_reason="stop")


def _engine(provider: BaseProvider, **overrides: Any) -> LLMEngine:
    e = LLMEngine.__new__(LLMEngine)
    e._agent_name = "test"
    e.model = provider.model or provider.default_model
    e.provider = "anthropic"
    e.system = None
    e.max_turns = 5
    e.tool_choice = "auto"
    e.temperature = None
    e.thinking = False
    e.request_timeout = None
    e.max_retries = 0
    e.retry_delay = 0.0
    e.native_tools = []
    for k, v in overrides.items():
        setattr(e, k, v)
    fake_exec = Executor(provider, max_retries=0)
    e._make_executor = lambda: fake_exec  # type: ignore[method-assign]
    return e


def _agent(provider: BaseProvider, *, name: str, tools: list[Any]) -> Agent:
    """Build a real Agent bound to a stub provider's engine."""
    agent = Agent(engine=_engine(provider), name=name, tools=tools)
    return agent


@pytest.mark.asyncio
async def test_conclude_direct_tool() -> None:
    """conclude called directly (no nesting) → Envelope.ok with the message."""
    tcs = [ToolCall(id="c0", name="conclude", arguments={"message": "the answer"})]
    agent = _agent(_CallsProvider(model="fake-tool-model", tool_calls=tcs),
                   name="solo", tools=[conclude])

    result = await agent.run("question")

    assert result.ok
    assert result.text() == "the answer"


@pytest.mark.asyncio
async def test_conclude_propagates_through_nesting() -> None:
    """Inner agent concludes; the message returns from the OUTER agent.run."""
    # Inner agent: its first turn calls conclude.
    inner_tcs = [ToolCall(id="i0", name="conclude", arguments={"message": "deep answer"})]
    inner = _agent(_CallsProvider(model="fake-tool-model", tool_calls=inner_tcs),
                   name="inner", tools=[conclude])

    # Outer agent: its first turn delegates to inner (agent-as-tool).
    outer_tcs = [ToolCall(id="o0", name="inner", arguments={"task": "dig"})]
    outer = _agent(_CallsProvider(model="fake-tool-model", tool_calls=outer_tcs, final="outer-fallback"),
                   name="outer", tools=[inner])

    result = await outer.run("start")

    assert result.ok
    # If the signal had been swallowed, outer would have continued and
    # answered 'outer-fallback'.  The deep conclude must win.
    assert result.text() == "deep answer"


@pytest.mark.asyncio
async def test_conclude_signal_is_base_exception() -> None:
    """ConcludeSignal must bypass ``except Exception`` handlers."""
    assert issubclass(ConcludeSignal, BaseException)
    assert not issubclass(ConcludeSignal, Exception)
    with pytest.raises(ConcludeSignal) as ei:
        conclude("x")
    assert ei.value.message == "x"
