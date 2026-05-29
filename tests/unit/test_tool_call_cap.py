"""Tests for ``LLMEngine(max_tool_calls_per_turn=...)``.

Distinct from ``max_parallel_tools`` (concurrency): this caps how many of the
tool calls the model emits in a turn are actually executed.  Calls beyond the
cap get an ``is_error`` result block so the provider's per-id contract holds
and the model learns to emit fewer next turn.
"""

from __future__ import annotations

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
from lazybridge.engines.llm import LLMEngine
from lazybridge.envelope import Envelope
from lazybridge.tools import Tool


class _ToolThenAnswerProvider(BaseProvider):
    """First turn emits the given tool calls; second turn answers 'done'.

    Records every ``ToolResultContent`` it sees on the second request so tests
    can assert which calls executed and which were rejected.
    """

    default_model = "fake-tool-model"

    def __init__(self, tool_calls: list[ToolCall], **kw: Any) -> None:
        super().__init__(**kw)
        self._tool_calls = tool_calls
        self.seen_results: list[Any] = []

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
        # Capture the tool-result blocks fed back after turn 1.
        for msg in request.messages:
            for block in getattr(msg, "content", []) or []:
                if type(block).__name__ == "ToolResultContent":
                    self.seen_results.append(block)
        return CompletionResponse(content="done", usage=UsageStats(), model=self.model)

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        return self.complete(request)

    def stream(self, request: CompletionRequest):  # pragma: no cover - unused
        yield StreamChunk(delta="", stop_reason="stop")

    async def astream(self, request: CompletionRequest):  # pragma: no cover - unused
        yield StreamChunk(delta="", stop_reason="stop")


def _build_engine(provider: BaseProvider, **overrides: Any) -> LLMEngine:
    engine = LLMEngine.__new__(LLMEngine)
    engine._agent_name = "test"
    engine.model = provider.model or provider.default_model
    engine.provider = "anthropic"
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


@pytest.mark.asyncio
async def test_cap_executes_only_first_and_rejects_rest() -> None:
    """Model emits 3 calls, cap=1 → 1 runs, 2 come back as is_error results."""
    ran: list[int] = []

    async def note(idx: int) -> str:
        ran.append(idx)
        return f"ran-{idx}"

    tcs = [ToolCall(id=f"c{i}", name="note", arguments={"idx": i}) for i in range(3)]
    provider = _ToolThenAnswerProvider(model="fake-tool-model", tool_calls=tcs)
    engine = _build_engine(provider, max_tool_calls_per_turn=1)

    result = await engine.run(
        Envelope.from_task("go"), tools=[Tool(note, name="note")],
        output_type=str, memory=None, session=None,
    )

    assert result.ok
    assert ran == [0], "only the first call should execute"
    # 3 result blocks total: 1 executed + 2 rejected.
    assert len(provider.seen_results) == 3
    errs = [b for b in provider.seen_results if getattr(b, "is_error", False)]
    assert len(errs) == 2
    assert all("per turn" in b.content for b in errs)


@pytest.mark.asyncio
async def test_no_cap_runs_all_calls() -> None:
    """Default (None) preserves current behaviour: every emitted call runs."""
    ran: list[int] = []

    async def note(idx: int) -> str:
        ran.append(idx)
        return f"ran-{idx}"

    tcs = [ToolCall(id=f"c{i}", name="note", arguments={"idx": i}) for i in range(3)]
    provider = _ToolThenAnswerProvider(model="fake-tool-model", tool_calls=tcs)
    engine = _build_engine(provider)  # max_tool_calls_per_turn defaults to None

    result = await engine.run(
        Envelope.from_task("go"), tools=[Tool(note, name="note")],
        output_type=str, memory=None, session=None,
    )

    assert result.ok
    assert sorted(ran) == [0, 1, 2]


def test_constructor_rejects_zero_and_negative() -> None:
    with pytest.raises(ValueError, match="max_tool_calls_per_turn"):
        LLMEngine("gpt-5.5", max_tool_calls_per_turn=0)
    with pytest.raises(ValueError, match="max_tool_calls_per_turn"):
        LLMEngine("gpt-5.5", max_tool_calls_per_turn=-2)


def test_default_is_none() -> None:
    assert LLMEngine("gpt-5.5").max_tool_calls_per_turn is None
    # __new__ bypass still sees the class-level default.
    assert LLMEngine.__new__(LLMEngine).max_tool_calls_per_turn is None
