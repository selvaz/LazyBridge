"""Tests for ``AgentPool`` — name-based routing + bounded recursion."""

from __future__ import annotations

from typing import Any

import pytest

from lazybridge import Agent, AgentPool, conclude
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
from lazybridge.tools import Tool


class _CallsProvider(BaseProvider):
    """Turn 1 emits ``tool_calls``; later turns answer ``final``."""

    default_model = "fake-tool-model"

    def __init__(self, tool_calls: list[ToolCall] | None = None, final: str = "answer", **kw: Any) -> None:
        super().__init__(**kw)
        self._tool_calls = tool_calls or []
        self._final = final

    def _init_client(self, **kwargs: Any) -> None:
        self._n = 0

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self._n += 1
        if self._n == 1 and self._tool_calls:
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


def _agent(provider: BaseProvider, *, name: str, tools: list[Any] | None = None) -> Agent:
    return Agent(engine=_engine(provider), name=name, tools=tools or [])


@pytest.mark.asyncio
async def test_route_delegates_to_named_agent() -> None:
    worker = _agent(_CallsProvider(model="fake-tool-model", final="from worker"), name="worker")
    pool = AgentPool()
    pool.register(worker)

    out = await pool.route("worker", "do it")

    assert out == "from worker"


@pytest.mark.asyncio
async def test_unknown_agent_returns_message_not_exception() -> None:
    pool = AgentPool()
    pool.register(_agent(_CallsProvider(model="fake-tool-model"), name="known"))

    out = await pool.route("ghost", "x")

    assert "Unknown agent 'ghost'" in out
    assert "known" in out


@pytest.mark.asyncio
async def test_depth_guard_stops_recursion() -> None:
    """At max_depth the pool refuses to descend, returning a nudge to conclude."""
    worker = _agent(_CallsProvider(model="fake-tool-model"), name="worker")
    pool = AgentPool(max_depth=3)
    pool.register(worker)

    # Simulate being already 3 levels deep.
    token = pool._depth.set(3)
    try:
        out = await pool.route("worker", "x")
    finally:
        pool._depth.reset(token)

    assert "Max routing depth 3 reached" in out


def test_max_depth_validation() -> None:
    with pytest.raises(ValueError, match="max_depth"):
        AgentPool(max_depth=0)


def test_as_tool_is_named_route() -> None:
    pool = AgentPool()
    tool = pool.as_tool()
    assert isinstance(tool, Tool)
    assert tool.name == "route"


def test_as_tool_custom_name_lets_one_agent_use_two_pools() -> None:
    team = AgentPool()
    peers = AgentPool()
    team.register(_agent(_CallsProvider(model="fake-tool-model"), name="t1"))
    peers.register(_agent(_CallsProvider(model="fake-tool-model"), name="p1"))

    # Distinct names → no collision when both pools sit on one agent.
    agent = _agent(
        _CallsProvider(model="fake-tool-model"),
        name="hub",
        tools=[team.as_tool("ask_team"), peers.as_tool("ask_peer")],
    )

    names = {t.name for t in agent._tool_map.values()}
    assert {"ask_team", "ask_peer"} <= names


def test_roster_lists_agents() -> None:
    pool = AgentPool()
    a = _agent(_CallsProvider(model="fake-tool-model"), name="alpha")
    a.description = "the alpha"
    pool.register(a)
    assert "alpha" in pool.roster()


@pytest.mark.asyncio
async def test_pool_plus_conclude_integration() -> None:
    """Orchestrator routes to a worker that concludes → answer reaches the top."""
    # Worker concludes on its first turn.
    worker_tcs = [ToolCall(id="w0", name="conclude", arguments={"message": "final via pool"})]
    worker = _agent(_CallsProvider(model="fake-tool-model", tool_calls=worker_tcs), name="worker", tools=[conclude])

    pool = AgentPool()

    # Orchestrator routes to the worker on its first turn.
    orch_tcs = [ToolCall(id="o0", name="route", arguments={"agent_name": "worker", "task": "go"})]
    orchestrator = _agent(
        _CallsProvider(model="fake-tool-model", tool_calls=orch_tcs, final="orch-fallback"),
        name="orch",
        tools=[pool.as_tool(), conclude],
    )
    pool.register(worker)

    result = await orchestrator.run("start")

    assert result.ok
    assert result.text() == "final via pool"
