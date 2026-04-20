"""Pipeline concurrency — verifies LazyTool.parallel() actually runs
participants concurrently, including when participants are themselves
pipeline tools (chain / nested parallel).

Regression guard for the bug where ``arun()`` on a pipeline tool fell back
to the sync ``func`` (driven by ``run_async()``), which blocked the event
loop thread and serialized the supposedly-parallel participants.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from lazybridge.core.types import CompletionResponse, UsageStats
from lazybridge.lazy_tool import LazyTool


# ── Fake participants ───────────────────────────────────────────────────────


class _SleepingAgent:
    """Minimal LazyAgent-shaped fake whose ``achat`` awaits ``asyncio.sleep``.

    Exposes only the attributes touched by pipeline_builders / lazy_tool so
    tests stay hermetic — no provider, no executor, no event log.
    """

    def __init__(self, name: str, delay: float, reply: str = "ok") -> None:
        self.name = name
        self._delay = delay
        self._reply = reply
        # Discriminators consulted by _is_agent_instance / _clone_for_invocation
        # / build_achain_func respectively.
        self._last_output: Any = None
        self._last_response: Any = None
        self.id = f"fake-{name}"
        self.session = None
        self._log = None
        self._executor = None
        self.output_schema = None
        self.tools: list = []
        self.native_tools: list = []

    async def achat(self, task: str, **_: Any) -> CompletionResponse:
        await asyncio.sleep(self._delay)
        return CompletionResponse(content=f"{self._reply}:{task}", usage=UsageStats())

    async def aloop(self, task: str, **kwargs: Any) -> CompletionResponse:
        return await self.achat(task, **kwargs)

    async def ajson(self, task: str, schema: Any, **kwargs: Any) -> CompletionResponse:
        return await self.achat(task, **kwargs)

    # Sync counterparts — not used by parallel/chain async paths but kept
    # defined so the builders' feature-detection (hasattr checks) is unambiguous.
    def chat(self, task: str, **_: Any) -> CompletionResponse:
        return CompletionResponse(content=f"{self._reply}:{task}", usage=UsageStats())

    def loop(self, task: str, **kwargs: Any) -> CompletionResponse:
        return self.chat(task, **kwargs)

    def json(self, task: str, schema: Any, **kwargs: Any) -> CompletionResponse:
        return self.chat(task, **kwargs)


# ── test_parallel_of_chains_runs_concurrently ───────────────────────────────


@pytest.mark.asyncio
async def test_parallel_of_chains_runs_concurrently() -> None:
    """Three chains inside a parallel must overlap, not serialize.

    Each chain sleeps 0.3s.  Serial execution would take ~0.9s; concurrent
    execution finishes near 0.3s.  The threshold is generous (< 0.6s) to
    stay stable under CI load while still catching the regression (which
    produced ~0.9s+ on the sync path).
    """
    delay = 0.3
    chains = [
        LazyTool.chain(_SleepingAgent(f"a{i}", delay), name=f"chain_{i}", description="d")
        for i in range(3)
    ]
    outer = LazyTool.parallel(*chains, name="par", description="d")

    start = time.perf_counter()
    await outer.arun({"task": "go"})
    elapsed = time.perf_counter() - start

    assert elapsed < 0.6, f"expected concurrent execution (<0.6s), got {elapsed:.3f}s"


# ── test_parallel_of_parallels_runs_concurrently ────────────────────────────


@pytest.mark.asyncio
async def test_parallel_of_parallels_runs_concurrently() -> None:
    """Nested parallels must also overlap their inner work."""
    delay = 0.3
    inners = [
        LazyTool.parallel(_SleepingAgent(f"a{i}", delay), name=f"inner_{i}", description="d")
        for i in range(3)
    ]
    outer = LazyTool.parallel(*inners, name="outer", description="d")

    start = time.perf_counter()
    await outer.arun({"task": "go"})
    elapsed = time.perf_counter() - start

    assert elapsed < 0.6, f"expected concurrent execution (<0.6s), got {elapsed:.3f}s"


# ── test_session_as_tool_parallel_concurrent ────────────────────────────────


@pytest.mark.asyncio
async def test_session_as_tool_parallel_concurrent() -> None:
    """LazySession.as_tool(mode='parallel') uses the same fix path."""
    from lazybridge.lazy_session import LazySession

    sess = LazySession()
    delay = 0.3
    chains = [
        LazyTool.chain(_SleepingAgent(f"a{i}", delay), name=f"chain_{i}", description="d")
        for i in range(3)
    ]
    par = sess.as_tool(
        name="par_via_session",
        description="d",
        participants=chains,
        mode="parallel",
    )

    start = time.perf_counter()
    await par.arun({"task": "go"})
    elapsed = time.perf_counter() - start

    assert elapsed < 0.6, f"expected concurrent execution (<0.6s), got {elapsed:.3f}s"


# ── test_arun_prefers_afunc_when_set ────────────────────────────────────────


@pytest.mark.asyncio
async def test_arun_prefers_afunc_when_set() -> None:
    """arun() must await _afunc when present, bypassing the sync func path."""
    called: dict[str, int] = {"func": 0, "afunc": 0}

    def _sync_func(task: str) -> str:
        called["func"] += 1
        return "from_sync"

    async def _async_func(task: str) -> str:
        called["afunc"] += 1
        return "from_async"

    tool = LazyTool.from_function(_sync_func, name="t", description="d")
    tool._afunc = _async_func

    result = await tool.arun({"task": "x"})
    assert result == "from_async"
    assert called == {"func": 0, "afunc": 1}


# ── test_sync_run_still_works_for_pipeline_tool ─────────────────────────────


def test_sync_run_still_works_for_pipeline_tool() -> None:
    """tool.run() on a pipeline tool must keep routing through the sync func."""
    delay = 0.01
    chain_tool = LazyTool.chain(
        _SleepingAgent("a0", delay, reply="hello"),
        name="c",
        description="d",
    )
    out = chain_tool.run({"task": "world"})
    # _SleepingAgent.achat returns "hello:world" as CompletionResponse.content,
    # which build_achain_func extracts into state.text.
    assert out == "hello:world"
