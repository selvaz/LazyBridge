"""Agent — the single public-facing abstraction for LazyBridge v1.0."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from lazybridge.envelope import Envelope
from lazybridge.tools import Tool, build_tool_map, wrap_tool


class Agent:
    """Universal agent — delegates execution to a swappable Engine.

    Tier 1 (2 lines):
        Agent("claude-opus-4-7")("hello").text()

    Tier 2 (with tools):
        Agent("claude-opus-4-7", tools=[search])("find news").text()

    Tier 4 (chain):
        Agent.chain(researcher, writer)("AI trends").text()

    Tier 4b (parallel):
        Agent.parallel(a, b, c)("task")  # → list[Envelope]

    Tier 5/6 (Plan / full config):
        Agent(engine=Plan(...), tools=[...], output=Report, session=session)
    """

    _is_lazy_agent = True  # sentinel for wrap_tool() detection

    def __init__(
        self,
        engine_or_model: "str | Any" = "claude-opus-4-7",
        tools: "list[Tool | Callable | Agent]" = (),
        output: type = str,
        memory: "Any | None" = None,
        sources: list = (),
        guard: "Any | None" = None,
        verify: "Agent | None" = None,
        max_verify: int = 3,
        name: str | None = None,
        description: str | None = None,
        session: "Any | None" = None,
    ) -> None:
        from lazybridge.engines.llm import LLMEngine

        if isinstance(engine_or_model, str):
            self.engine: Any = LLMEngine(engine_or_model)
        else:
            self.engine = engine_or_model

        self._tools_raw = list(tools)
        self._tool_map: dict[str, Tool] = build_tool_map(self._tools_raw)
        self.output = output
        self.memory = memory
        self.sources = list(sources)
        self.guard = guard
        self.verify = verify
        self.max_verify = max_verify
        self.name = name or getattr(self.engine, "model", "agent")
        self.description = description
        self.session = session

        # Pass agent name down to engine for event naming
        self.engine._agent_name = self.name

        # Validate Plan at construction time
        if hasattr(self.engine, "_validate"):
            self.engine._validate(self._tool_map)

    # ------------------------------------------------------------------
    # Core async API
    # ------------------------------------------------------------------

    async def run(self, task: "str | Envelope") -> Envelope:
        env = self._to_envelope(task)

        # Inject live sources into context
        env = self._inject_sources(env)

        # Guard: check input
        if self.guard:
            action = await self.guard.acheck_input(env.task or "")
            if not action.allowed:
                return Envelope.error_envelope(ValueError(action.message or "Blocked by guard"))
            if action.modified_text is not None:
                env = Envelope(task=action.modified_text, context=env.context, payload=action.modified_text)

        # Verify loop
        if self.verify:
            from lazybridge.evals import verify_with_retry
            result = await verify_with_retry(self, env, self.verify, max_verify=self.max_verify)
        else:
            result = await self._run_engine(env)

        # Guard: check output
        if self.guard and result.ok:
            action = await self.guard.acheck_output(result.text())
            if not action.allowed:
                from lazybridge.envelope import ErrorInfo
                result = Envelope(
                    task=result.task,
                    payload=result.payload,
                    error=ErrorInfo(type="GuardBlocked", message=action.message or "Output blocked"),
                    metadata=result.metadata,
                )

        return result

    async def _run_engine(self, env: Envelope) -> Envelope:
        return await self.engine.run(
            env,
            tools=list(self._tool_map.values()),
            output_type=self.output,
            memory=self.memory,
            session=self.session,
        )

    async def stream(self, task: "str | Envelope") -> AsyncIterator[str]:
        env = self._to_envelope(task)
        env = self._inject_sources(env)
        async for chunk in self.engine.stream(
            env,
            tools=list(self._tool_map.values()),
            output_type=self.output,
            memory=self.memory,
            session=self.session,
        ):
            yield chunk

    # ------------------------------------------------------------------
    # Sync API — runs the event loop
    # ------------------------------------------------------------------

    def __call__(self, task: "str | Envelope") -> Envelope:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # In Jupyter / async context — run in thread executor
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(asyncio.run, self.run(task))
                    return fut.result()
            return loop.run_until_complete(self.run(task))
        except RuntimeError:
            return asyncio.run(self.run(task))

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def chain(cls, *agents: "Agent", **kwargs: Any) -> "Agent":
        """Return a new Agent whose engine runs ``agents`` sequentially."""
        from lazybridge.engines.plan import Plan, Step

        steps = [Step(target=a, name=a.name) for a in agents]
        plan = Plan(*steps)
        tools = [wrap_tool(a) for a in agents]
        name = kwargs.pop("name", "chain")
        return cls(engine=plan, tools=tools, name=name, **kwargs)

    @classmethod
    def parallel(
        cls,
        *agents: "Agent",
        concurrency_limit: int | None = None,
        step_timeout: float | None = None,
        **kwargs: Any,
    ) -> "_ParallelAgent":
        """Return a _ParallelAgent that runs all agents concurrently → list[Envelope]."""
        return _ParallelAgent(
            agents=list(agents),
            concurrency_limit=concurrency_limit,
            step_timeout=step_timeout,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Tool Protocol — makes Agent usable as a tool in another Agent
    # ------------------------------------------------------------------

    def definition(self) -> Any:
        from lazybridge.tools import Tool, wrap_tool
        return wrap_tool(self).definition()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_envelope(task: "str | Envelope") -> Envelope:
        if isinstance(task, Envelope):
            return task
        return Envelope.from_task(str(task))

    def _inject_sources(self, env: Envelope) -> Envelope:
        if not self.sources:
            return env
        parts: list[str] = []
        if env.context:
            parts.append(env.context)
        for src in self.sources:
            if hasattr(src, "text"):
                parts.append(src.text())
            elif callable(src):
                parts.append(str(src()))
            else:
                parts.append(str(src))
        merged = "\n\n".join(p for p in parts if p)
        return Envelope(task=env.task, context=merged or env.context, payload=env.payload)


class _ParallelAgent:
    """Runs multiple agents concurrently on the same task. Returns list[Envelope]."""

    _is_lazy_agent = True

    def __init__(
        self,
        agents: list[Agent],
        *,
        concurrency_limit: int | None = None,
        step_timeout: float | None = None,
        name: str = "parallel",
        description: str | None = None,
        session: Any | None = None,
    ) -> None:
        self.agents = agents
        self.concurrency_limit = concurrency_limit
        self.step_timeout = step_timeout
        self.name = name
        self.description = description
        self.session = session

    async def run(self, task: "str | Envelope") -> list[Envelope]:
        env = Agent._to_envelope(task) if isinstance(task, str) else task
        sem = asyncio.Semaphore(self.concurrency_limit) if self.concurrency_limit else None

        async def _run_one(agent: Agent) -> Envelope:
            coro = agent.run(env)
            if sem:
                async with sem:
                    coro2 = agent.run(env)
                    if self.step_timeout:
                        return await asyncio.wait_for(coro2, timeout=self.step_timeout)
                    return await coro2
            if self.step_timeout:
                return await asyncio.wait_for(coro, timeout=self.step_timeout)
            return await coro

        results = await asyncio.gather(*[_run_one(a) for a in self.agents], return_exceptions=True)
        return [
            r if isinstance(r, Envelope) else Envelope.error_envelope(r if isinstance(r, Exception) else RuntimeError(str(r)))
            for r in results
        ]

    def __call__(self, task: "str | Envelope") -> list[Envelope]:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(asyncio.run, self.run(task))
                    return fut.result()
            return loop.run_until_complete(self.run(task))
        except RuntimeError:
            return asyncio.run(self.run(task))
