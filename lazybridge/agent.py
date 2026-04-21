"""Agent — the single public-facing abstraction for LazyBridge v1.0."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator, Callable
from typing import Any

from lazybridge.envelope import Envelope
from lazybridge.tools import Tool, build_tool_map, wrap_tool


class Agent:
    """Universal agent — delegates execution to a swappable Engine.

    Tier 1 (2 lines):
        Agent("claude-opus-4-7")("hello").text()

    Tier 2 (with tools):
        Agent("claude-opus-4-7", tools=[search])("find news").text()

    Tier 3 (structured output):
        Agent("claude-opus-4-7", output=Summary)("...").payload.title

    Tier 4 (chain / parallel):
        Agent.chain(researcher, writer)("AI trends").text()
        Agent.parallel(a, b, c)("task")   # → list[Envelope]

    Tier 5/6 (Plan / full config):
        Agent(engine=Plan(...), tools=[...], output=Report, session=session)
    """

    _is_lazy_agent = True  # recognised by wrap_tool()

    def __init__(
        self,
        engine_or_model: "str | Any" = "claude-opus-4-7",
        tools: "list[Tool | Callable | Agent]" = (),
        output: type = str,
        memory: "Any | None" = None,
        sources: "list[Any]" = (),
        guard: "Any | None" = None,
        verify: "Agent | None" = None,
        max_verify: int = 3,
        name: str | None = None,
        description: str | None = None,
        session: "Any | None" = None,
        # Convenience: pass provider + model separately
        # Agent("anthropic", model="top") or Agent("anthropic", model="claude-opus-4-7")
        model: str | None = None,
    ) -> None:
        from lazybridge.engines.llm import LLMEngine

        if isinstance(engine_or_model, str):
            model_str = model or engine_or_model
            self.engine: Any = LLMEngine(model_str)
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

        self.engine._agent_name = self.name

        # PlanCompiler runs at construction time
        if hasattr(self.engine, "_validate"):
            self.engine._validate(self._tool_map)

    # ------------------------------------------------------------------
    # Core async API
    # ------------------------------------------------------------------

    async def run(self, task: "str | Envelope") -> Envelope:
        env = self._to_envelope(task)
        env = self._inject_sources(env)

        if self.guard:
            action = await self.guard.acheck_input(env.task or "")
            if not action.allowed:
                return Envelope.error_envelope(ValueError(action.message or "Blocked by guard"))
            if action.modified_text is not None:
                env = Envelope(task=action.modified_text, context=env.context,
                               payload=action.modified_text)

        if self.verify:
            from lazybridge.evals import verify_with_retry
            result = await verify_with_retry(self, env, self.verify, max_verify=self.max_verify)
        else:
            result = await self._run_engine(env)

        if self.guard and result.ok:
            action = await self.guard.acheck_output(result.text())
            if not action.allowed:
                from lazybridge.envelope import ErrorInfo
                result = Envelope(
                    task=result.task,
                    payload=result.payload,
                    error=ErrorInfo(type="GuardBlocked",
                                   message=action.message or "Output blocked"),
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

    async def stream(self, task: "str | Envelope") -> AsyncGenerator[str, None]:
        """Stream LLM tokens across the full tool-calling loop."""
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
    # Sync API
    # ------------------------------------------------------------------

    def __call__(self, task: "str | Envelope") -> Envelope:
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

    # ------------------------------------------------------------------
    # Tool exposure
    # ------------------------------------------------------------------

    def as_tool(
        self,
        name: str | None = None,
        description: str | None = None,
    ) -> Tool:
        """Return a Tool that wraps this agent.

        The tool schema is: ``(task: str) -> str``.
        Use this to plug one agent into another agent's tools list.

            researcher = Agent("claude-opus-4-7", tools=[search])
            orchestrator = Agent("claude-opus-4-7",
                                 tools=[researcher.as_tool("researcher",
                                                           "Search and summarise papers")])
        """
        agent = self
        effective_name = name or self.name
        effective_desc = description or self.description or f"Run the {effective_name} agent."

        async def _run(task: str) -> str:
            env = await agent.run(task)
            return env.text()

        _run.__name__ = effective_name
        _run.__doc__ = effective_desc

        return Tool(_run, name=effective_name, description=effective_desc, mode="signature")

    def definition(self) -> Any:
        """ToolDefinition for this agent — used when passed in tools=[] of another agent."""
        return self.as_tool().definition()

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def chain(cls, *agents: "Agent", **kwargs: Any) -> "Agent":
        """Run agents sequentially: output of each becomes input to the next."""
        from lazybridge.engines.plan import Plan, Step

        steps = [Step(target=a, name=a.name) for a in agents]
        plan = Plan(*steps)
        tools = [a.as_tool() for a in agents]
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
        """Run all agents concurrently on the same task → list[Envelope]."""
        return _ParallelAgent(
            agents=list(agents),
            concurrency_limit=concurrency_limit,
            step_timeout=step_timeout,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_envelope(task: "str | Envelope") -> Envelope:
        if isinstance(task, Envelope):
            return task
        return Envelope.from_task(str(task))

    def _inject_sources(self, env: Envelope) -> Envelope:
        """Build context string from sources (live view — read at call time)."""
        if not self.sources:
            return env
        parts: list[str] = []
        if env.context:
            parts.append(env.context)
        for src in self.sources:
            text = _read_source(src)
            if text:
                parts.append(text)
        merged = "\n\n".join(p for p in parts if p)
        return Envelope(task=env.task, context=merged or env.context, payload=env.payload)


def _read_source(src: Any) -> str:
    """Extract a string from any source object — Memory, Store, callable, str."""
    # Memory / Store / objects with .text() or .to_text()
    if hasattr(src, "to_text"):
        return src.to_text()
    if hasattr(src, "text"):
        val = src.text
        return val() if callable(val) else str(val)
    # Callable — call it
    if callable(src):
        return str(src())
    # Plain string
    return str(src)


class _ParallelAgent:
    """Runs multiple agents concurrently on the same task. Returns list[Envelope]."""

    _is_lazy_agent = True

    def __init__(
        self,
        agents: "list[Agent]",
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

    async def run(self, task: "str | Envelope") -> "list[Envelope]":
        env = Agent._to_envelope(task) if isinstance(task, str) else task
        sem = asyncio.Semaphore(self.concurrency_limit) if self.concurrency_limit else None

        async def _run_one(agent: "Agent") -> Envelope:
            async def _coro() -> Envelope:
                if self.step_timeout:
                    return await asyncio.wait_for(agent.run(env), timeout=self.step_timeout)
                return await agent.run(env)

            if sem:
                async with sem:
                    return await _coro()
            return await _coro()

        results = await asyncio.gather(*[_run_one(a) for a in self.agents],
                                       return_exceptions=True)
        return [
            r if isinstance(r, Envelope)
            else Envelope.error_envelope(r if isinstance(r, Exception) else RuntimeError(str(r)))
            for r in results
        ]

    def __call__(self, task: "str | Envelope") -> "list[Envelope]":
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
