"""Tools — zero-ceremony callable wrappers for LLM tool use."""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Callable
from typing import Any, Literal

from lazybridge.core.tool_schema import ToolSchemaMode, ToolSchemaBuilder
from lazybridge.core.types import ToolDefinition


class Tool:
    """Wraps any Python callable as an LLM-accessible tool.

    Pass raw functions directly; Tool auto-wraps them on the agent level.
    Use Tool(fn, ...) only when you need explicit configuration.
    """

    def __init__(
        self,
        func: Callable,
        *,
        name: str | None = None,
        description: str | None = None,
        guidance: str | None = None,
        mode: Literal["signature", "llm", "hybrid"] = "signature",
        schema_llm: Any | None = None,
        strict: bool = False,
    ) -> None:
        self.func = func
        self.name = name or func.__name__
        self.description = description
        self.guidance = guidance
        self.mode = mode
        self.schema_llm = schema_llm
        self.strict = strict
        self._definition: ToolDefinition | None = None
        self._lock = threading.Lock()

    def definition(self) -> ToolDefinition:
        if self._definition is not None:
            return self._definition
        with self._lock:
            if self._definition is not None:
                return self._definition
            schema_mode = ToolSchemaMode(self.mode)
            builder = ToolSchemaBuilder()
            self._definition = builder.build(
                self.func,
                name=self.name,
                description=self.description,
                strict=self.strict,
                mode=schema_mode,
                schema_llm=self.schema_llm,
            )
            return self._definition

    async def run(self, **kwargs: Any) -> Any:
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(**kwargs)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.func(**kwargs))

    def run_sync(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)

    def __repr__(self) -> str:
        return f"Tool({self.name!r})"


def wrap_tool(obj: Any) -> Tool:
    """Convert a raw callable or Agent into a Tool. Returns Tool unchanged."""
    if isinstance(obj, Tool):
        return obj
    # Circular import guard: Agent exposes .name, .description, .definition(), .__call__
    if hasattr(obj, "_is_lazy_agent"):
        from lazybridge.agent import Agent
        if isinstance(obj, Agent):
            return _agent_as_tool(obj)
    if callable(obj):
        return Tool(obj)
    raise TypeError(f"Cannot convert {type(obj).__name__!r} to Tool")


def _agent_as_tool(agent: Any) -> Tool:
    """Expose an Agent as a Tool with signature (task: str) -> str."""
    import asyncio

    async def _run(task: str) -> str:
        env = await agent.run(task)
        return env.text()

    _run.__name__ = agent.name or "agent"
    _run.__doc__ = agent.description or f"Run the {agent.name} agent on the given task."

    return Tool(
        _run,
        name=agent.name or "agent",
        description=agent.description or f"Run the {agent.name} agent.",
        mode="signature",
    )


def build_tool_map(tools: list[Any]) -> dict[str, Tool]:
    """Wrap and index tools by name."""
    result: dict[str, Tool] = {}
    for t in tools:
        wrapped = wrap_tool(t)
        result[wrapped.name] = wrapped
    return result
