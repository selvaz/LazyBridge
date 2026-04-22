"""Tools — zero-ceremony callable wrappers for LLM tool use."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from lazybridge.envelope import Envelope

from lazybridge.core.tool_schema import ToolSchemaBuilder, ToolSchemaMode
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
        returns_envelope: bool = False,
    ) -> None:
        self.func = func
        self.name = name or func.__name__
        self.description = description
        self.guidance = guidance
        self.mode = mode
        self.schema_llm = schema_llm
        self.strict = strict
        #: When ``True``, ``func`` returns an ``Envelope`` instead of a
        #: plain Python value.  Engines aware of this hint will preserve
        #: the inner envelope's metadata (tokens / cost / error) when
        #: aggregating results from a turn's tool calls.  The flag is
        #: set automatically by ``wrap_tool`` for Agents wrapped via
        #: ``agent.as_tool()``.
        self.returns_envelope = returns_envelope
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
        # ``asyncio.get_event_loop`` is deprecated in 3.10+ and errors on
        # 3.13+ when no loop is running.  ``run`` is always called from an
        # already-running coroutine, so ``get_running_loop`` is the right
        # primitive — it also avoids accidentally creating a second loop.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.func(**kwargs))

    def run_sync(self, **kwargs: Any) -> Any:
        """Blocking tool invocation.

        Handles three cases so that callers never see a stray coroutine:

        * plain sync function → called directly.
        * async function → executed inside the current event loop if one
          is running (a worker thread hops out of it), otherwise on a
          fresh ``asyncio.run`` loop.  Needed because :meth:`Agent.as_tool`
          wraps the agent's ``.run()`` coroutine into ``Tool.func`` —
          ``SupervisorEngine`` / REPL callers were previously getting
          ``"<coroutine object _run at 0x...>"`` instead of the result.
        """
        if not asyncio.iscoroutinefunction(self.func):
            return self.func(**kwargs)

        coro_factory = lambda: self.func(**kwargs)  # noqa: E731
        # ``asyncio.get_running_loop`` is the forward-compatible check
        # (it raises cleanly when no loop is running, unlike the
        # deprecated ``get_event_loop``).  When a loop is running we
        # hop to a worker thread so we never try to nest.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro_factory())
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro_factory()).result()

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
    """Expose an Agent as a Tool with signature (task: str) -> Envelope.

    The inner agent's full Envelope (payload + metadata + error) is
    returned verbatim.  Engines that detect ``tool.returns_envelope``
    unpack the metadata into the outer Envelope's nested-* buckets so
    cost / tokens / errors propagate through the whole agent tree.

    At the LLM content-block boundary the Envelope is stringified via
    ``Envelope.__str__`` / ``.text()`` — the model sees the same text
    it would have seen under the old "flatten to str" contract, but
    the framework keeps the structured metadata.
    """

    async def _run(task: str) -> Envelope:  # type: ignore[name-defined]
        env = await agent.run(task)
        return env

    _run.__name__ = agent.name or "agent"
    _run.__doc__ = agent.description or f"Run the {agent.name} agent on the given task."

    return Tool(
        _run,
        name=agent.name or "agent",
        description=agent.description or f"Run the {agent.name} agent.",
        mode="signature",
        returns_envelope=True,
    )


def build_tool_map(tools: list[Any]) -> dict[str, Tool]:
    """Wrap and index tools by name."""
    import warnings

    result: dict[str, Tool] = {}
    for t in tools:
        wrapped = wrap_tool(t)
        if wrapped.name in result:
            warnings.warn(
                f"Tool name collision: '{wrapped.name}' appears more than once "
                f"in the tools list. The first registration will be replaced by "
                f"the second. Rename one of the tools to avoid silent shadowing.",
                UserWarning,
                stacklevel=2,
            )
        result[wrapped.name] = wrapped
    return result
