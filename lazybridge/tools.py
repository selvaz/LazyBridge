"""Tools — zero-ceremony callable wrappers for LLM tool use."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from lazybridge.envelope import Envelope

from lazybridge.core.tool_schema import ToolSchemaBuilder, ToolSchemaMode
from lazybridge.core.types import ToolDefinition


@runtime_checkable
class ToolProvider(Protocol):
    """A ``tools=[...]`` entry that expands itself into one or more Tools.

    Implementors set ``_is_lazy_tool_provider = True`` and define
    ``as_tools() -> list[Tool]``. ``MCPServer`` and
    ``ExternalToolProvider`` both satisfy this protocol structurally;
    custom providers (OpenAPI imports, internal tool registries, etc.)
    can do the same — drop the instance into ``Agent(tools=[provider])``
    and ``build_tool_map`` will expand it on construction.
    """

    _is_lazy_tool_provider: bool

    def as_tools(self) -> list[Tool]: ...


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
        mode: Literal["signature", "llm", "hybrid"] = "signature",
        schema_llm: Any | None = None,
        strict: bool = False,
        returns_envelope: bool = False,
    ) -> None:
        self.func = func
        self.name = name or func.__name__
        self.description = description
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

    @classmethod
    def from_schema(
        cls,
        name: str,
        description: str,
        parameters: dict[str, Any],
        func: Callable[..., Any],
        *,
        strict: bool = False,
        returns_envelope: bool = False,
    ) -> Tool:
        """Create a Tool with a pre-built JSON Schema for parameters.

        Use this when the schema is already known (from MCP, OpenAPI, a
        third-party tool registry, ...) and signature introspection would
        either be unavailable or produce the wrong shape.

        ``parameters`` must be a JSON Schema object (the same shape that
        ``ToolDefinition.parameters`` carries).
        """
        tool = cls.__new__(cls)
        tool.func = func
        tool.name = name
        tool.description = description
        tool.mode = "signature"  # unused — we set ``_definition`` directly
        tool.schema_llm = None
        tool.strict = strict
        tool.returns_envelope = returns_envelope
        tool._definition = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            strict=strict,
        )
        tool._lock = threading.Lock()
        return tool

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
        # Propagate the caller's contextvars context (OTel, structured
        # logging, request IDs) into the worker loop.  A raw
        # ``asyncio.run`` on a fresh thread would start in an empty
        # context and silently break observability for sync callers.
        import concurrent.futures
        import contextvars

        ctx = contextvars.copy_context()

        def _run() -> Any:
            return ctx.run(asyncio.run, coro_factory())

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_run).result()

    def __repr__(self) -> str:
        return f"Tool({self.name!r})"


def wrap_tool(obj: Any) -> Tool:
    """Convert a raw callable or Agent into a Tool. Returns Tool unchanged.

    Agent-likes are recognised by the duck-typed ``_is_lazy_agent`` marker so
    test doubles (``lazybridge.testing.MockAgent``) and custom Agent-compatible
    classes share the same composition path as the real ``Agent`` — nested
    envelope metadata propagates through ``returns_envelope=True``.
    """
    if isinstance(obj, Tool):
        return obj
    # Duck-typed: any object flagged ``_is_lazy_agent`` with ``.run`` /
    # ``.name`` / ``.description`` is treated as an Agent for composition.
    if getattr(obj, "_is_lazy_agent", False):
        return _agent_as_tool(obj)
    if callable(obj):
        return Tool(obj)
    raise TypeError(f"Cannot convert {type(obj).__name__!r} to Tool")


def _agent_as_tool(agent: Any) -> Tool:
    """Expose an Agent as a Tool with signature ``(task: str) -> Envelope``.

    Routes through ``agent.as_tool()`` (the verify-aware path with
    ``verify=None``) so the two construction paths produce a structurally
    identical Tool. ``MockAgent`` and other duck-typed agent doubles
    (no ``as_tool`` method) fall back to the inline shim below.
    """
    if hasattr(agent, "as_tool"):
        return agent.as_tool()

    async def _run(task: str) -> Envelope:  # type: ignore[name-defined]
        return await agent.run(task)

    _run.__name__ = agent.name or "agent"
    _run.__doc__ = agent.description or f"Run the {agent.name} agent on the given task."

    return Tool(
        _run,
        name=agent.name or "agent",
        description=agent.description or f"Run the {agent.name} agent.",
        mode="signature",
        returns_envelope=True,
    )


def build_tool_map(
    tools: list[Any],
    *,
    collision_policy: Literal["raise", "replace"] = "raise",
) -> dict[str, Tool]:
    """Wrap and index tools by name.

    Items in ``tools`` may be:
      - a callable / Agent / :class:`Tool` (wrapped via :func:`wrap_tool`);
      - a **tool provider** — any object with ``_is_lazy_tool_provider = True``
        and an ``as_tools() -> list[Tool]`` method.  The provider is expanded
        into its constituent tools.  This is how, e.g., an MCP server lands
        in ``Agent(tools=[github])`` and contributes its whole tool surface.

    Args:
        collision_policy: What to do when two tools share a name.
            ``"raise"`` (default) — raise ``ValueError`` immediately so the
            duplicate is caught at construction time rather than silently
            changing which tool the LLM invokes.
            ``"replace"`` — keep the last registration and emit a
            ``UserWarning`` (previous behaviour, useful when composing MCP
            servers that may overlap on common names like ``search``).
    """
    import warnings

    result: dict[str, Tool] = {}
    seen_warnings: set[str] = set()
    for t in tools:
        if getattr(t, "_is_lazy_tool_provider", False):
            expanded = list(t.as_tools())
        else:
            expanded = [wrap_tool(t)]
        for wrapped in expanded:
            if wrapped.name in result:
                if collision_policy == "raise":
                    raise ValueError(
                        f"Tool name collision: '{wrapped.name}' appears more than once "
                        f"in the tools list. Rename one of the tools or pass "
                        f"collision_policy='replace' to keep the last registration."
                    )
                # collision_policy == "replace": warn once per name.
                if wrapped.name not in seen_warnings:
                    warnings.warn(
                        f"Tool name collision: '{wrapped.name}' appears more than once "
                        f"in the tools list. The first registration will be replaced by "
                        f"the second. Rename one of the tools to avoid silent shadowing.",
                        UserWarning,
                        stacklevel=4,
                    )
                    seen_warnings.add(wrapped.name)
            result[wrapped.name] = wrapped
    return result
