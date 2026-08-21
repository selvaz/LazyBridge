"""Tools — zero-ceremony callable wrappers for LLM tool use."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import inspect
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from lazybridge.envelope import Envelope

from lazybridge._asyncbridge import abandoned_tasks, run_coroutine_blocking
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


class _AbandonedWorkers:
    """Census of workers still running after their caller gave up on them.

    Process-wide on purpose: what matters operationally is how many threads
    this process is leaking, not how many any single Tool leaked.
    """

    def __init__(self) -> None:
        self._live: set[threading.Thread] = set()

    def record(self, worker: threading.Thread) -> int:
        """Register ``worker``; return how many abandoned workers still run.

        Pruning belongs here, not in the worker itself: one that finished in
        the instant between its deadline expiring and being recorded would
        drop itself before it was ever added, then sit in the set as a dead
        thread inflating the count forever.
        """
        self._live.add(worker)
        for finished in tuple(self._live):
            if not finished.is_alive():
                self._live.discard(finished)
        return len(self._live)

    def __len__(self) -> int:
        return len(self._live)

    def clear(self) -> None:
        self._live.clear()


_abandoned = _AbandonedWorkers()

#: Sentinel distinguishing "caller omitted strict=" from "caller passed strict=False".
#: Needed so ``Tool.wrap(base, strict=False)`` can override a base with ``strict=True``.
_UNSET_BOOL: Any = object()


class ToolTimeoutError(Exception):
    """A tool ran past the time it was given and the caller stopped waiting.

    Reported to the model loop as a failed tool result, not raised out of the
    run: the model sees the timeout and carries on with what it does have.
    Raised by ``Tool.timeout`` and, for the outer per-call bound, by
    ``LLMEngine.tool_timeout``.  For a synchronous tool the work is abandoned
    rather than stopped -- see :meth:`Tool._run_bounded`.
    """

    def __init__(self, message: str, *, tool_name: str | None = None, timeout: float | None = None) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.timeout = timeout


def check_timeout(value: float | None, label: str) -> float | None:
    """Reject a deadline that can never be met, at configuration time.

    Zero or negative would otherwise be accepted and then fire immediately on
    every call — for a side-effecting tool that means the caller sees a
    timeout while the work runs on anyway.  Mirrors the same check on
    ``LLMEngine(tool_timeout=)``.
    """
    if value is not None and value <= 0:
        raise ValueError(f"{label} must be > 0 or None, got {value!r}")
    return value


async def _stop_task(task: asyncio.Task, grace: float) -> None:
    """Cancel a tool task and give it a BOUNDED chance to unwind.

    Bounded, because cancelling does not guarantee stopping: a coroutine may
    catch ``CancelledError`` and carry on, or spend arbitrarily long in
    cleanup.  Awaiting that unconditionally would put the hang back exactly
    where the deadline was supposed to remove it.  Past the grace period the
    task is abandoned like a sync worker — left running, with its eventual
    outcome read so it cannot resurface as an unretrieved exception.

    Two things this cannot reach, both asyncio's nature rather than gaps in
    the bound.  A coroutine that BLOCKS the loop instead of yielding — CPU
    work or a sync call inside ``async def``, body or cleanup — stops
    everything while it runs, including the clock below; such work belongs in
    a ``def`` tool, where the thread in ``_run_bounded`` handles it.  And a
    task abandoned on a loop we do not own still delays that loop's shutdown:
    ``asyncio.run`` gathers every pending task on the way out.  Our own
    bridge is taught to skip these (``_asyncbridge.abandoned_tasks``), so
    ``run_sync`` returns on time; a caller awaiting inside their own
    ``asyncio.run`` gets the exception on time but may wait at exit.
    """
    task.cancel()
    # Attached before the wait, not only after it: cleanup that raises INSIDE
    # the grace period leaves the task done with an exception nobody read,
    # which asyncio reports much later and out of context.
    task.add_done_callback(lambda t: t.cancelled() or t.exception())
    with contextlib.suppress(BaseException):
        await asyncio.wait({task}, timeout=grace)
    if not task.done():
        abandoned_tasks.add(task)


async def _bounded_await(coro: Any, limit: float, on_timeout: Callable[[], Exception], *, grace: float) -> Any:
    """Await ``coro`` under ``limit``, telling the two timeouts apart.

    ``wait_for`` cannot: it raises ``TimeoutError`` for its own deadline and
    relays one raised by the tool, and the two mean opposite things.  An HTTP
    client reporting its own deadline is a tool failure the model should see
    as such — reported as ``TOOL_TIMEOUT`` it would tell the model to retry
    smaller when the real answer is that the endpoint is down.  Completion is
    the test: only a coroutine that never finished timed out on OUR clock.
    """
    task = asyncio.ensure_future(coro)
    try:
        done, _ = await asyncio.wait({task}, timeout=limit)
    except BaseException:
        # ``wait`` does not touch the task it was given, so a caller cancelled
        # from outside (an ``Agent(timeout=)`` firing inside a longer tool
        # bound) would otherwise leave the tool running and free to complete
        # its side effect afterwards.
        await _stop_task(task, grace)
        raise
    if not done:
        await _stop_task(task, grace)
        raise on_timeout()
    return task.result()


async def run_tool_bounded(tool: Any, arguments: dict[str, Any], timeout: float | None) -> Any:
    """Invoke ``tool`` under the tighter of its own bound and ``timeout``.

    The one place engines should call a tool when they carry a per-call
    deadline of their own.  ``asyncio.wait_for(tool.run(...))`` is not an
    equivalent: a synchronous tool runs in an executor, and an executor
    future that has already started ignores cancellation, so ``wait_for``
    waits forever for a cancellation that never lands.

    Raises :class:`ToolTimeoutError`.  Anything whose execution path we do
    not own — a duck-typed tool, or a subclass that overrides ``run`` to add
    authorization or tracing — falls back to ``wait_for``.  That still bounds
    an async tool, and it is the honest limit: reaching past an override into
    the base dispatch would silently skip what the override is there to do.
    """
    own = getattr(tool, "timeout", None)
    limit = own if own is not None else timeout
    if limit is None:
        return await tool.run(**arguments)
    dispatch = getattr(tool, "_dispatch", None)
    if dispatch is not None and getattr(type(tool), "run", None) is Tool.run:
        return await dispatch(arguments, limit)
    return await _bounded_await(
        tool.run(**arguments),
        limit,
        lambda: ToolTimeoutError(f"Tool {tool.name!r} timed out after {limit}s", tool_name=tool.name, timeout=limit),
        grace=getattr(tool, "cancel_grace_seconds", Tool.cancel_grace_seconds),
    )


class Tool:
    """Wraps any Python callable as an LLM-accessible tool.

    Pass raw functions directly; Tool auto-wraps them on the agent level.
    Use Tool(fn, ...) only when you need explicit configuration.
    """

    #: Validate and coerce LLM-provided arguments against the function
    #: signature (Pydantic-backed) before every dispatch.  Bad arguments
    #: raise :class:`~lazybridge.core.tool_schema.ToolArgumentValidationError`
    #: with a readable message the engine can feed back to the model,
    #: instead of an opaque ``TypeError`` from inside the user function.
    #: Class-level so instances built via ``from_schema`` (``__new__``)
    #: inherit it; set ``tool.validate_args = False`` to opt out.
    validate_args: bool = True

    #: How long a cancelled async tool is given to unwind before it is
    #: abandoned too.  A well-behaved coroutine unwinds in microseconds;
    #: anything slower is doing heavy cleanup or ignoring cancellation.
    cancel_grace_seconds: float = 1.0

    #: How many abandoned workers may pile up before ``_abandon`` warns.
    #: A handful is ordinary — one slow call per turn, each finishing later.
    #: Sustained growth is a tool that hangs on every attempt.
    abandoned_worker_warning_threshold: int = 8

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
        agent_memory: Any | None = None,
        agent_store: Any | None = None,
        timeout: float | None = None,
    ) -> None:
        if mode not in ("signature", "llm", "hybrid"):
            # ``"auto"`` was the 0.7-era default — removed in 0.7.9.
            # Reject it eagerly so the failure surfaces at construction
            # time, not lazily at the first ``definition()`` call.
            raise ValueError(
                f"Tool(mode={mode!r}) is invalid.  Accepted values: "
                f"'signature' (default), 'hybrid', 'llm'.  "
                f"The legacy 'auto' value was removed in 0.7.9; pass "
                f"'hybrid' or 'llm' explicitly to opt into LLM-driven "
                f"schema generation."
            )
        self.func = func
        self.name = name or func.__name__
        self.description = description
        self.mode = mode
        #: Seconds this tool may take before the caller gives up on it, or
        #: ``None`` for no bound.  ``Agent(tool_timeout=...)`` supplies a
        #: default to tools that do not set their own.
        self.timeout = check_timeout(timeout, "Tool(timeout=)")
        self.schema_llm = schema_llm
        self.strict = strict
        #: When ``True``, ``func`` returns an ``Envelope`` instead of a
        #: plain Python value.  Engines aware of this hint will preserve
        #: the inner envelope's metadata (tokens / cost / error) when
        #: aggregating results from a turn's tool calls.  The flag is
        #: set automatically by ``_wrap_tool`` for Agents wrapped via
        #: ``agent.as_tool()``.
        self.returns_envelope = returns_envelope
        #: Live reference to the source agent's Memory, set by ``agent.as_tool()``.
        #: Resolved lazily at step execution time via ``from_memory("name")``.
        #: None for plain function tools.
        self.agent_memory = agent_memory
        #: Live reference to the source agent's Store, set by ``agent.as_tool()``.
        #: Used by ``from_agent("name")`` to read the agent's last output.
        #: None for plain function tools.
        self.agent_store = agent_store
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
        timeout: float | None = None,
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
        tool.timeout = check_timeout(timeout, "Tool.from_schema(timeout=)")
        tool.returns_envelope = returns_envelope
        tool.agent_memory = None
        tool.agent_store = None
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

    def _coerce_arguments(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Validate/coerce ``kwargs`` against ``self.func``'s signature."""
        if not self.validate_args:
            return kwargs
        from lazybridge.core.tool_schema import _validate_and_coerce_arguments

        return _validate_and_coerce_arguments(self.func, kwargs)

    async def run(self, **kwargs: Any) -> Any:
        return await self._dispatch(kwargs, self.timeout)

    async def _dispatch(self, kwargs: dict[str, Any], timeout: float | None) -> Any:
        """One invocation under an explicit bound.

        Separate from :meth:`run` so a caller with a bound of its own —
        ``LLMEngine(tool_timeout=...)`` and the two coding-engine tool
        bridges — can apply it through the same machinery instead of
        wrapping ``run()`` in ``wait_for``, which cannot interrupt a
        synchronous tool.  See :func:`run_tool_bounded`.
        """
        kwargs = self._coerce_arguments(kwargs)
        if inspect.iscoroutinefunction(self.func):
            coro = self.func(**kwargs)
            if timeout is None:
                return await coro
            return await _bounded_await(
                coro, timeout, lambda: self._timeout_error(timeout), grace=self.cancel_grace_seconds
            )
        # ``asyncio.get_event_loop`` is deprecated in 3.10+ and errors on
        # 3.13+ when no loop is running.  ``run`` is always called from an
        # already-running coroutine, so ``get_running_loop`` is the right
        # primitive — it also avoids accidentally creating a second loop.
        loop = asyncio.get_running_loop()
        if timeout is None:
            return await loop.run_in_executor(None, lambda: self.func(**kwargs))
        return await self._run_bounded(loop, kwargs, timeout)

    def _timeout_error(self, timeout: float) -> ToolTimeoutError:
        return ToolTimeoutError(
            f"Tool {self.name!r} timed out after {timeout}s",
            tool_name=self.name,
            timeout=timeout,
        )

    async def _run_bounded(self, loop: Any, kwargs: dict[str, Any], timeout: float) -> Any:
        """Run a synchronous tool, giving up on it when its time is out.

        The bound has to be enforced by ABANDONING the call, not by
        cancelling it: a synchronous function cannot be interrupted, and an
        executor future that has already started ignores ``cancel()`` while
        anyone awaiting it keeps waiting.  That is not a theoretical corner
        -- an ``Agent(timeout=8)`` whose tool blocked was still running two
        minutes later, because the agent's own timeout can only cancel at an
        await point and this await never came back.

        So the work goes to a daemon thread we own.  Daemon, because an
        abandoned worker must not hold the interpreter open at exit: a
        scheduled job that finishes its report and then hangs on shutdown is
        the same outage by another name.
        """
        future: asyncio.Future = loop.create_future()
        # Installed before anything can settle, and not only on the timeout
        # path: the caller can also leave through cancellation (an
        # ``Agent(timeout=)`` firing inside a longer ``Tool(timeout=)``), and
        # a worker that raises after that would otherwise leave its exception
        # on an unobserved future and surface much later, out of context, as
        # asyncio's "Future exception was never retrieved".
        future.add_done_callback(lambda f: f.cancelled() or f.exception())

        def _worker() -> None:
            try:
                esito = self.func(**kwargs)
            # BaseException deliberately: whatever the tool raises has to reach
            # the awaiting future.  Catching only Exception would let a
            # SystemExit die with this thread, leaving the caller to wait out
            # its whole timeout for an answer that is never coming.
            except BaseException as exc:
                _settle(future.set_exception, exc)
            else:
                _settle(future.set_result, esito)

        def _settle(setter: Callable, value: Any) -> None:
            def _apply() -> None:
                if not future.done():
                    setter(value)

            try:
                loop.call_soon_threadsafe(_apply)
            except RuntimeError:
                pass  # the loop closed while we were abandoned; nobody is listening

        worker = threading.Thread(target=_worker, daemon=True, name=f"lazybridge-tool-{self.name}")
        worker.start()

        try:
            done, _ = await asyncio.wait({future}, timeout=timeout)
        except BaseException:
            # A caller cancelled from outside abandons the worker just as
            # surely as our own deadline does, and it is the likelier of the
            # two to repeat — an outer ``Agent(timeout=)`` shorter than the
            # tool bound cancels here on EVERY call.  Untracked, that is
            # exactly the leak the warning exists to catch.
            self._abandon(worker)
            raise
        if not done:
            # Deliberately not awaited and not cancelled: the thread runs on
            # until it returns on its own, and the caller stops waiting.
            self._abandon(worker)
            raise self._timeout_error(timeout)
        return future.result()

    def _abandon(self, worker: threading.Thread) -> None:
        """Give up on a worker, and say so out loud once there are too many.

        An abandoned thread is never reclaimed until its call returns, so a
        tool that hangs *every* time — an endpoint that black-holes, a lock
        nobody releases — leaks one thread per retry.  Each caller still gets
        its timely timeout, which is exactly what makes the leak silent: the
        run looks healthy while the process fills up.  Warn rather than
        refuse, because the retry may well be the thing that recovers.
        """
        live = _abandoned.record(worker)
        if live >= self.abandoned_worker_warning_threshold:
            import warnings

            warnings.warn(
                f"{live} abandoned tool workers are still running "
                f"(latest: {self.name!r}).  A tool that times out on every call "
                f"leaks one thread per attempt — give the underlying call its own "
                f"deadline, or stop retrying it.",
                UserWarning,
                stacklevel=3,
            )

    def run_sync(self, **kwargs: Any) -> Any:
        """Blocking tool invocation.

        Handles two cases so that callers never see a stray coroutine:

        * plain sync function → called directly.
        * async function → driven to completion through the shared
          sync↔async bridge (:func:`lazybridge._asyncbridge.run_coroutine_blocking`),
          which handles nest_asyncio, contextvars propagation, and
          loop-closed cleanup identically to ``Agent.__call__``.  Needed
          because :meth:`Agent.as_tool` wraps the agent's ``.run()``
          coroutine into ``Tool.func`` — ``SupervisorEngine`` / REPL
          callers were previously getting ``"<coroutine object _run at
          0x...>"`` instead of the result.
        """
        if self.timeout is not None:
            # Through the async path so the bound is enforced by the same
            # machinery, thread included: a blocking call must not be able
            # to outlast its deadline just because the caller was sync.
            return run_coroutine_blocking(lambda: self._dispatch(kwargs, self.timeout))
        kwargs = self._coerce_arguments(kwargs)
        if not inspect.iscoroutinefunction(self.func):
            return self.func(**kwargs)
        return run_coroutine_blocking(lambda: self.func(**kwargs))

    def __repr__(self) -> str:
        return f"Tool({self.name!r})"

    @classmethod
    def wrap(
        cls,
        obj: Any,
        *,
        name: str | None = None,
        description: str | None = None,
        mode: Literal["signature", "hybrid", "llm"] = "signature",
        schema_llm: Any | None = None,
        strict: bool = _UNSET_BOOL,  # type: ignore[assignment]
        timeout: float | None = None,
    ) -> Tool:
        """Canonical multi-input factory — accepts a callable, an Agent, or an
        existing :class:`Tool`, and returns a properly wrapped ``Tool``.

        **For Python functions** — ``name`` is required so Plan steps, tool
        maps, and LLM calls all share the same stable identifier::

            search = Tool.wrap(search_web, name="search", description="Search the web.")
            researcher = Agent(name="research", engine=LLMEngine(...), tools=[search])

        **For Agents** — the canonical path is ``tools=[agent]`` directly;
        ``Tool.wrap`` is useful when you need a local alias::

            Tool.wrap(researcher, name="deep_research")

        **For existing Tools** — returns the object unchanged (no overrides) or
        clones it with the specified overrides (non-mutating)::

            search_v2 = Tool.wrap(search, name="web_search")

        Parameters
        ----------
        obj:
            A callable, :class:`Agent`, or existing :class:`Tool` to wrap.
        name:
            Required for callables.  Optional alias for agents and Tools.
        description:
            Human-readable description forwarded to the LLM.
        mode:
            Schema generation mode.  ``"signature"`` (default) introspects the
            function signature and docstring deterministically.  Pass
            ``"hybrid"`` (signature + LLM-enriched descriptions) or ``"llm"``
            (full LLM-inferred schema) explicitly when the signature alone
            is insufficient — both require ``schema_llm=`` to be set.
        schema_llm:
            Engine used when ``mode="hybrid"`` or ``mode="llm"``.
        strict:
            Enable JSON Schema strict mode.
        timeout:
            Seconds this tool may take before the caller gives up on it.
            ``None`` leaves the tool unbounded, or keeps the bound a wrapped
            ``Tool`` already carries.

        Notes
        -----
        Module-level :func:`tool` is a thin alias for backwards compatibility
        and is kept indefinitely; new code should prefer ``Tool.wrap``.
        """
        # ── Case 1: already a Tool ──────────────────────────────────────────
        if isinstance(obj, Tool):
            reshapes_schema = (
                name is not None
                or description is not None
                or mode != "signature"
                or schema_llm is not None
                or strict is not _UNSET_BOOL
            )
            if not reshapes_schema:
                if timeout is None:
                    return obj
                # A deadline says nothing about the tool's shape, so copy
                # rather than rebuild: reconstruction would regenerate the
                # schema from the signature and throw away an explicit one
                # set by ``from_schema`` — for an imported tool whose callable
                # is ``lambda **kwargs`` that means showing the model a tool
                # with no parameters.
                clone = copy.copy(obj)
                clone.timeout = check_timeout(timeout, "Tool.wrap(timeout=)")
                return clone
            return cls(
                obj.func,
                name=name if name is not None else obj.name,
                description=description if description is not None else obj.description,
                mode=mode if mode != "signature" else obj.mode,
                schema_llm=schema_llm if schema_llm is not None else obj.schema_llm,
                strict=obj.strict if strict is _UNSET_BOOL else bool(strict),
                returns_envelope=obj.returns_envelope,
                agent_memory=obj.agent_memory,
                agent_store=obj.agent_store,
                timeout=timeout if timeout is not None else obj.timeout,
            )

        # ── Case 2: Agent-like ──────────────────────────────────────────────
        if getattr(obj, "_is_lazy_agent", False):
            # An explicit alias passed here is always accepted.
            # Without an alias, the agent must have _name_explicit=True.
            if name is None and getattr(obj, "_name_explicit", True) is False:
                # Only reject real Agent instances that set _name_explicit=False.
                # Duck-typed agents (MockAgent, custom subclasses) default to True.
                agent_name = getattr(obj, "name", repr(obj))
                raise ValueError(
                    f"Agent used as a tool must have an explicit name=...\n"
                    f"The agent currently has name={agent_name!r} "
                    f"(derived from the model or left as the default).\n\n"
                    f"Set an explicit name:\n"
                    f'    Agent(name="research", engine=LLMEngine(...))\n\n'
                    f"Or pass an alias to the factory:\n"
                    f'    Tool.wrap(agent, name="research")'
                )
            effective_name = name or getattr(obj, "name", None)
            if not effective_name or not str(effective_name).strip():
                raise ValueError(
                    "Agent used as a tool must have an explicit name=...\n"
                    "Example:\n"
                    '    Agent(name="research", engine=LLMEngine(...))'
                )
            if hasattr(obj, "as_tool"):
                agent_tool = obj.as_tool(effective_name, description=description)
            else:
                agent_tool = _agent_as_tool_named(obj, effective_name, description)
            if timeout is not None:
                # ``as_tool`` builds a fresh Tool per call, so this bounds
                # this alias only and not the agent everywhere it is used.
                agent_tool.timeout = check_timeout(timeout, "Tool.wrap(timeout=)")
            return agent_tool

        # ── Case 3: plain callable ──────────────────────────────────────────
        if callable(obj):
            if name is None:
                fn_name = getattr(obj, "__name__", repr(obj))
                raise ValueError(
                    f"Tool.wrap() requires an explicit name=... for callables.\n"
                    f'Example: Tool.wrap({fn_name!r}, name="{fn_name}")'
                )
            strict_val = False if strict is _UNSET_BOOL else bool(strict)  # type: ignore[arg-type]
            return cls(
                obj,
                name=name,
                description=description,
                mode=mode,
                schema_llm=schema_llm,
                strict=strict_val,
                timeout=timeout,
            )

        raise TypeError(f"Tool.wrap() cannot wrap {type(obj).__name__!r}")


def _agent_as_tool_named(agent: Any, name: str, description: str | None) -> Tool:
    """Fallback for duck-typed agents that have no ``as_tool()`` method."""

    async def _run(task: str) -> Any:
        return await agent.run(task)

    desc = description or getattr(agent, "description", None) or f"Run the {name} agent."
    _run.__name__ = name
    _run.__doc__ = desc
    return Tool(_run, name=name, description=desc, mode="signature", returns_envelope=True)


def tool(
    obj: Any,
    *,
    name: str | None = None,
    description: str | None = None,
    mode: Literal["signature", "hybrid", "llm"] = "signature",
    schema_llm: Any | None = None,
    strict: bool = _UNSET_BOOL,  # type: ignore[assignment]
    timeout: float | None = None,
) -> Tool:
    """Backwards-compatibility alias for :meth:`Tool.wrap`.

    New code should call ``Tool.wrap(obj, name=...)`` — it lives on the class
    alongside the explicit constructor, mirroring Python stdlib factories
    like :meth:`dict.fromkeys` and :meth:`datetime.datetime.fromisoformat`.
    The lowercase :func:`tool` is kept indefinitely so existing imports
    (``from lazybridge import tool``) continue to work; no deprecation
    timer is set.
    """
    return Tool.wrap(
        obj,
        name=name,
        description=description,
        mode=mode,
        schema_llm=schema_llm,
        strict=strict,
        timeout=timeout,
    )


def _wrap_tool(obj: Any) -> Tool:
    """Convert a raw callable or Agent into a Tool. Returns Tool unchanged.

    Internal helper — public callers use ``tool()`` (which accepts the same
    inputs and dispatches by type) or pass objects directly to
    ``Agent(tools=[...])`` (which calls this function via
    :func:`build_tool_map`).

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

    async def _run(task: str) -> Envelope[Any]:  # type: ignore[name-defined]
        # ``_run_as_tool`` lets a nested ``conclude`` propagate to the top-level
        # caller; fall back to ``run`` for duck-typed doubles without it.
        runner = getattr(agent, "_run_as_tool", agent.run)
        return await runner(task)

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
    default_timeout: float | None = None,
) -> dict[str, Tool]:
    """Wrap and index tools by name.

    Items in ``tools`` may be:
      - a callable / Agent / :class:`Tool` (wrapped via :func:`_wrap_tool`);
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
        default_timeout: Bound applied to every tool that does not carry one
            of its own — ``Agent(tool_timeout=...)``.  Applied to a copy, so
            a Tool shared between agents does not silently acquire the first
            agent's bound.
    """
    import warnings

    check_timeout(default_timeout, "build_tool_map(default_timeout=)")
    result: dict[str, Tool] = {}
    seen_warnings: set[str] = set()
    for t in tools:
        if getattr(t, "_is_lazy_tool_provider", False):
            expanded = list(t.as_tools())
        else:
            expanded = [_wrap_tool(t)]
        for wrapped in expanded:
            if default_timeout is not None and wrapped.timeout is None:
                wrapped = copy.copy(wrapped)
                wrapped.timeout = default_timeout
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
