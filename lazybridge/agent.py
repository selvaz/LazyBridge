"""Agent — the single public-facing abstraction for LazyBridge."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from typing import Any, cast

from lazybridge.core.types import AgentRuntimeConfig, ObservabilityConfig, ResilienceConfig
from lazybridge.envelope import Envelope
from lazybridge.tools import Tool, build_tool_map

#: Private sentinel — distinguishes "caller omitted the flat kwarg" from
#: "caller explicitly passed its default value".  When a flat kwarg is the
#: sentinel, values from ``resilience=`` / ``observability=`` / ``runtime=``
#: fill in; an explicit value (including the documented default) wins.
_UNSET: Any = object()


#: Documented default + source-config attribute for every runtime knob
#: routed through ``_resolve_runtime_kwargs``.  Each entry is
#: ``(default, source)`` where ``source`` is ``"resilience"`` or
#: ``"observability"``.  Centralising this table avoids the eleven
#: hand-written ``if x is _UNSET:`` blocks we used to have inline in
#: ``Agent.__init__`` and gives us one place to register a new knob.
_RUNTIME_KNOB_DEFAULTS: dict[str, tuple[Any, str]] = {
    # Resilience — wraps retries / timeout / cache / fallback / output
    # retry knobs into a single shareable object across an agent fleet.
    "timeout": (None, "resilience"),
    "max_retries": (3, "resilience"),
    "retry_delay": (1.0, "resilience"),
    "cache": (False, "resilience"),
    "max_output_retries": (2, "resilience"),
    "output_validator": (None, "resilience"),
    "fallback": (None, "resilience"),
    # Observability — session / verbose / identity.
    "verbose": (False, "observability"),
    "session": (None, "observability"),
    "name": (None, "observability"),
    "description": (None, "observability"),
}


def _resolve_runtime_kwargs(
    *,
    runtime: AgentRuntimeConfig | None,
    resilience: ResilienceConfig | None,
    observability: ObservabilityConfig | None,
    flat: dict[str, Any],
) -> dict[str, Any]:
    """Merge precedence ``flat kwarg > config object > default``.

    Each ``flat`` value is either a real user-supplied value or the
    private :data:`_UNSET` sentinel.  When sentinel, the value falls
    through to ``resilience``/``observability`` (taken from the
    explicit kwarg if provided, otherwise from ``runtime``), and from
    there to the documented default in :data:`_RUNTIME_KNOB_DEFAULTS`.

    Pure function — no side effects, no logging, no warnings — so it's
    trivially testable in isolation from ``Agent.__init__``.

    Returns
    -------
    dict[str, Any]
        Same keys as :data:`_RUNTIME_KNOB_DEFAULTS`; values resolved.
    """
    res = resilience if resilience is not None else (runtime.resilience if runtime else None)
    obs = observability if observability is not None else (runtime.observability if runtime else None)
    sources = {"resilience": res, "observability": obs}

    resolved: dict[str, Any] = {}
    for key, (default, src_name) in _RUNTIME_KNOB_DEFAULTS.items():
        v = flat.get(key, _UNSET)
        if v is _UNSET:
            src_obj = sources[src_name]
            v = getattr(src_obj, key, default) if src_obj is not None else default
        resolved[key] = v
    return resolved


class Agent:
    """Universal agent — ``Container(engine, tools, state)``.

    **The framework's invariant.**  Every Agent is the same shape: an
    *engine* that decides how the agent behaves, a *tools* list it can
    invoke, and *state* (memory, session, guard, verify, fallback,
    output, name).  Reading any agent constructor tells you immediately
    which engine is in there — the engine is the lever that swaps
    behaviour, everything else is uniform.

    **Tool-is-Tool contract.**  ``tools=[...]`` accepts plain functions,
    :class:`Tool` instances, other :class:`Agent` instances, and Agents
    whose tools are themselves Agents.  The composition is closed and
    uniform: an Agent of an Agent of a function looks the same at every
    level.  Nested Agents inherit the outer session so observability
    flows through the whole tree.

    **Parallelism is a capability, not a configuration.**  When the
    underlying engine emits multiple tool invocations in a single step
    (e.g. an LLM tool loop turn with N tool calls), LazyBridge executes
    them concurrently via ``asyncio.gather``.  There is no "serial vs
    parallel" mode to pick — if you want N things to happen at once,
    put N things in ``tools=[]`` and the engine will do it for you.

    Construction surface — ``Agent.from_<engine_kind>(...)``
    --------------------------------------------------------

    Every factory builds the right engine, forwards the same uniform
    state kwargs (``memory=``/``session=``/``guard=``/``verify=``/
    ``fallback=``/``output=``/``name=``/etc.) to the unified
    constructor, and returns an :class:`Agent` (or, for the one
    documented asymmetry, a fan-out runner).

    *Core engines* (built into ``lazybridge``)::

        Agent("claude-opus-4-7")               # implicit shortcut for from_model
        Agent.from_model("claude-opus-4-7", tools=[search])
        Agent.from_provider("anthropic", tier="top")
        Agent.from_engine(any_engine_instance) # escape hatch — any engine
        Agent.from_plan(*steps, store=..., resume=True)
        Agent.from_chain(researcher, writer)           # linear Plan sugar
        Agent.from_parallel(a, b, c)                   # → list[Envelope]

    *Extension engines* (live in :mod:`lazybridge.ext`, kept off the
    core ``Agent`` class to respect the core/ext import boundary —
    see ``docs/guides/core-vs-ext.md``).  Two equivalent paths::

        # 1. Escape-hatch through Agent.from_engine
        from lazybridge.ext.hil import SupervisorEngine
        Agent.from_engine(SupervisorEngine(tools=[...], agents=[...]))

        # 2. Module-level factory (symmetric ergonomic shape)
        from lazybridge.ext.hil import supervisor_agent, human_agent
        from lazybridge.ext.planners import (
            orchestrator_agent, blackboard_orchestrator_agent,
        )
        supervisor_agent(tools=[...], agents=[...], session=sess)
        human_agent(timeout=60.0, default="approve")
        orchestrator_agent(agents=[researcher, writer])
        blackboard_orchestrator_agent(agents=[researcher, writer])

    All paths return the same :class:`Agent` shape.  Call ``.run()`` /
    ``__call__()`` / ``stream()`` exactly the same way regardless of
    which engine is inside.

    Examples
    --------

    Tier 1 (2 lines)::

        Agent("claude-opus-4-7")("hello").text()

    Tier 2 (with tools — functions, agents, or mixed)::

        Agent.from_model("claude-opus-4-7", tools=[search, summarizer])("…").text()

    Tier 3 (structured output)::

        Agent.from_model("claude-opus-4-7", output=Summary)("…").payload.title

    Tier 4 (deterministic fan-out / sequential chain)::

        Agent.from_chain(researcher, writer)("AI trends").text()
        Agent.from_parallel(a, b, c)("task")   # → list[Envelope]

    Tier 5 (declared pipeline)::

        Agent.from_plan(
            Step(researcher, name="search"),
            Step(writer,     name="write"),
            store=Store(db="run.sqlite"),
            checkpoint_key="research",
            resume=True,
        )("AI trends April 2026")
    """

    _is_lazy_agent = True  # recognised by wrap_tool()

    def __init__(
        self,
        engine_or_model: str | Any = "claude-opus-4-7",
        tools: list[Tool | Callable | Agent] | None = None,
        output: type = str,
        memory: Any | None = None,
        sources: list[Any] | None = None,
        guard: Any | None = None,
        verify: Agent | None = None,
        max_verify: int = 3,
        name: str | None = _UNSET,
        description: str | None = _UNSET,
        session: Any | None = _UNSET,
        verbose: bool = _UNSET,  # type: ignore[assignment]
        # Convenience: pass provider + model separately
        # Agent("anthropic", model="top") or Agent("anthropic", model="claude-opus-4-7")
        model: str | None = None,
        # Keyword alias for engine_or_model when passing a non-string Engine
        # (e.g. ``Agent(engine=SupervisorEngine(...))``).
        engine: Any | None = None,
        # Provider-native server-side tools (WEB_SEARCH, CODE_EXECUTION, …).
        # Accepted directly on Agent as a shortcut for
        # ``Agent(engine=LLMEngine(..., native_tools=[...]))``.  Ignored when
        # ``engine=`` is a non-LLM engine.
        native_tools: list[Any] | None = None,
        # Structured alternatives to the flat resilience / observability
        # kwargs below.  Precedence: flat kwarg > config object > default.
        # ``Agent(resilience=cfg, timeout=30.0)`` uses the config's
        # retries/cache but overrides its timeout.
        runtime: AgentRuntimeConfig | None = None,
        resilience: ResilienceConfig | None = None,
        observability: ObservabilityConfig | None = None,
        # --- Resilience kwargs (also reachable via resilience=...) ---
        # Optional post-parse validator.  Runs on the structured ``payload``
        # after schema validation; may raise ValueError to force a
        # retry-with-feedback loop (up to ``max_output_retries``).
        output_validator: Callable[[Any], Any] | None = _UNSET,
        max_output_retries: int = _UNSET,  # type: ignore[assignment]
        # Total deadline (seconds) for ``run()``.  Applied at the
        # top-level Agent boundary so a hung tool, runaway tool loop,
        # or slow provider can't block a caller forever.  ``None``
        # disables the deadline (backwards-compatible default).
        timeout: float | None = _UNSET,
        # Convenience shortcuts for provider retry/backoff — forwarded to
        # LLMEngine when the engine is auto-created from a model string.
        # Ignored when ``engine=`` is supplied explicitly (use LLMEngine
        # directly to configure retries on a pre-built engine).
        max_retries: int = _UNSET,  # type: ignore[assignment]
        retry_delay: float = _UNSET,  # type: ignore[assignment]
        # Fallback agent tried when the primary engine returns an error.
        # Useful for provider redundancy: Agent("claude-opus-4-7", fallback=Agent("gpt-4o")).
        # The fallback runs its own full pipeline (tools, memory, guard, etc.) on the
        # same envelope, so it should be configured with compatible output= / tools=.
        fallback: Agent | None = _UNSET,
        # Prompt caching — when True, marks the static prefix (system
        # prompt + tools) as cacheable so providers that support it
        # (Anthropic today; OpenAI/DeepSeek auto-cache; Google uses a
        # different API) serve cache hits at ~10% of input token cost.
        # Forwarded to LLMEngine when the engine is auto-created from a
        # model string.  Ignored when ``engine=`` is supplied explicitly
        # (configure ``LLMEngine(cache=...)`` directly in that case).
        # Pass a ``CacheConfig(ttl="1h")`` for the longer Anthropic TTL.
        cache: bool | Any = _UNSET,
    ) -> None:
        # Merge config objects into flat kwargs via the centralised
        # precedence helper.  See ``_resolve_runtime_kwargs`` for the
        # contract; the table-driven approach keeps this block
        # constant-size as we add new shareable knobs.
        _resolved = _resolve_runtime_kwargs(
            runtime=runtime,
            resilience=resilience,
            observability=observability,
            flat={
                "timeout": timeout,
                "max_retries": max_retries,
                "retry_delay": retry_delay,
                "cache": cache,
                "max_output_retries": max_output_retries,
                "output_validator": output_validator,
                "fallback": fallback,
                "verbose": verbose,
                "session": session,
                "name": name,
                "description": description,
            },
        )
        timeout = _resolved["timeout"]
        max_retries = _resolved["max_retries"]
        retry_delay = _resolved["retry_delay"]
        cache = _resolved["cache"]
        max_output_retries = _resolved["max_output_retries"]
        output_validator = _resolved["output_validator"]
        fallback = _resolved["fallback"]
        verbose = _resolved["verbose"]
        session = _resolved["session"]
        name = _resolved["name"]
        description = _resolved["description"]
        from lazybridge.engines.llm import LLMEngine

        if engine is not None:
            self.engine: Any = engine
        elif isinstance(engine_or_model, str):
            model_str = model or engine_or_model
            self.engine = LLMEngine(
                model_str,
                native_tools=native_tools,
                max_retries=max_retries,
                retry_delay=retry_delay,
                cache=cache,
            )
        else:
            self.engine = engine_or_model

        # If the caller passed native_tools but also supplied a pre-built
        # engine, push the list onto the engine if it has the attribute.
        # This lets ``Agent(engine=LLMEngine("claude"), native_tools=[...])``
        # work the same as ``Agent("claude", native_tools=[...])``.
        if native_tools and hasattr(self.engine, "native_tools"):
            from lazybridge.core.types import NativeTool

            resolved = [NativeTool(t) if isinstance(t, str) else t for t in native_tools]
            # Merge without dup — preserve order of existing + append new.
            existing = list(getattr(self.engine, "native_tools", []) or [])
            for t in resolved:
                if t not in existing:
                    existing.append(t)
            self.engine.native_tools = existing

        self._tools_raw = list(tools or [])
        self._tool_map: dict[str, Tool] = build_tool_map(self._tools_raw)
        self.output = output
        self.output_validator = output_validator
        self.max_output_retries = max_output_retries
        self.timeout = timeout
        self.memory = memory
        self.sources = list(sources or [])
        if max_verify < 1:
            raise ValueError(f"max_verify must be >= 1, got {max_verify!r}")
        if max_output_retries < 0:
            raise ValueError(f"max_output_retries must be >= 0, got {max_output_retries!r}")
        self.guard = guard
        self.verify = verify
        self.max_verify = max_verify
        self.fallback = fallback
        self.name: str = str(name or getattr(self.engine, "model", None) or "agent")
        self.description = description

        # Private per-agent console exporter when verbose= is set without
        # an explicit Session. Attached when we bind to a session below.
        self._verbose = verbose

        # Bind to session — create an implicit private Session if verbose=
        # is requested without one, so events print to stdout out of the box.
        if session is None and verbose:
            from lazybridge.session import Session

            session = Session(console=True)
        self.session = session

        self.engine._agent_name = self.name

        # Register with session graph so it's visible in session.graph.to_json()
        _safe_register_agent(self.session, self)

        # Propagate session to nested Agents passed as tools (they become
        # part of the same observability surface — events from B called
        # via A flow into A's EventLog). Agents that already have a
        # session keep it. This is the fix for the "as_tool" observability
        # paradox: without it, calling B through A's tool loop recorded
        # nothing anywhere.
        if self.session is not None:
            for raw in self._tools_raw:
                # Duck-typed: any Agent-compatible object (real Agent,
                # MockAgent from lazybridge.testing, user subclasses) with
                # the ``_is_lazy_agent`` marker gets the outer session
                # propagated when it has none of its own.
                if getattr(raw, "_is_lazy_agent", False) and getattr(raw, "session", None) is None:
                    # Duck-typed Agent (real Agent, MockAgent, user subclass).
                    # The marker check above narrows the type at runtime; cast
                    # so mypy accepts the attribute write + helper calls.
                    agent_raw = cast("Agent", raw)
                    agent_raw.session = self.session
                    _safe_register_agent(self.session, agent_raw)
                    _safe_register_tool_edge(self.session, self, agent_raw, label="as_tool")
            # ``fallback=`` and ``verify=`` Agents inherit the same
            # session + graph-registration the tools list gets, so any
            # events they produce (errors handled by the fallback, judge
            # verdicts from verify) flow into the outer EventLog.  Edge
            # labels distinguish provenance.
            for related, label in (
                (self.fallback, "fallback"),
                (self.verify, "verify"),
            ):
                if (
                    related is not None
                    and getattr(related, "_is_lazy_agent", False)
                    and getattr(related, "session", None) is None
                ):
                    related.session = self.session
                    _safe_register_agent(self.session, related)
                    _safe_register_tool_edge(self.session, self, related, label=label)

        # PlanCompiler runs at construction time
        if hasattr(self.engine, "_validate"):
            self.engine._validate(self._tool_map)

    # ------------------------------------------------------------------
    # Core async API
    # ------------------------------------------------------------------

    async def run(
        self,
        task: str | Envelope,
        *,
        images: list[Any] | None = None,
        audio: Any | None = None,
    ) -> Envelope:
        # ``getattr`` with a default keeps this backwards-compatible for
        # Agents constructed via ``Agent.__new__`` (test helpers, custom
        # subclasses) that haven't set ``self.timeout``.
        timeout = getattr(self, "timeout", None)
        if timeout is None:
            return await self._run_body(task, images=images, audio=audio)
        try:
            return await asyncio.wait_for(self._run_body(task, images=images, audio=audio), timeout=timeout)
        except TimeoutError:
            # Emit a synthetic AGENT_FINISH so callers reading
            # ``session.events.query()`` see a complete trace even when
            # the engine's own AGENT_FINISH was skipped by the
            # ``CancelledError`` propagation (BaseException, not caught
            # by the engine's ``except Exception``).
            session = getattr(self, "session", None)
            if session is not None and hasattr(session, "emit"):
                from lazybridge.session import EventType

                try:
                    session.emit(
                        EventType.AGENT_FINISH,
                        {
                            "agent_name": self.name,
                            "error": f"Agent.run() exceeded timeout={timeout}s",
                            "cancelled": True,
                        },
                    )
                except Exception:
                    # Session is best-effort here; never let observability
                    # mask the underlying timeout.
                    pass
            return Envelope.error_envelope(TimeoutError(f"Agent.run() exceeded timeout={timeout}s"))

    async def _run_body(
        self,
        task: str | Envelope,
        *,
        images: list[Any] | None = None,
        audio: Any | None = None,
    ) -> Envelope:
        env = self._to_envelope(task, images=images, audio=audio)
        env = self._inject_sources(env)

        if self.guard:
            action = await self.guard.acheck_input(env.task or "")
            if not action.allowed:
                return Envelope.error_envelope(ValueError(action.message or "Blocked by guard"))
            if action.modified_text is not None:
                env = Envelope(
                    task=action.modified_text,
                    context=env.context,
                    images=env.images,
                    audio=env.audio,
                    payload=action.modified_text,
                )

        if self.verify:
            from lazybridge._verify import verify_with_retry

            result = await verify_with_retry(self, env, self.verify, max_verify=self.max_verify)
        else:
            result = await self._run_engine(env)

        # Provider fallback: if primary failed and a fallback agent is
        # configured, run the fallback's full pipeline on the same
        # already-processed input — but thread the primary's failure
        # mode into the fallback's ``context`` so the fallback can
        # adapt (e.g. switch tactics on rate-limit vs schema error).
        if result.error is not None and self.fallback is not None:
            err = result.error
            note = f"Previous attempt failed with {err.type}: {err.message}"
            merged_context = f"{env.context}\n\n{note}" if env.context else note
            fallback_env: Envelope[Any] = Envelope(
                task=env.task,
                context=merged_context,
                images=env.images,
                audio=env.audio,
                payload=env.payload,
            )
            result = await self.fallback.run(fallback_env)

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
        result = await self.engine.run(
            env,
            tools=list(self._tool_map.values()),
            output_type=self.output,
            memory=self.memory,
            session=self.session,
        )
        # Structured output retry loop.  When ``output=`` declares a
        # concrete type (``SomeModel``, ``list[SomeModel]``, …) and the
        # engine returned a plain string instead of the expected
        # payload — or ``output_validator`` raised — re-attempt the
        # call with the validation error fed back as context. Up to
        # ``max_output_retries`` times; the final attempt is returned
        # as-is.  Function-style output (``output=str``) skips this
        # entirely.
        if self.output is not str and result.ok:
            result = await self._validate_and_retry(env, result)
        return result

    async def _validate_and_retry(self, original_env: Envelope, first: Envelope) -> Envelope:
        from lazybridge.core.structured import validate_payload_against_output_type

        # Initialise feedback before the loop so any code path that
        # reaches attempt ≥ 1 always has a defined value.
        feedback: str = ""
        current = first

        for attempt in range(self.max_output_retries + 1):
            if attempt > 0:
                # Pass memory=None for correction retries: these are a
                # private feedback loop and must not be stored as real
                # conversation turns.  Only the final successful result
                # (returned below) contributes to memory history.
                current = await self.engine.run(
                    _feedback_env(original_env, feedback),
                    tools=list(self._tool_map.values()),
                    output_type=self.output,
                    memory=None,
                    session=self.session,
                )
                if not current.ok:
                    return current

            # Schema validation and custom validator are separated so each
            # failure produces a precise, actionable feedback message —
            # a custom-validator rejection isn't blamed on a schema
            # mismatch.

            # Step 1 — schema validation
            try:
                validated = validate_payload_against_output_type(current.payload, self.output)
            except Exception as exc:
                if attempt == self.max_output_retries:
                    return current
                feedback = (
                    f"The previous response did not satisfy the required "
                    f"output schema ({_describe_output_type(self.output)}). "
                    f"Schema validation error: {exc}. Please respond again "
                    f"with a valid instance."
                )
                continue

            # Step 2 — optional custom post-parse validator
            if self.output_validator is not None:
                try:
                    maybe = self.output_validator(validated)
                    validated = maybe if maybe is not None else validated
                except Exception as exc:
                    if attempt == self.max_output_retries:
                        return current
                    feedback = (
                        f"The previous response passed schema validation but "
                        f"failed domain validation for "
                        f"{_describe_output_type(self.output)}. "
                        f"Validator rejected with: {exc}. Please respond again."
                    )
                    continue

            # Swap the validated payload back in; keep metadata/error unchanged.
            return current.model_copy(update={"payload": validated})

        return current  # type: ignore[return-value]

    async def stream(
        self,
        task: str | Envelope,
        *,
        images: list[Any] | None = None,
        audio: Any | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream LLM tokens across the full tool-calling loop.

        **Guard enforcement.** ``self.guard`` is checked via
        ``acheck_input`` before the first token is emitted.  A blocked
        input raises :class:`ValueError` immediately; a modified input
        (``GuardAction.modify``) replaces the task in the envelope sent
        to the provider.  This is identical to the guard contract in
        :meth:`run` — streaming mode does not bypass the guard.

        Honours ``self.timeout`` between chunks so a stalled provider
        can't hang the caller.  Each ``__anext__`` is wrapped in
        ``asyncio.wait_for`` (per-chunk, not whole-stream) so short
        chunks don't consume the deadline budget.

        Multimodal: pass ``images=`` / ``audio=`` to attach blocks to
        the streamed turn — same coercion semantics as :meth:`run`.
        """
        env = self._to_envelope(task, images=images, audio=audio)
        env = self._inject_sources(env)
        timeout = getattr(self, "timeout", None)

        # Apply input guard before the first token is emitted.  A blocked
        # task must never reach the provider even in streaming mode.
        if self.guard:
            action = await self.guard.acheck_input(env.task or "")
            if not action.allowed:
                raise ValueError(action.message or "Blocked by guard")
            if action.modified_text is not None:
                env = env.model_copy(update={"task": action.modified_text, "payload": action.modified_text})

        gen = self.engine.stream(
            env,
            tools=list(self._tool_map.values()),
            output_type=self.output,
            memory=self.memory,
            session=self.session,
        ).__aiter__()
        try:
            while True:
                try:
                    if timeout is None:
                        chunk = await gen.__anext__()
                    else:
                        chunk = await asyncio.wait_for(gen.__anext__(), timeout=timeout)
                except StopAsyncIteration:
                    return
                yield chunk
        finally:
            aclose = getattr(gen, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------

    def __call__(
        self,
        task: str | Envelope,
        *,
        images: list[Any] | None = None,
        audio: Any | None = None,
    ) -> Envelope:
        # Detect whether we're already inside a running event loop.
        # ``asyncio.get_running_loop`` is the forward-compatible way — it
        # raises ``RuntimeError`` when there is no current loop, unlike
        # the deprecated ``get_event_loop`` which implicitly creates one.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No loop running — safe to use asyncio.run directly.
            return asyncio.run(self.run(task, images=images, audio=audio))

        # Running inside a loop (Jupyter, FastAPI, asyncio tests, …).
        # Spin up a fresh loop on a worker thread, copying the caller's
        # contextvars context so OTel spans / request IDs / structured-
        # logging context flow into the agent's loop instead of starting
        # empty (which would silently break observability for sync
        # callers running inside an async framework).
        return _run_coro_with_context(self.run(task, images=images, audio=audio))

    # ------------------------------------------------------------------
    # Tool exposure
    # ------------------------------------------------------------------

    def as_tool(
        self,
        name: str | None = None,
        description: str | None = None,
        *,
        verify: Agent | Callable[[str], Any] | None = None,
        max_verify: int = 3,
    ) -> Tool:
        """Return a Tool that wraps this agent.

        The tool schema is: ``(task: str) -> str``.
        Use this to plug one agent into another agent's tools list.

            researcher = Agent("claude-opus-4-7", tools=[search])
            orchestrator = Agent("claude-opus-4-7",
                                 tools=[researcher.as_tool("researcher",
                                                           "Search and summarise papers")])

        Verify (Option B) — wrap the tool in a judge/retry loop so every
        call through this tool is vetted by a judge before returning::

            judge = Agent(engine=LLMEngine(
                "claude-opus-4-7",
                system="Reply 'approved' or 'rejected: <reason>'.",
            ))
            synth = Agent(...)
            orchestrator = Agent(
                ...,
                tools=[synth.as_tool("synthesize", verify=judge, max_verify=2)],
            )

        ``verify`` can be either an :class:`Agent` (its ``run`` method is
        called with the output) or a plain callable taking the output text
        and returning a verdict string / bool. On rejection, the judge's
        feedback is injected into the next attempt's task.
        """
        if max_verify < 1:
            raise ValueError(f"max_verify must be >= 1, got {max_verify!r}")
        agent = self
        effective_name = name or self.name
        effective_desc = description or self.description or f"Run the {effective_name} agent."

        if verify is None:

            async def _run(task: str) -> Envelope:
                return await agent.run(task)
        else:

            async def _run(task: str) -> Envelope:  # type: ignore[misc]
                from lazybridge._verify import verify_with_retry
                from lazybridge.envelope import Envelope as _Env

                env = _Env.from_task(str(task))
                return await verify_with_retry(
                    agent,
                    env,
                    verify,
                    max_verify=max_verify,
                )

        _run.__name__ = effective_name
        _run.__doc__ = effective_desc

        return Tool(
            _run,
            name=effective_name,
            description=effective_desc,
            mode="signature",
            returns_envelope=True,
        )

    def definition(self) -> Any:
        """ToolDefinition for this agent — used when passed in tools=[] of another agent."""
        return self.as_tool().definition()

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_model(cls, model: str, **kwargs: Any) -> Agent:
        """Construct an Agent backed by an LLMEngine for ``model``.

        Explicit counterpart to ``Agent("claude-opus-4-7")``.  Use this
        when you want the model-path to be unambiguous at the call site
        (code review, static analysis, teaching material)::

            Agent.from_model("claude-opus-4-7", tools=[search])
        """
        return cls(model, **kwargs)

    @classmethod
    def from_engine(cls, engine: Any, **kwargs: Any) -> Agent:
        """Construct an Agent from an already-built Engine instance.

        Explicit counterpart to ``Agent(engine=some_engine)``::

            Agent.from_engine(Plan(Step(researcher), Step(writer)))
            Agent.from_engine(SupervisorEngine(tools=[t], agents=[a]))
        """
        return cls(engine=engine, **kwargs)

    @classmethod
    def from_provider(
        cls,
        provider: str,
        *,
        tier: str = "medium",
        **kwargs: Any,
    ) -> Agent:
        """Construct an Agent for ``provider`` using its tier alias for model selection.

        Tiers (``super_cheap`` / ``cheap`` / ``medium`` / ``expensive`` /
        ``top``) resolve to each provider's current lineup, so preview and
        date-pinned model names stay in one place::

            Agent.from_provider("anthropic", tier="top")
            Agent.from_provider("openai", tier="cheap", tools=[search])
        """
        from lazybridge.engines.llm import LLMEngine

        # Pass both the tier (as the provider-facing model string, which
        # the BaseProvider resolves via its tier map) AND the explicit
        # provider name (so _infer_provider doesn't fall back to the
        # default when the tier alone isn't a recognised model).
        return cls(engine=LLMEngine(tier, provider=provider), **kwargs)

    @classmethod
    def chain(cls, *agents: Agent, **kwargs: Any) -> Agent:
        """Run agents sequentially: output of each becomes input to the next."""
        from lazybridge.engines.plan import Plan, Step

        steps = [Step(target=a, name=a.name) for a in agents]
        plan = Plan(*steps)
        name = kwargs.pop("name", "chain")
        # Don't auto-wrap agents as tools — ``Plan._exec_step`` dispatches
        # Agent targets via ``target.run()`` directly, so wrapping them
        # would just waste schema-compilation on every chain call.
        # Caller-supplied tools= in kwargs still pass through unchanged.
        return cls(engine=plan, name=name, **kwargs)

    @classmethod
    def parallel(
        cls,
        *agents: Agent,
        concurrency_limit: int | None = None,
        step_timeout: float | None = None,
        **kwargs: Any,
    ) -> _ParallelAgent:
        """Deterministic fan-out: run ``agents`` concurrently on the same task.

        Returns ``list[Envelope]`` — one entry per input agent, preserving
        order.  No LLM orchestrator mediates the call; this is just sugar
        for ``asyncio.gather(*[a.run(task) for a in agents])`` with an
        optional semaphore (``concurrency_limit``) and per-agent timeout
        (``step_timeout``).

        Use this when you **know** you want N things to happen in
        parallel.  If you want the LLM to decide whether to call agents
        in parallel (and which, and how), don't use this — pass them as
        ``tools=[...]`` on a regular ``Agent`` instead; the engine emits
        parallel tool calls automatically when the model requests them.
        """
        return _ParallelAgent(
            agents=list(agents),
            concurrency_limit=concurrency_limit,
            step_timeout=step_timeout,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Unified ``from_<engine_kind>`` factories
    # ------------------------------------------------------------------
    #
    # Every Agent is ``Container(engine, tools, state)``.  The engine
    # decides HOW the agent behaves; everything else (memory, session,
    # guard, verify, fallback, output, name) is uniform across every
    # engine.  These factories are sugar that build the right engine
    # and forward shared kwargs through to the unified Agent
    # constructor.  Reading any ``Agent.from_X(...)`` call site tells
    # you immediately which engine is in there.

    @classmethod
    def from_plan(
        cls,
        *steps: Any,
        max_iterations: int = 100,
        store: Any | None = None,
        checkpoint_key: str | None = None,
        resume: bool = False,
        on_concurrent: str = "fail",
        **kwargs: Any,
    ) -> Agent:
        """Construct an Agent backed by a declarative :class:`Plan`.

        Explicit counterpart to ``Agent(engine=Plan(*steps, ...))``::

            Agent.from_plan(
                Step(researcher, name="search", writes="hits"),
                Step(ranker,     name="rank",   context=from_prev),
                Step(writer,     name="write",  context=from_step("rank")),
                store=Store(db="run.sqlite"),
                checkpoint_key="research",
                resume=True,
            )

        All :class:`Plan` kwargs are accepted directly; remaining
        ``**kwargs`` (``memory=`` / ``session=`` / ``output=`` /
        ``verify=`` / ``fallback=`` / ``guard=`` / ``name=`` / etc.)
        forward to the unified Agent constructor.
        """
        from lazybridge.engines.plan import Plan

        plan = Plan(
            *steps,
            max_iterations=max_iterations,
            store=store,
            checkpoint_key=checkpoint_key,
            resume=resume,
            on_concurrent=on_concurrent,  # type: ignore[arg-type]
        )
        return cls(engine=plan, **kwargs)

    @classmethod
    def from_chain(cls, *agents: Agent, **kwargs: Any) -> Agent:
        """Construct an Agent that runs ``agents`` sequentially (linear pipeline).

        Each agent's output becomes the next agent's input.  Internally
        wraps a :class:`Plan` of one ``Step`` per agent — the canonical
        narrative is "linear Plan", just spelled in two characters less.

        Equivalent to :meth:`Agent.chain`; the ``from_`` form is the
        documented canonical surface so reading the call site tells you
        immediately which engine is in there.
        """
        return cls.chain(*agents, **kwargs)

    @classmethod
    def from_parallel(
        cls,
        *agents: Agent,
        concurrency_limit: int | None = None,
        step_timeout: float | None = None,
        **kwargs: Any,
    ) -> _ParallelAgent:
        """Construct a deterministic fan-out runner over ``agents``.

        Equivalent to :meth:`Agent.parallel`.  **Note:** this is the one
        ``from_*`` factory that does NOT return a single ``Agent``; it
        returns a :class:`_ParallelAgent` whose ``__call__`` returns
        ``list[Envelope]`` (one per input agent, in order).  The
        asymmetry is intentional: this is **scripted** fan-out, not an
        orchestrated agent — there is no single "result" to wrap into
        one envelope.  Use ``Agent(tools=[a, b, c])`` if you want a
        proper Agent that the engine orchestrates.
        """
        return cls.parallel(
            *agents,
            concurrency_limit=concurrency_limit,
            step_timeout=step_timeout,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Note on ext-engine factories
    # ------------------------------------------------------------------
    #
    # ``Agent.from_<kind>`` factories for ext engines (Supervisor,
    # Human, dynamic planners) intentionally do NOT live on this class —
    # the core-vs-ext boundary (see ``docs/guides/core-vs-ext.md``)
    # forbids ``lazybridge/`` core from importing ``lazybridge.ext.*``,
    # even via lazy/local imports.  Use either of:
    #
    # 1. The escape-hatch :meth:`from_engine` with the ext engine instance::
    #
    #        from lazybridge.ext.hil import SupervisorEngine
    #        Agent.from_engine(SupervisorEngine(tools=[...], agents=[...]))
    #
    # 2. The module-level ergonomic factories shipped in each ext package::
    #
    #        from lazybridge.ext.hil import supervisor_agent, human_agent
    #        supervisor_agent(tools=[...], agents=[...])
    #        human_agent(timeout=60.0, default="approve")
    #
    #        from lazybridge.ext.planners import make_planner, make_blackboard_planner
    #        make_planner(agents=[researcher, writer])
    #        make_blackboard_planner(agents=[researcher, writer])
    #
    # The unified narrative — ``Agent = Container(engine, tools, state)``
    # — still holds: every path returns an Agent, every Agent has an
    # engine, and the engine is always the swappable behaviour.

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_envelope(
        task: str | Envelope,
        *,
        images: list[Any] | None = None,
        audio: Any | None = None,
    ) -> Envelope:
        """Coerce ``task`` into an :class:`Envelope`, attaching multimodal blocks.

        Conflict policy: when ``task`` is already an :class:`Envelope`
        with attachments AND ``images=`` / ``audio=`` are also passed,
        :class:`ValueError` is raised — explicit > implicit.  Pass
        attachments through one channel only.
        """
        from lazybridge.core.types import _coerce_audio, _coerce_image

        coerced_images = [_coerce_image(x) for x in images] if images else None
        coerced_audio = _coerce_audio(audio) if audio is not None else None

        if isinstance(task, Envelope):
            if coerced_images is not None and task.images:
                raise ValueError(
                    "images= passed to .run()/.__call__() but the supplied "
                    "Envelope already carries images.  Pass attachments via "
                    "exactly one channel."
                )
            if coerced_audio is not None and task.audio is not None:
                raise ValueError(
                    "audio= passed to .run()/.__call__() but the supplied "
                    "Envelope already carries audio.  Pass attachments via "
                    "exactly one channel."
                )
            if coerced_images is not None or coerced_audio is not None:
                return task.model_copy(
                    update={
                        "images": coerced_images if coerced_images is not None else task.images,
                        "audio": coerced_audio if coerced_audio is not None else task.audio,
                    }
                )
            return task

        env = Envelope.from_task(str(task))
        if coerced_images is not None or coerced_audio is not None:
            return env.model_copy(update={"images": coerced_images, "audio": coerced_audio})
        return env

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
        return Envelope(
            task=env.task,
            context=merged or env.context,
            images=env.images,
            audio=env.audio,
            payload=env.payload,
        )


def _safe_register_agent(session: Any, agent: Agent) -> None:
    """Register ``agent`` on ``session.graph`` if possible, warning on failure.

    A buggy ``register_agent`` on a custom Session subclass surfaces as
    a ``UserWarning`` rather than silently dropping graph entries.
    """
    if session is None or not hasattr(session, "register_agent"):
        return
    try:
        session.register_agent(agent)
    except Exception as exc:
        import warnings

        warnings.warn(
            f"session.register_agent({getattr(agent, 'name', '?')!r}) raised {type(exc).__name__}: {exc}",
            stacklevel=2,
        )


def _safe_register_tool_edge(
    session: Any,
    outer: Agent,
    inner: Agent,
    *,
    label: str,
) -> None:
    """Register a graph edge between two agents, warning on failure."""
    if session is None or not hasattr(session, "register_tool_edge"):
        return
    try:
        session.register_tool_edge(outer, inner, label=label)
    except Exception as exc:
        import warnings

        warnings.warn(
            f"session.register_tool_edge({getattr(outer, 'name', '?')!r}→"
            f"{getattr(inner, 'name', '?')!r}) raised "
            f"{type(exc).__name__}: {exc}",
            stacklevel=2,
        )


def _feedback_env(original: Envelope, feedback: str) -> Envelope:
    """Build a retry envelope that keeps the pristine user task but
    appends the validation failure reason as context so the model
    understands what went wrong.
    """
    merged = f"{original.context}\n\n{feedback}" if original.context else feedback
    return Envelope(task=original.task, context=merged, payload=original.payload)


def _describe_output_type(output: Any) -> str:
    """Human-readable rendering of an ``output=`` annotation for feedback."""
    name = getattr(output, "__name__", None)
    if name:
        return name
    return str(output).replace("typing.", "")


def _run_coro_with_context(coro: Any) -> Any:
    """Run ``coro`` on a fresh event loop in a worker thread, with the
    caller's :mod:`contextvars` context propagated into it.

    Without ``ctx.run`` here, ``asyncio.run`` on a worker thread starts
    with an empty context, so contextvars set by the outer framework
    (OpenTelemetry spans, request IDs, structured-logging context) are
    invisible to the agent.  Living at the ``__call__`` boundary means
    every Engine type benefits without individually re-implementing
    the bridge.
    """
    import concurrent.futures
    import contextvars

    ctx = contextvars.copy_context()

    def _run() -> Any:
        return ctx.run(asyncio.run, coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run).result()


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
    """Deterministic fan-out over N agents — the shape behind :meth:`Agent.parallel`.

    This is a **pre-scripted** parallel runner, not a parallelism
    paradigm.  Every input agent receives the same task; their per-run
    Envelopes are returned as a list in input order.  No orchestrator
    LLM is involved.

    Prefer :class:`Agent` with ``tools=[...]`` when you want the engine
    (LLM, Supervisor, Plan) to decide dynamically which tools to invoke
    and when — parallel execution is automatic on that path.
    """

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

    async def run(self, task: str | Envelope) -> list[Envelope]:
        env = Agent._to_envelope(task) if isinstance(task, str) else task
        sem = asyncio.Semaphore(self.concurrency_limit) if self.concurrency_limit else None

        async def _run_one(agent: Agent) -> Envelope:
            async def _coro() -> Envelope:
                if self.step_timeout:
                    return await asyncio.wait_for(agent.run(env), timeout=self.step_timeout)
                return await agent.run(env)

            if sem:
                async with sem:
                    return await _coro()
            return await _coro()

        results = await asyncio.gather(*[_run_one(a) for a in self.agents], return_exceptions=True)
        return [
            r
            if isinstance(r, Envelope)
            else Envelope.error_envelope(r if isinstance(r, Exception) else RuntimeError(str(r)))
            for r in results
        ]

    def __call__(self, task: str | Envelope) -> list[Envelope]:
        # Mirror ``Agent.__call__`` — ``get_running_loop`` is the only
        # forward-compatible detection (``get_event_loop`` is deprecated
        # under 3.12 and errors under 3.14+ when no loop is running).
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(task))
        # Propagate caller contextvars into the worker loop.
        return _run_coro_with_context(self.run(task))

    # ------------------------------------------------------------------
    # Tool-is-Tool — fold the list[Envelope] into a single Envelope so
    # this fan-out runner can be passed as a tool to another agent
    # (like every other ``Agent`` / agent-like in LazyBridge).
    # ------------------------------------------------------------------

    def as_tool(
        self,
        name: str | None = None,
        description: str | None = None,
    ) -> Tool:
        """Expose the fan-out runner as a single :class:`Tool`.

        The N branches' ``Envelope`` results are folded into ONE
        :class:`Envelope` whose ``payload`` is a labelled-text join —
        same shape as :class:`Plan`'s ``from_parallel_all`` aggregator,
        so the outer agent / model reads the output uniformly.  Cost
        roll-up is transitive (every branch's ``input_tokens`` /
        ``output_tokens`` / ``cost_usd`` is summed into the wrapper's
        ``nested_*`` fields).  The first non-``None`` branch error
        propagates as the wrapper's ``error`` so downstream can
        short-circuit.

        Without this method, ``wrap_tool(parallel_runner)`` falls
        through the inline shim in :func:`lazybridge.tools._agent_as_tool`
        which assumes ``run()`` returns an ``Envelope``; the
        ``list[Envelope]`` then leaks into the LLM's tool result block
        as an opaque ``str(list)`` that the model can't parse.
        """
        from lazybridge.envelope import EnvelopeMetadata
        from lazybridge.tools import Tool

        actual_name = name or self.name or "parallel"
        actual_desc = (
            description or self.description or (f"Run {len(self.agents)} agents in parallel and join their outputs.")
        )

        async def _run(task: str) -> Envelope:
            results: list[Envelope] = await self.run(task)
            # Labelled-text join — same shape as
            # ``Plan._aggregate_parallel_band``.  Reading the joined
            # text gives the consumer a structured, model-friendly
            # view of every branch's contribution.
            sections = [
                f"[{a.name}]\n{e.text() if not e.error else f'(error) {e.error.message}'}"
                for a, e in zip(self.agents, results)
            ]
            joined = "\n\n".join(sections)
            # Transitive cost rollup — the outer envelope's
            # ``nested_*`` reports the total spend of every branch
            # (their direct + their own nested_*) so an N-deep tree
            # of parallel-of-parallel composes cleanly.
            nested_in = sum(e.metadata.input_tokens + e.metadata.nested_input_tokens for e in results)
            nested_out = sum(e.metadata.output_tokens + e.metadata.nested_output_tokens for e in results)
            nested_cost = sum(e.metadata.cost_usd + e.metadata.nested_cost_usd for e in results)
            # First error wins so callers reading ``.error`` can detect
            # branch failure without scanning the whole list.
            first_error = next((e.error for e in results if e.error), None)
            return Envelope(
                task=task,
                payload=joined,
                metadata=EnvelopeMetadata(
                    nested_input_tokens=nested_in,
                    nested_output_tokens=nested_out,
                    nested_cost_usd=nested_cost,
                ),
                error=first_error,
            )

        _run.__name__ = actual_name
        _run.__doc__ = actual_desc

        return Tool(
            _run,
            name=actual_name,
            description=actual_desc,
            mode="signature",
            returns_envelope=True,
        )
