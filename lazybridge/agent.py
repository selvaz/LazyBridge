"""Agent — the single public-facing abstraction for LazyBridge v1.0."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from typing import Any

from lazybridge.envelope import Envelope
from lazybridge.tools import Tool, build_tool_map


class Agent:
    """Universal agent — delegates execution to a swappable Engine.

    ``Agent`` is the only public abstraction the user interacts with.  The
    engine (LLM, Human, Supervisor, Plan) is swappable; the call surface
    is identical.

    **Tool-is-Tool contract.** ``tools=[...]`` accepts plain functions,
    ``Tool`` instances, other ``Agent`` instances, and Agents whose tools
    are themselves Agents.  The composition is closed and uniform: an
    Agent of an Agent of a function looks the same at every level.
    LazyBridge wraps each entry via :func:`wrap_tool` at construction
    time; nested Agents inherit the outer session so observability flows
    through the whole tree.

    **Parallelism is a capability, not a configuration.**  When the
    underlying engine emits multiple tool invocations in a single step
    (e.g. an LLM tool loop turn with N tool calls), LazyBridge executes
    them concurrently via ``asyncio.gather``.  There is no "serial vs
    parallel" mode to pick — if you want N things to happen at once, put
    N things in ``tools=[]`` and the engine will do it for you.

    Tier 1 (2 lines)::

        Agent("claude-opus-4-7")("hello").text()

    Tier 2 (with tools — functions, agents, or mixed)::

        Agent("claude-opus-4-7", tools=[search, summarizer])("…").text()

    Tier 3 (structured output)::

        Agent("claude-opus-4-7", output=Summary)("…").payload.title

    Tier 4 (deterministic fan-out / sequential chain — sugar helpers)::

        Agent.chain(researcher, writer)("AI trends").text()
        Agent.parallel(a, b, c)("task")   # → list[Envelope], no LLM orchestrator

    Tier 5/6 (Plan or SupervisorEngine — same ``tools=`` surface)::

        Agent(engine=Plan(...), tools=[...], output=Report, session=session)
        Agent(engine=SupervisorEngine(...), tools=[...], session=session)
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
        name: str | None = None,
        description: str | None = None,
        session: Any | None = None,
        verbose: bool = False,
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
        # Optional post-parse validator.  Runs on the structured ``payload``
        # after schema validation; may raise ValueError to force a
        # retry-with-feedback loop (up to ``max_output_retries``).
        output_validator: Callable[[Any], Any] | None = None,
        max_output_retries: int = 2,
        # Total deadline (seconds) for ``run()``.  Applied at the
        # top-level Agent boundary so a hung tool, runaway tool loop,
        # or slow provider can't block a caller forever.  ``None``
        # disables the deadline (backwards-compatible default).
        timeout: float | None = None,
        # Convenience shortcuts for provider retry/backoff — forwarded to
        # LLMEngine when the engine is auto-created from a model string.
        # Ignored when ``engine=`` is supplied explicitly (use LLMEngine
        # directly to configure retries on a pre-built engine).
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
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
        self.guard = guard
        self.verify = verify
        self.max_verify = max_verify
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
                if isinstance(raw, Agent) and raw.session is None:
                    raw.session = self.session
                    _safe_register_agent(self.session, raw)
                    _safe_register_tool_edge(self.session, self, raw, label="as_tool")

        # PlanCompiler runs at construction time
        if hasattr(self.engine, "_validate"):
            self.engine._validate(self._tool_map)

    # ------------------------------------------------------------------
    # Core async API
    # ------------------------------------------------------------------

    async def run(self, task: str | Envelope) -> Envelope:
        # ``getattr`` with a default keeps this backwards-compatible for
        # Agents constructed via ``Agent.__new__`` (test helpers, custom
        # subclasses) that haven't set ``self.timeout``.
        timeout = getattr(self, "timeout", None)
        if timeout is None:
            return await self._run_body(task)
        try:
            return await asyncio.wait_for(self._run_body(task), timeout=timeout)
        except TimeoutError:
            return Envelope.error_envelope(
                TimeoutError(f"Agent.run() exceeded timeout={timeout}s")
            )

    async def _run_body(self, task: str | Envelope) -> Envelope:
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

        # F1-a: initialize feedback before the loop so any future code path that
        # reaches attempt≥1 always has a defined value, even after refactoring.
        feedback: str = ""
        current = first

        for attempt in range(self.max_output_retries + 1):
            if attempt > 0:
                current = await self.engine.run(
                    _feedback_env(original_env, feedback),
                    tools=list(self._tool_map.values()),
                    output_type=self.output,
                    memory=self.memory,
                    session=self.session,
                )
                if not current.ok:
                    return current

            # F1-b: schema validation and custom validator are separated so each
            # failure produces a precise, actionable feedback message rather than
            # attributing a custom-validator rejection to a schema mismatch.

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

    async def stream(self, task: str | Envelope) -> AsyncGenerator[str, None]:
        """Stream LLM tokens across the full tool-calling loop.

        Honours ``self.timeout`` between chunks so a stalled provider
        can't hang the caller.  Each ``__anext__`` is wrapped in
        ``asyncio.wait_for`` (per-chunk, not whole-stream) so short
        chunks don't consume the deadline budget.
        """
        env = self._to_envelope(task)
        env = self._inject_sources(env)
        timeout = getattr(self, "timeout", None)

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

    def __call__(self, task: str | Envelope) -> Envelope:
        # Detect whether we're already inside a running event loop.
        # ``asyncio.get_running_loop`` is the forward-compatible way — it
        # raises ``RuntimeError`` when there is no current loop, unlike
        # the deprecated ``get_event_loop`` which implicitly creates one.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No loop running — safe to use asyncio.run directly.
            return asyncio.run(self.run(task))

        # Running inside a loop (Jupyter, FastAPI, asyncio tests, …).
        # We need a fresh loop on a worker thread so we don't try to nest.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self.run(task)).result()

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

            judge = Agent("claude-opus-4-7",
                          system="Reply 'approved' or 'rejected: <reason>'.")
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
        agent = self
        effective_name = name or self.name
        effective_desc = description or self.description or f"Run the {effective_name} agent."

        if verify is None:
            async def _run(task: str) -> Envelope:
                return await agent.run(task)
        else:
            async def _run(task: str) -> Envelope:  # type: ignore[misc]
                from lazybridge.envelope import Envelope as _Env
                from lazybridge.evals import verify_with_retry

                env = _Env.from_task(str(task))
                return await verify_with_retry(
                    agent, env, verify, max_verify=max_verify,
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
        from lazybridge.engines.llm import LLMEngine

        return cls(engine=LLMEngine(model), **kwargs)

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
        # F5: do NOT auto-wrap agents as tools — Plan._exec_step dispatches
        # Agent targets via target.run() directly; the tool wrappers were
        # built but never used, wasting schema-compilation on every chain call.
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
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_envelope(task: str | Envelope) -> Envelope:
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


def _safe_register_agent(session: Any, agent: Agent) -> None:
    """Register ``agent`` on ``session.graph`` if possible, warning on failure.

    Pre-fix this was a silent ``try: ... except Exception: pass`` at
    three call sites — a buggy ``register_agent`` on a custom Session
    subclass would silently drop graph entries and nobody would know.
    """
    if session is None or not hasattr(session, "register_agent"):
        return
    try:
        session.register_agent(agent)
    except Exception as exc:
        import warnings

        warnings.warn(
            f"session.register_agent({getattr(agent, 'name', '?')!r}) raised "
            f"{type(exc).__name__}: {exc}",
            stacklevel=2,
        )


def _safe_register_tool_edge(
    session: Any, outer: Agent, inner: Agent, *, label: str,
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

        results = await asyncio.gather(*[_run_one(a) for a in self.agents],
                                       return_exceptions=True)
        return [
            r if isinstance(r, Envelope)
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

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self.run(task)).result()
