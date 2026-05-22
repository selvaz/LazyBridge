"""Agent — the single public-facing abstraction for LazyBridge."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from typing import Any, cast

from lazybridge.envelope import Envelope
from lazybridge.tools import Tool, build_tool_map


class Agent:
    """Universal agent — ``Agent(engine, tools, state)``.

    Every Agent has the same shape, regardless of what it does:

    - ``engine`` — the brain: decides what happens (LLM, Plan, Human, …)
    - ``tools``  — the capabilities: what the agent can invoke
    - state      — ``memory``, ``session``, ``guard``, ``verify``, ``output``

    **Canonical composition** — give each sub-agent an explicit ``name=``
    and pass it directly in ``tools=[...]``::

        from lazybridge import Agent, LLMEngine, Plan, Step, tool, from_step

        search = tool(search_web, name="search", description="Search the web.")

        researcher = Agent(
            name="research",
            engine=LLMEngine("claude-opus-4-7", system="You are a research expert."),
            tools=[search],
        )
        writer = Agent(
            name="write",
            engine=LLMEngine("gpt-4o", system="You are a concise writer."),
        )

        # Deterministic orchestrator — Plan engine
        pipeline = Agent(
            name="pipeline",
            engine=Plan(
                Step("research"),
                Step("write", task=from_prev, context=from_step("research")),
            ),
            tools=[researcher, writer],   # agents passed directly
            session=sess,
        )

        # Dynamic orchestrator — LLM engine
        orchestrator = Agent(
            name="orchestrator",
            engine=LLMEngine("claude-opus-4-7"),
            tools=[researcher, writer],
            session=sess,
        )

    The engine is the only thing that changes. Everything else — tools,
    memory, session, guard, output — is the same surface on every Agent.

    **String shortcut** — ``Agent("claude-opus-4-7")`` is sugar for
    ``Agent(engine=LLMEngine("claude-opus-4-7"))``.  Use the explicit
    form when you need to configure the engine (``system=``, ``max_turns=``,
    ``thinking=``, etc.).

    **The name chain** — ``Agent(name=...)`` is the authoritative key that
    connects every part of the system::

        Agent(name="research")      →  tool map key when passed in tools=[researcher]
        Step("research")            →  looks up "research" in tool map ✓
        from_step("research")       →  reads output of step "research" (in-Plan) ✓
        from_agent("research")      →  reads stored output of "research" (cross-run) ✓
        from_memory("research")     →  reads live memory of "research" ✓

    **Advanced alias / backward compat** — ``.as_tool("alias")`` remains
    available when you need a different name than the agent's own::

        tools=[researcher.as_tool("deep_research")]

    **Factory methods** that build real structure (not pure aliases) live on
    the class:

    - ``Agent.chain(a, b)`` — sequential: builds a ``Plan`` of one ``Step``
      per agent.
    - ``Agent.parallel(*agents)`` — scripted fan-out: returns a
      ``ParallelAgent`` whose ``__call__`` yields one ``Envelope``
      (labelled-text join across every branch, with transitive cost
      rollup).  For typed per-branch ``list[Envelope]`` access call
      ``parallel.run_branches(task)`` (async).
    - ``Agent.from_provider(provider, tier="medium")`` — resolves a tier
      alias (``cheap`` / ``medium`` / ``top`` / …) to that provider's
      current model.

    Extension engines live in :mod:`lazybridge.ext` to respect the
    core/ext import boundary::

        from lazybridge.ext.hil import HumanEngine, SupervisorEngine
        Agent(engine=HumanEngine(timeout=60), tools=[approve])
        Agent(engine=SupervisorEngine(tools=[...]))
    """

    _is_lazy_agent = True  # recognised by _wrap_tool()

    def __init__(
        self,
        engine: str | Any | None = None,
        tools: list[Tool | Callable | Agent] | None = None,
        output: type = str,
        memory: Any | None = None,
        store: Any | None = None,
        sources: list[Any] | None = None,
        guard: Any | None = None,
        verify: Agent | None = None,
        max_verify: int = 3,
        name: str | None = None,
        description: str | None = None,
        session: Any | None = None,
        verbose: bool = False,
        # Convenience: pin a specific model id via ``model=`` on the
        # auto-LLMEngine path.  Only consumed when ``engine=None`` or
        # ``engine=<model_string>``; passing ``model=`` alongside a
        # pre-built engine raises (see the model-vs-engine check below).
        # For tier-alias model selection use ``Agent.from_provider(
        # "anthropic", tier="top")`` — the bare-provider-name shortcut
        # ``Agent("anthropic", ...)`` was removed in 0.7.9.x because it
        # left the model id ambiguous at request time.
        model: str | None = None,
        # Provider-native server-side tools (WEB_SEARCH, CODE_EXECUTION, …).
        # Accepted directly on Agent as a shortcut for
        # ``Agent(engine=LLMEngine(..., native_tools=[...]))``.  Ignored when
        # ``engine=`` is a non-LLM engine.
        native_tools: list[Any] | None = None,
        # Required opt-in for capabilities with broad access (CODE_EXECUTION,
        # COMPUTER_USE).  Forwarded to LLMEngine when auto-created and used
        # to gate the pre-built engine path so callers can't silently bypass
        # the LLMEngine.__init__ check by passing engine= separately.
        allow_dangerous_native_tools: bool = False,
        # --- Resilience kwargs ---
        # Optional post-parse validator.  Runs on the structured ``payload``
        # after schema validation; may raise ValueError to force a
        # retry-with-feedback loop (up to ``max_output_retries``).
        output_validator: Callable[[Any], Any] | None = None,
        max_output_retries: int = 2,
        # Total deadline (seconds) for ``run()``.  ``None`` disables.
        timeout: float | None = None,
        # Provider retry/backoff — forwarded to LLMEngine when the engine
        # is auto-created from a model string.  Ignored when ``engine=``
        # is supplied explicitly (configure on ``LLMEngine`` directly).
        max_retries: int = 3,
        retry_delay: float = 1.0,
        # Fallback agent tried when the primary engine returns an error.
        # ``Agent("claude-opus-4-7", fallback=Agent("gpt-4o"))``.
        fallback: Agent | None = None,
        # Prompt caching — when True, marks the static prefix (system
        # prompt + tools) as cacheable so providers that support it
        # (Anthropic today; OpenAI/DeepSeek auto-cache; Google uses a
        # different API) serve cache hits at ~10% of input token cost.
        # Pass a ``CacheConfig(ttl="1h")`` instance for the longer
        # Anthropic TTL.  Forwarded to LLMEngine when the engine is
        # auto-created.  Ignored when ``engine=`` is supplied explicitly
        # (configure ``LLMEngine(cache=...)`` directly).
        cache: bool | Any = False,
    ) -> None:
        # ``name`` is "explicit" when the caller supplied a real string
        # value (not None / blank).  Used downstream to require a name
        # when the agent is later passed in ``tools=[...]``.
        _name_explicit_flag: bool = name is not None and str(name).strip() != ""
        from lazybridge.engines.llm import LLMEngine

        # Phase-3 Block H, T6 — ``model=`` is only meaningful on the LLM-engine
        # construction path (engine is None or a model-string).  Passing both
        # ``model=`` and a non-string ``engine=`` was silently dropped pre-0.8;
        # 0.7.9 raises so the typo / misconfiguration is visible.
        if model is not None and engine is not None and not isinstance(engine, str):
            raise ValueError(
                f"Agent(model={model!r}, engine={type(engine).__name__}(...)): "
                f"the ``model=`` kwarg is only consumed when ``engine=None`` or "
                f"``engine=<model_string>`` (in which case Agent auto-builds an "
                f"``LLMEngine``).  When you pass a pre-built engine, configure "
                f"the model on that engine itself.\n"
                f"  Fix: drop ``model=`` (engine controls the model), or pass "
                f"the model string directly: ``Agent({model!r}, ...)``."
            )

        # Canonical: Agent(engine=LLMEngine(...)) or Agent(engine=Plan(...))
        # Sugar:     Agent("claude-opus-4-7") → engine is a model string → auto-builds LLMEngine
        #            Agent() → engine is None → defaults to "claude-opus-4-7"
        if engine is None or isinstance(engine, str):
            model_str = model or engine or "claude-opus-4-7"
            self.engine: Any = LLMEngine(
                model_str,
                native_tools=native_tools,
                allow_dangerous_native_tools=allow_dangerous_native_tools,
                max_retries=max_retries,
                retry_delay=retry_delay,
                cache=cache,
            )
        else:
            self.engine = engine
            # Phase-3 Block H, T7 — when the engine isn't an LLM (Plan,
            # SupervisorEngine, HumanEngine, custom), the auto-name fallback
            # to the engine's ``model`` attribute (or to the literal
            # ``"agent"`` placeholder) silently produces ambiguous names that
            # collide once the agent is used as a tool or referenced by a
            # ``Step``.  Require ``name=`` upfront so the failure is at the
            # construction point rather than at first composition.
            if not _name_explicit_flag and not hasattr(self.engine, "model"):
                engine_kind = type(self.engine).__name__
                raise ValueError(
                    f"Agent(engine={engine_kind}(...)) requires an explicit ``name=``.\n"
                    f"  Engines other than ``LLMEngine`` have no ``.model`` attribute to derive\n"
                    f"  a default name from, so the agent would silently get the placeholder\n"
                    f"  ``'agent'`` and collide the moment another agent is built or composed.\n"
                    f"  Fix: pass ``name=`` (e.g. ``Agent(engine={engine_kind}(...), name='pipeline')``)."
                )

        # If the caller passed native_tools but also supplied a pre-built
        # engine, push the list onto the engine if it has the attribute.
        # This lets ``Agent(engine=LLMEngine("claude"), native_tools=[...])``
        # work the same as ``Agent("claude", native_tools=[...])``.
        if native_tools and hasattr(self.engine, "native_tools"):
            from lazybridge.core.types import NativeTool

            resolved = [NativeTool(t) if isinstance(t, str) else t for t in native_tools]
            # Run the same dangerous-tools gate that LLMEngine.__init__ would
            # run — prevents bypassing it by passing engine= separately.
            _DANGEROUS = {NativeTool.CODE_EXECUTION, NativeTool.COMPUTER_USE}
            found = [t for t in resolved if t in _DANGEROUS]
            if found and not allow_dangerous_native_tools:
                names = ", ".join(t.value for t in found)
                raise ValueError(
                    f"Native tools {names} have broad system access. Pass allow_dangerous_native_tools=True to opt in."
                )
            # Merge without dup — preserve order of existing + append new.
            existing = list(getattr(self.engine, "native_tools", []) or [])
            for t in resolved:
                if t not in existing:
                    existing.append(t)
            self.engine.native_tools = existing

        self._tools_raw = list(tools or [])
        # Validate before building the tool map so errors surface early with
        # the agent's current name rather than a wrapped Tool name.
        for _raw in self._tools_raw:
            # Default True so duck-typed agents (MockAgent, custom subclasses)
            # that predate _name_explicit are not rejected.  Only real Agent
            # instances explicitly set this to False when no name= was given.
            if getattr(_raw, "_is_lazy_agent", False) and getattr(_raw, "_name_explicit", True) is False:
                _raw_name = getattr(_raw, "name", repr(_raw))
                raise ValueError(
                    f"Agent used as a tool must have an explicit name=...\n"
                    f"The agent currently has name={_raw_name!r} "
                    f"(derived from the model or left as the default).\n\n"
                    f"Set an explicit name:\n"
                    f'    Agent(name="research", engine=LLMEngine(...))\n\n'
                    f"Or use an alias:\n"
                    f'    agent.as_tool("research")\n'
                    f'    tool(agent, name="research")'
                )
        self._tool_map: dict[str, Tool] = build_tool_map(self._tools_raw)
        self.output = output
        self.output_validator = output_validator
        self.max_output_retries = max_output_retries
        self.timeout = timeout
        self.memory = memory
        self.store = store
        self.sources = list(sources or [])
        if max_verify < 1:
            raise ValueError(f"max_verify must be >= 1, got {max_verify!r}")
        if max_output_retries < 0:
            raise ValueError(f"max_output_retries must be >= 0, got {max_output_retries!r}")
        self.guard = guard
        self.verify = verify
        self.max_verify = max_verify
        self.fallback = fallback
        if self.fallback is not None:
            seen: set[int] = {id(self)}
            fb: Agent | None = self.fallback
            while fb is not None:
                if id(fb) in seen:
                    raise ValueError("fallback= chain contains a cycle. Check your Agent(fallback=...) configuration.")
                seen.add(id(fb))
                fb = getattr(fb, "fallback", None)
        self.name: str = str(name or getattr(self.engine, "model", None) or "agent")
        self.description = description
        #: True when the caller supplied an explicit ``name=``.  False
        #: when the name was derived from the model string or left as
        #: the ``"agent"`` default.
        #: Used by ``build_tool_map`` and the ``tool()`` factory to require
        #: an explicit identity before an Agent is used as a sub-agent tool.
        self._name_explicit: bool = _name_explicit_flag

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
                if not getattr(raw, "_is_lazy_agent", False):
                    continue
                agent_raw = cast("Agent", raw)
                child_session = getattr(agent_raw, "session", None)

                if child_session is None:
                    # Propagate parent session down to child and register both
                    # the agent node and the parent → child edge.
                    agent_raw.session = self.session
                    _safe_register_agent(self.session, agent_raw)
                    _safe_register_tool_edge(self.session, self, agent_raw, label=agent_raw.name)
                elif child_session is self.session:
                    # Child already shares the same session (canonical pattern:
                    # all agents built with session= up front).  Register the
                    # edge — it was missing because the old guard checked for
                    # ``session is None`` only.
                    _safe_register_agent(self.session, agent_raw)
                    _safe_register_tool_edge(self.session, self, agent_raw, label=agent_raw.name)
                # else: child belongs to a different session — don't steal it.
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

        # Write last output to shared store so from_agent("name") can read it.
        # Only written on success — failed runs do not overwrite the last good output.
        _store = getattr(self, "store", None)
        if _store is not None and result.ok:
            from lazybridge.sentinels import _AGENT_OUTPUT_KEY_PREFIX

            _store.write(_AGENT_OUTPUT_KEY_PREFIX + self.name, result.text())

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
        chunks: list[str] = []
        _completed = False
        try:
            while True:
                try:
                    if timeout is None:
                        chunk = await gen.__anext__()
                    else:
                        chunk = await asyncio.wait_for(gen.__anext__(), timeout=timeout)
                except StopAsyncIteration:
                    _completed = True
                    return
                yield chunk
                chunks.append(chunk)
        finally:
            aclose = getattr(gen, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except asyncio.CancelledError:
                    # Cancellation is BaseException; let it propagate so
                    # structured cancellation works through the boundary.
                    raise
                except Exception as exc:
                    # Surface non-cancellation errors as a UserWarning so
                    # buggy provider cleanup paths don't disappear; we
                    # still don't re-raise (the consumer has either
                    # finished or already abandoned the stream).
                    import warnings as _w

                    _w.warn(
                        f"stream() aclose raised {type(exc).__name__}: {exc}.",
                        stacklevel=2,
                    )
            if _completed:
                _store = getattr(self, "store", None)
                if _store is not None and self.name and chunks:
                    from lazybridge.sentinels import _AGENT_OUTPUT_KEY_PREFIX

                    _store.write(_AGENT_OUTPUT_KEY_PREFIX + self.name, "".join(chunks))

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
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop running — safe to run directly on a new loop.
            return _run_on_new_loop(self.run(task, images=images, audio=audio))

        # nest_asyncio (Spyder / Jupyter) patches the loop and sets
        # ``_nest_patched = True``.  Running directly on the caller's loop
        # avoids an event-loop lifetime mismatch: httpx/anyio transports
        # created during the call stay bound to a loop that is never closed,
        # so their cleanup tasks succeed silently instead of raising
        # ``RuntimeError: Event loop is closed``.
        if getattr(loop, "_nest_patched", False):
            return loop.run_until_complete(self.run(task, images=images, audio=audio))

        # True async framework (FastAPI, asyncio tests, …) — spin up a fresh
        # loop on a worker thread, copying the caller's contextvars context so
        # OTel spans / request IDs / structured-logging context flow into the
        # agent's loop instead of starting empty.
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
        """Wrap this agent as a :class:`Tool` (advanced / compatibility API).

        The canonical way to use a sub-agent is to give it an explicit
        ``name=`` and pass it directly in ``tools=[...]``::

            # Canonical
            researcher = Agent(name="research", engine=LLMEngine(...))
            orchestrator = Agent(..., tools=[researcher])

        ``.as_tool()`` remains available for **local aliases** and
        **backward compatibility**::

            # Advanced alias — use a different name than the agent's own
            tools=[researcher.as_tool("deep_research")]

            # Backward compat — existing code that already calls as_tool()
            tools=[researcher.as_tool("research")]

        The tool schema is ``(task: str) -> Envelope``.

        Verify (Option B) — wrap the call in a judge/retry loop so every
        invocation is vetted before returning::

            judge = Agent(engine=LLMEngine(
                "claude-opus-4-7",
                system="Reply 'approved' or 'rejected: <reason>'.",
            ))
            synth = Agent(name="synth", engine=LLMEngine(...))
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
                result = await agent.run(task)
                # Always write under the alias so from_agent("alias") can find
                # the output regardless of agent.name.  _run_body also writes
                # under agent.name (for standalone callers); the alias write
                # here is the authoritative key for Plan sentinel resolution.
                _store = getattr(agent, "store", None)
                if _store is not None and result.ok and effective_name != getattr(agent, "name", None):
                    from lazybridge.sentinels import _AGENT_OUTPUT_KEY_PREFIX

                    _store.write(_AGENT_OUTPUT_KEY_PREFIX + effective_name, result.text())
                return result
        else:

            async def _run(task: str) -> Envelope:  # type: ignore[misc]
                from lazybridge._verify import verify_with_retry
                from lazybridge.envelope import Envelope as _Env

                env = _Env.from_task(str(task))
                result = await verify_with_retry(
                    agent,
                    env,
                    verify,
                    max_verify=max_verify,
                )
                _store = getattr(agent, "store", None)
                if _store is not None and result.ok and effective_name != getattr(agent, "name", None):
                    from lazybridge.sentinels import _AGENT_OUTPUT_KEY_PREFIX

                    _store.write(_AGENT_OUTPUT_KEY_PREFIX + effective_name, result.text())
                return result

        _run.__name__ = effective_name
        _run.__doc__ = effective_desc

        return Tool(
            _run,
            name=effective_name,
            description=effective_desc,
            mode="signature",
            returns_envelope=True,
            agent_memory=getattr(self, "memory", None),
            agent_store=getattr(self, "store", None),
        )

    def definition(self) -> Any:
        """ToolDefinition for this agent — used when passed in tools=[] of another agent."""
        return self.as_tool().definition()

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

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
    ) -> ParallelAgent:
        """Deterministic fan-out: run ``agents`` concurrently on the same task.

        Returns a :class:`ParallelAgent` whose ``__call__`` produces a
        single :class:`Envelope` — labelled-text join of every branch's
        output, with transitive cost rollup.  For typed access to per-branch
        envelopes call ``ParallelAgent.run_branches(task)`` (async).

        Use this when you **know** you want N things to happen in
        parallel.  If you want the LLM to decide whether to call agents
        in parallel (and which, and how), don't use this — pass them as
        ``tools=[...]`` on a regular ``Agent`` instead; the engine emits
        parallel tool calls automatically when the model requests them.
        """
        return ParallelAgent(
            agents=list(agents),
            concurrency_limit=concurrency_limit,
            step_timeout=step_timeout,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Note on ext-engine factories
    # ------------------------------------------------------------------
    #
    # Ext engines (Supervisor, Human, dynamic planners) deliberately have
    # no factory on this class — the core-vs-ext boundary (see
    # ``docs/guides/core-vs-ext.md``) forbids ``lazybridge/`` core from
    # importing ``lazybridge.ext.*``, even via lazy/local imports.
    # Two construction paths:
    #
    # 1. Direct: pass the ext engine instance to the canonical Agent ctor::
    #
    #        from lazybridge.ext.hil import SupervisorEngine
    #        Agent(engine=SupervisorEngine(tools=[...], agents=[...]))
    #
    # 2. Module-level ergonomic factories shipped in each ext package::
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


def _suppress_loop_closed(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
    """Swallow 'Event loop is closed' noise emitted by httpx/anyio cleanup tasks.

    When a fresh loop is closed after the coroutine finishes, the GC may later
    call ``AsyncClient.__del__`` which tries to schedule an ``aclose()`` task on
    the now-closed loop.  The resulting ``RuntimeError`` is benign — the request
    already completed successfully — but without this handler it prints a
    confusing traceback to stderr.
    """
    exc = context.get("exception")
    if isinstance(exc, RuntimeError) and str(exc) == "Event loop is closed":
        return
    loop.default_exception_handler(context)


def _run_on_new_loop(coro: Any) -> Any:
    """Run *coro* on a fresh event loop with a clean-exit exception handler.

    Replaces bare ``asyncio.run()`` at the ``__call__`` boundary so that
    httpx/anyio 'Event loop is closed' cleanup noise is suppressed without
    hiding real errors.
    """
    loop = asyncio.new_event_loop()
    loop.set_exception_handler(_suppress_loop_closed)
    try:
        return loop.run_until_complete(coro)
    finally:
        try:
            pending = asyncio.all_tasks(loop)
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        loop.close()


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
        return ctx.run(_run_on_new_loop, coro)

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


class ParallelAgent:
    """Deterministic fan-out over N agents — the shape behind :meth:`Agent.parallel`.

    Pre-scripted parallel runner.  Every input agent receives the same
    task; the N branch results are folded into a single :class:`Envelope`
    via labelled-text join — same shape as :class:`Plan`'s
    ``from_parallel_all`` aggregator.  Cost roll-up is transitive.
    The first non-``None`` branch error propagates as the wrapper's
    ``error`` so downstream consumers can short-circuit.

    Prefer :class:`Agent` with ``tools=[...]`` when you want the engine
    (LLM, Supervisor, Plan) to decide dynamically which tools to invoke
    and when — parallel execution is automatic on that path.

    Per-branch typed access: call :meth:`run_branches` (async) when you
    need ``list[Envelope]`` rather than the joined wrapper.
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

    async def run_branches(self, task: str | Envelope) -> list[Envelope]:
        """Async per-branch entry point — returns one ``Envelope`` per
        input agent in input order.  Use this when you need typed
        access to individual branch results; for the framework-uniform
        single-Envelope view, use :meth:`run` or ``__call__``.
        """
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
        out: list[Envelope] = []
        for r in results:
            if isinstance(r, Envelope):
                out.append(r)
            elif isinstance(r, asyncio.CancelledError):
                # CancelledError is BaseException (not Exception) in Python 3.8+;
                # wrapping it as an error envelope would silently swallow the
                # cancellation signal. Re-raise so structured cancellation works.
                raise r
            elif isinstance(r, Exception):
                out.append(Envelope.error_envelope(r))
            else:
                out.append(Envelope.error_envelope(RuntimeError(str(r))))
        return out

    def _join_branches(self, task: str | Envelope, branches: list[Envelope]) -> Envelope:
        """Fold N branch envelopes into ONE Envelope (labelled-text join).

        Same shape as :meth:`Plan._aggregate_parallel_band`.  Used by both
        ``run()`` / ``__call__()`` and ``as_tool()`` so direct callers and
        tool-wrapped callers see identical output.
        """
        from lazybridge.envelope import EnvelopeMetadata

        sections = [
            f"[{a.name}]\n{e.text() if not e.error else f'(error) {e.error.message}'}"
            for a, e in zip(self.agents, branches)
        ]
        joined = "\n\n".join(sections)
        nested_in = sum(e.metadata.input_tokens + e.metadata.nested_input_tokens for e in branches)
        nested_out = sum(e.metadata.output_tokens + e.metadata.nested_output_tokens for e in branches)
        nested_cost = sum(e.metadata.cost_usd + e.metadata.nested_cost_usd for e in branches)
        first_error = next((e.error for e in branches if e.error), None)
        return Envelope(
            task=task if isinstance(task, str) else task.task,
            payload=joined,
            metadata=EnvelopeMetadata(
                nested_input_tokens=nested_in,
                nested_output_tokens=nested_out,
                nested_cost_usd=nested_cost,
            ),
            error=first_error,
        )

    async def run(self, task: str | Envelope) -> Envelope:
        """Run every branch and return one folded :class:`Envelope`.

        The wrapper's ``payload`` is the labelled-text join of every
        branch's ``.text()``; ``metadata.nested_*`` rolls every branch's
        cost up so the outer envelope reports total spend.  The first
        non-``None`` branch error propagates as the wrapper's ``error``.

        For typed per-branch access, call :meth:`run_branches`.
        """
        branches = await self.run_branches(task)
        return self._join_branches(task, branches)

    def __call__(self, task: str | Envelope) -> Envelope:
        # Mirror ``Agent.__call__`` — ``get_running_loop`` is the only
        # forward-compatible detection (``get_event_loop`` is deprecated
        # under 3.12 and errors under 3.14+ when no loop is running).
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return _run_on_new_loop(self.run(task))
        if getattr(loop, "_nest_patched", False):
            return loop.run_until_complete(self.run(task))
        # Propagate caller contextvars into the worker loop.
        return _run_coro_with_context(self.run(task))

    # ------------------------------------------------------------------
    # Tool-is-Tool — ``run()`` already returns a single Envelope so the
    # tool wrapper just delegates.  Pre-Block-F this method duplicated
    # the labelled-text join because ``run()`` returned ``list[Envelope]``;
    # now both paths share ``_join_branches``.
    # ------------------------------------------------------------------

    def as_tool(
        self,
        name: str | None = None,
        description: str | None = None,
    ) -> Tool:
        """Expose the fan-out runner as a single :class:`Tool`.

        Just delegates to :meth:`run` — same labelled-text Envelope as
        every direct caller sees, so a ``ParallelAgent`` passed in
        ``tools=[...]`` produces output identical to a hand-call.
        """
        from lazybridge.tools import Tool

        actual_name = name or self.name or "parallel"
        actual_desc = (
            description or self.description or (f"Run {len(self.agents)} agents in parallel and join their outputs.")
        )

        async def _run(task: str) -> Envelope:
            return await self.run(task)

        _run.__name__ = actual_name
        _run.__doc__ = actual_desc

        return Tool(
            _run,
            name=actual_name,
            description=actual_desc,
            mode="signature",
            returns_envelope=True,
        )
