"""Agent — the single public-facing abstraction for LazyBridge."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any, cast

from lazybridge.envelope import Envelope
from lazybridge.tools import Tool, build_tool_map

if TYPE_CHECKING:
    from lazybridge.core.providers.base import Tier


class Agent:
    """Universal agent — ``Agent(engine, tools, state)``.

    Every Agent has the same shape, regardless of what it does:

    - ``engine`` — the brain: decides what happens (LLM, Plan, Human, …)
    - ``tools``  — the capabilities: what the agent can invoke
    - state      — ``memory``, ``session``, ``guard``, ``verify``, ``output``

    **Canonical composition** — give each sub-agent an explicit ``name=``
    and pass it directly in ``tools=[...]``::

        from lazybridge import Agent, LLMEngine, Plan, Step, tool, from_prev, from_step

        search = tool(search_web, name="search", description="Search the web.")

        researcher = Agent(
            name="research",
            engine=LLMEngine("claude-haiku-4-5", system="You are a research expert."),
            tools=[search],
        )
        writer = Agent(
            name="write",
            engine=LLMEngine("gpt-5.4-mini", system="You are a concise writer."),
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
            engine=LLMEngine("claude-opus-4-8"),
            tools=[researcher, writer],
            session=sess,
        )

    The engine is the only thing that changes. Everything else — tools,
    memory, session, guard, output — is the same surface on every Agent.

    **String shortcut** — ``Agent("claude-opus-4-8")`` is sugar for
    ``Agent(engine=LLMEngine("claude-opus-4-8"))``.  Use the explicit
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
        verify: Agent | Callable[[str], Any] | None = None,
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
        # to gate the pre-built engine path so callers can\'t silently bypass
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
            # Phase-3 Block H, T7 — when the engine isn\'t an LLM (Plan,
            # SupervisorEngine, HumanEngine, custom), the auto-name fallback
            # to the engine\'s ``model`` attribute (or to the literal
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
                    f"  ``\'agent\'`` and collide the moment another agent is built or composed.\n"
                    f"  Fix: pass ``name=`` (e.g. ``Agent(engine={engine_kind}(...), name=\'pipeline\')``)."
                )

        # If the caller passed native_tools but also supplied a pre-built
        # engine, push the list onto the engine if it has the attribute.
        # This lets ``Agent(engine=LLMEngine("claude"), native_tools=[...])``
        # work the same as ``Agent("claude", native_tools=[...])``.