"""LLMEngine — agentic tool-calling loop over any provider."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Literal

from lazybridge.core.executor import Executor
from lazybridge.core.types import (
    CompletionRequest,
    CompletionResponse,
    Message,
    NativeTool,
    Role,
    StructuredOutputConfig,
    ToolCall,
)
from lazybridge.envelope import Envelope, EnvelopeMetadata, ErrorInfo
from lazybridge.session import EventType

if TYPE_CHECKING:
    from lazybridge.core.providers.base import BaseProvider
    from lazybridge.memory import Memory
    from lazybridge.session import Session
    from lazybridge.tools import Tool


class ToolTimeoutError(Exception):
    """Raised when a tool exceeds ``LLMEngine.tool_timeout``.

    The engine catches this internally and reports the failure to the
    model loop as ``ToolResultContent(is_error=True)`` so the model
    can recover; it does not abort the agent run.
    """


class StreamStallError(Exception):
    """Raised when a streaming response goes idle past ``stream_idle_timeout``.

    Distinct from ``request_timeout`` (total deadline) — this fires
    when the time *between* successive chunks exceeds the threshold,
    catching half-open streams and partial provider outages without
    killing fast streams that legitimately take a long time end-to-end.
    """


class LLMEngine:
    """Drives the LLM ↔ tool-call loop for a single agent invocation.

    Parameters
    ----------
    model:
        Model string, e.g. "claude-opus-4-7". Provider is inferred automatically.
    thinking:
        Enable extended thinking (Anthropic) or reasoning (OpenAI o-series).
    max_turns:
        Maximum tool-call rounds before giving up with MaxTurnsExceeded error.
        Default 20 — a plain tool-using task typically completes in 2-5 rounds;
        the 20-round budget leaves headroom for deeper reasoning loops without
        capping legitimate pipelines.  Bump higher for deliberately long
        agentic tasks; lower it during dev to fail fast.
    tool_choice:
        "auto" — provider decides when to call tools (default).
        "any"  — provider must call at least one tool.

        When the model emits multiple tool calls in a single turn,
        LazyBridge always executes them concurrently via ``asyncio.gather``.
        That is a capability of the engine, not a configuration knob;
        there is no "serial" execution path for LLM-emitted tool calls.
    temperature:
        Sampling temperature. None = provider default.
    system:
        Static system prompt. Agent.sources= / Envelope.context are added on top.
    native_tools:
        Provider-native server-side tools, e.g. NativeTool.WEB_SEARCH.
    max_retries:
        Retries on transient provider errors (429, 5xx, network/timeout).
        Default 3 — production-safe.  Pass 0 to disable.
    retry_delay:
        Base delay (seconds) for exponential backoff with ±10% jitter.
    request_timeout:
        Per-completion deadline in seconds.  Caps the time a hung
        provider can block an agent run.  ``None`` disables the
        framework-level timeout and defers to the provider SDK.
    max_parallel_tools:
        Maximum number of tool calls executed concurrently within a
        single model turn.  ``None`` (default) means unbounded — every
        tool call returned by the model runs in parallel.  Set to a
        small integer (e.g. 4–8) to apply backpressure on wide tool
        fan-outs and prevent thread/socket/DB exhaustion on a single
        turn.
    tool_timeout:
        Per-tool deadline in seconds.  When set, each tool execution
        is wrapped in ``asyncio.wait_for``.  On timeout the tool's
        result is reported as ``is_error=True`` to the model loop so
        the model can recover; the run does not abort.  ``None``
        (default) leaves tools unbounded.
    stream_idle_timeout:
        Maximum time (seconds) the engine will wait between
        successive streaming chunks before raising
        ``StreamStallError``.  Catches half-open streams without
        killing legitimately long fast streams.  ``None`` (default)
        leaves streams unbounded — preserving the prior behavior.
    """

    # Class-level defaults so tests that bypass ``__init__`` via ``__new__``
    # (and any subclass that forgets to call super) still see safe values.
    max_parallel_tools: int | None = None
    tool_timeout: float | None = None
    stream_idle_timeout: float | None = None
    stream_buffer: int = 64

    def __init__(
        self,
        model: str,
        *,
        provider: str | None = None,
        thinking: bool = False,
        max_turns: int = 20,
        tool_choice: Literal["auto", "any"] = "auto",
        temperature: float | None = None,
        system: str | None = None,
        native_tools: list[NativeTool | str] | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        request_timeout: float | None = 120.0,
        max_parallel_tools: int | None = None,
        tool_timeout: float | None = None,
        stream_idle_timeout: float | None = None,
        stream_buffer: int = 64,
        cache: bool | Any = False,
        strict_multimodal: bool = False,
    ) -> None:
        self.model = model
        self.thinking = thinking
        self.max_turns = max_turns
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_timeout = request_timeout
        if max_parallel_tools is not None and max_parallel_tools < 1:
            raise ValueError(f"max_parallel_tools must be >= 1 or None, got {max_parallel_tools!r}")
        if tool_timeout is not None and tool_timeout <= 0:
            raise ValueError(f"tool_timeout must be > 0 or None, got {tool_timeout!r}")
        if stream_idle_timeout is not None and stream_idle_timeout <= 0:
            raise ValueError(f"stream_idle_timeout must be > 0 or None, got {stream_idle_timeout!r}")
        if stream_buffer < 1:
            raise ValueError(f"stream_buffer must be >= 1, got {stream_buffer!r}")
        self.max_parallel_tools = max_parallel_tools
        self.tool_timeout = tool_timeout
        self.stream_idle_timeout = stream_idle_timeout
        # Bounded queue between the streaming producer (provider) and
        # the consumer (the ``stream()`` async generator).  Pre-W4.1
        # the queue was unbounded, so a slow consumer (slow terminal,
        # slow network, blocked downstream) caused the queue to grow
        # without limit while the provider kept pushing tokens.  A
        # bounded queue propagates backpressure all the way to the
        # provider stream — when the consumer pauses, the producer
        # naturally pauses on ``await sink.put()``.
        self.stream_buffer = stream_buffer
        # Backward-compat: accept ``"parallel"`` but collapse to ``"auto"``
        # with a deprecation warning. The framework no longer has a
        # separate "parallel mode" — tool calls are always executed via
        # asyncio.gather when the model emits more than one in a turn.
        if tool_choice == "parallel":
            import warnings

            warnings.warn(
                "LLMEngine(tool_choice='parallel') is deprecated. "
                "Concurrent tool execution is now the default and cannot be "
                "disabled; drop the argument (or use 'auto'/'any').",
                DeprecationWarning,
                stacklevel=2,
            )
            tool_choice = "auto"
        self.tool_choice = tool_choice
        self.temperature = temperature
        self.system = system
        self.native_tools: list[NativeTool] = [NativeTool(t) if isinstance(t, str) else t for t in (native_tools or [])]
        # Prompt caching — ``cache=True`` enables the default
        # (5-minute TTL on Anthropic; no-op on OpenAI / Google /
        # DeepSeek because they either cache automatically or need a
        # different API).  Callers wanting the 1-hour TTL pass a
        # ``CacheConfig(ttl="1h")`` object directly.
        from lazybridge.core.types import CacheConfig

        if cache is True:
            self.cache: CacheConfig | None = CacheConfig(enabled=True)
        elif cache is False or cache is None:
            self.cache = None
        else:
            self.cache = cache  # assumed CacheConfig
        # When True, ``Envelope.images`` / ``.audio`` reaching a model
        # that does not support that modality raises
        # ``UnsupportedFeatureError`` instead of warning-and-stripping.
        # Off by default so a single agent fleet can mix vision and
        # text-only models without crashing on edge cases.
        self.strict_multimodal = strict_multimodal
        # Provider may be passed explicitly (used by Agent.from_provider
        # when the model is a tier alias like "top" / "cheap" that
        # _infer_provider can't route on its own).  Falls back to the
        # inference heuristic on the model string.
        self.provider = provider or self._infer_provider(model)

    # Provider name aliases accepted as the model argument
    _PROVIDER_NAMES = {
        "anthropic",
        "claude",
        "openai",
        "gpt",
        "google",
        "gemini",
        "deepseek",
        "lmstudio",
        "lm-studio",
        "lm_studio",
        "local",
    }

    #: Exact-match provider aliases — first lookup.  ``Agent("anthropic")`` and
    #: ``Agent("claude")`` both resolve to the ``anthropic`` provider.  Users
    #: can extend this at runtime via :meth:`register_provider_alias`.
    _PROVIDER_ALIASES: dict[str, str] = {
        "anthropic": "anthropic",
        "claude": "anthropic",
        "openai": "openai",
        "gpt": "openai",
        "google": "google",
        "gemini": "google",
        "deepseek": "deepseek",
        "lmstudio": "lmstudio",
        "lm-studio": "lmstudio",
        "lm_studio": "lmstudio",
        "local": "lmstudio",
        # LM Studio's conventional placeholder model identifier — accepting
        # it as a routing alias means ``Agent("local-model")`` Just Works
        # without forcing users to think about provider names.
        "local-model": "lmstudio",
    }

    #: Ordered substring / prefix rules applied when no exact alias matches.
    #: Each entry is ``(kind, pattern, provider)`` where ``kind`` is
    #: ``"contains"`` or ``"startswith"``.  First match wins.  Users can
    #: prepend new rules via :meth:`register_provider_rule` — newly
    #: registered rules take priority over built-ins so shipping a new
    #: "claude-opus-5-*" alias is one call, not a code edit.
    _PROVIDER_RULES: list[tuple[str, str, str]] = [
        ("startswith", "litellm/", "litellm"),  # opt-in catch-all bridge
        # LM Studio: explicit ``lmstudio/<model>`` prefix routes a specific
        # locally-loaded model through the local server without any other
        # configuration — mirrors the ``litellm/`` opt-in pattern.
        ("startswith", "lmstudio/", "lmstudio"),
        ("contains", "claude", "anthropic"),
        ("contains", "gpt", "openai"),
        ("startswith", "o1", "openai"),
        ("startswith", "o3", "openai"),
        ("contains", "gemini", "google"),
        ("contains", "deepseek", "deepseek"),
    ]

    #: Fallback provider when nothing matches.  A warning is emitted on
    #: fallback so ``Agent("grok-2")`` (an unrecognised model) does NOT
    #: silently get routed to Anthropic and fail with a cryptic API-side
    #: error.  Set to ``None`` in a subclass to disable the fallback and
    #: raise ``ValueError`` instead.
    _PROVIDER_DEFAULT: str | None = "anthropic"

    @classmethod
    def set_default_provider(cls, provider: str | None) -> None:
        """Set (or disable) the fallback provider used when no rule matches.

        Two common uses::

            # Production hardening: raise on unknown models rather than
            # silently falling back to the default. Recommended when you
            # only ever want the providers you've explicitly registered.
            LLMEngine.set_default_provider(None)

            # Redirect the safety-net to a different built-in:
            LLMEngine.set_default_provider("openai")

        Why this helper exists: ``Agent("grok-2")`` would default-route
        to Anthropic and fail several RTTs later with a cryptic
        provider-side "unknown model" error.  Disabling
        the fallback turns that into a loud ``ValueError`` at
        construction time.
        """
        cls._PROVIDER_DEFAULT = provider

    @classmethod
    def register_provider_alias(cls, alias: str, provider: str) -> None:
        """Register an exact-match model-string → provider alias.

        Example::

            LLMEngine.register_provider_alias("mistral", "mistral")
            Agent("mistral")   # resolves to the mistral provider
        """
        cls._PROVIDER_ALIASES = {**cls._PROVIDER_ALIASES, alias.lower(): provider}

    @classmethod
    def register_provider_rule(
        cls,
        pattern: str,
        provider: str,
        *,
        kind: Literal["contains", "startswith"] = "contains",
    ) -> None:
        """Register a substring / prefix routing rule.

        New rules take priority over built-ins so you can override default
        routing without editing the framework source::

            LLMEngine.register_provider_rule("claude-opus-5", "anthropic")
            Agent("claude-opus-5-20260701")   # routed to anthropic
        """
        cls._PROVIDER_RULES = [(kind, pattern.lower(), provider), *cls._PROVIDER_RULES]

    @classmethod
    def _infer_provider(cls, model: str) -> str:
        m = model.lower()
        alias = cls._PROVIDER_ALIASES.get(m)
        if alias is not None:
            return alias
        for kind, pattern, provider in cls._PROVIDER_RULES:
            if kind == "contains" and pattern in m:
                return provider
            if kind == "startswith" and m.startswith(pattern):
                return provider
        # Nothing matched — warn loudly rather than silently route to
        # the default provider.  Raising would break the "no-config
        # Agent('some-model')" ergonomic, so we stick with a warning
        # and let the provider surface its own "unknown model" error.
        if cls._PROVIDER_DEFAULT is None:
            raise ValueError(
                f"No provider rule matches model {model!r} and no default is "
                f"configured. Register a rule via "
                f"LLMEngine.register_provider_rule(...) or set "
                f"_PROVIDER_DEFAULT on a subclass."
            )
        import warnings

        warnings.warn(
            f"No provider rule matches model {model!r}; defaulting to "
            f"{cls._PROVIDER_DEFAULT!r}. Register a rule via "
            f"LLMEngine.register_provider_rule({model!r}, <provider>) to silence.",
            stacklevel=3,
        )
        return cls._PROVIDER_DEFAULT

    def _make_executor(self) -> Executor:
        # Use ``self.provider`` (set at __init__, possibly explicitly by
        # Agent.from_provider) rather than re-running inference.  The
        # BaseProvider handles tier / provider-name aliases on the model
        # string via ``resolve_model_alias``; there is no reason to
        # special-case them here.
        return Executor(
            self.provider,
            model=self.model,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        )

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------

    async def run(
        self,
        env: Envelope,
        *,
        tools: list[Tool],
        output_type: type,
        memory: Memory | None,
        session: Session | None,
        store: Any | None = None,  # accepted-and-ignored — Plan checkpoint surface
        plan_state: Any | None = None,  # accepted-and-ignored — Plan checkpoint surface
    ) -> Envelope:
        run_id = str(uuid.uuid4())
        t_start = time.monotonic()
        agent_name = getattr(self, "_agent_name", "agent")

        if session:
            session.emit(EventType.AGENT_START, {"agent_name": agent_name, "task": env.task}, run_id=run_id)

        try:
            result = await self._loop(
                env,
                tools=tools,
                output_type=output_type,
                memory=memory,
                session=session,
                run_id=run_id,
            )
        except Exception as exc:
            if session:
                session.emit(EventType.AGENT_FINISH, {"agent_name": agent_name, "error": str(exc)}, run_id=run_id)
            return Envelope.error_envelope(exc)

        latency_ms = (time.monotonic() - t_start) * 1000
        # Rebuild metadata via ``model_copy`` so we don't rely on
        # Pydantic v2's default-mutable semantics.  This stays correct
        # even if a future version ships with ``frozen=True`` on
        # EnvelopeMetadata.
        result = result.model_copy(
            update={
                "metadata": result.metadata.model_copy(
                    update={
                        "latency_ms": latency_ms,
                        "run_id": run_id,
                    }
                ),
            }
        )

        if session:
            session.emit(
                EventType.AGENT_FINISH,
                {"agent_name": agent_name, "payload": result.text(), "latency_ms": latency_ms},
                run_id=run_id,
            )

        return result

    # ------------------------------------------------------------------
    # Multimodal user-message construction
    # ------------------------------------------------------------------

    def _build_user_content(self, task_text: str, env: Envelope) -> str | list[Any]:
        """Build the user-message content payload for ``env``.

        Returns either:

        * a plain ``str`` — when the envelope carries no attachments OR
          all attachments were filtered out by capability checks; the
          existing single-string code path is unchanged.
        * a ``list[ContentBlock]`` of ``[TextContent, ImageContent...,
          AudioContent?]`` — when the resolved model honours one or
          more of the attached blocks.

        Capability gating: when the provider reports that the resolved
        model cannot handle a kind of attachment (vision / audio), the
        attachments of that kind are dropped with a ``UserWarning``.
        Setting :attr:`strict_multimodal` flips the warning into a
        :class:`UnsupportedFeatureError` so callers running in CI catch
        the misconfiguration before runtime.
        """
        from lazybridge.core.types import TextContent

        images = list(env.images) if env.images else []
        audio = env.audio
        if not images and audio is None:
            return task_text

        kept_images = self._filter_by_vision(images)
        kept_audio = self._filter_by_audio(audio)

        if not kept_images and kept_audio is None:
            return task_text

        blocks: list[Any] = [TextContent(text=task_text)]
        blocks.extend(kept_images)
        if kept_audio is not None:
            blocks.append(kept_audio)
        return blocks

    def _provider_class(self) -> type[BaseProvider]:
        """Resolve the provider class WITHOUT instantiating a client.

        Capability checks (``supports_vision`` / ``supports_audio``) are
        class-level — they read the model name and consult provider
        capability tables.  Going through ``Executor._resolve_provider``
        would needlessly construct an SDK client (and demand an API key)
        just to read static metadata.
        """
        from lazybridge.core.providers.anthropic import AnthropicProvider
        from lazybridge.core.providers.deepseek import DeepSeekProvider
        from lazybridge.core.providers.google import GoogleProvider
        from lazybridge.core.providers.lmstudio import LMStudioProvider
        from lazybridge.core.providers.openai import OpenAIProvider

        registry = {
            "anthropic": AnthropicProvider,
            "claude": AnthropicProvider,
            "openai": OpenAIProvider,
            "gpt": OpenAIProvider,
            "google": GoogleProvider,
            "gemini": GoogleProvider,
            "deepseek": DeepSeekProvider,
            "lmstudio": LMStudioProvider,
            "lm-studio": LMStudioProvider,
            "lm_studio": LMStudioProvider,
            "local": LMStudioProvider,
        }
        key = self.provider.lower().strip() if isinstance(self.provider, str) else "anthropic"
        if key == "litellm":
            from lazybridge.core.providers.litellm import LiteLLMProvider

            return LiteLLMProvider
        return registry.get(key, AnthropicProvider)

    def _filter_by_vision(self, images: list[Any]) -> list[Any]:
        if not images:
            return []
        provider_cls = self._provider_class()
        if provider_cls.supports_vision(self.model):
            return images
        msg = (
            f"{provider_cls.__name__} model {self.model!r} does not "
            f"support vision input — {len(images)} image(s) dropped from the "
            f"request.  Pass a vision-capable model or set strict_multimodal=True "
            f"to fail fast."
        )
        if self.strict_multimodal:
            from lazybridge.core.providers.base import UnsupportedFeatureError

            raise UnsupportedFeatureError(msg)
        import warnings

        warnings.warn(msg, UserWarning, stacklevel=4)
        return []

    def _filter_by_audio(self, audio: Any | None) -> Any | None:
        if audio is None:
            return None
        provider_cls = self._provider_class()
        if provider_cls.supports_audio(self.model):
            return audio
        msg = (
            f"{provider_cls.__name__} model {self.model!r} does not "
            f"support audio input — attachment dropped from the request.  "
            f"Pass an audio-capable model or set strict_multimodal=True to "
            f"fail fast."
        )
        if self.strict_multimodal:
            from lazybridge.core.providers.base import UnsupportedFeatureError

            raise UnsupportedFeatureError(msg)
        import warnings

        warnings.warn(msg, UserWarning, stacklevel=4)
        return None

    # ------------------------------------------------------------------
    # _loop() — shared by run() and stream()
    # ------------------------------------------------------------------

    async def _loop(
        self,
        env: Envelope,
        *,
        tools: list[Tool],
        output_type: type,
        memory: Memory | None,
        session: Session | None,
        run_id: str,
        # When truthy, yield str tokens instead of building Envelope
        _stream_sink: asyncio.Queue[str | None] | None = None,
    ) -> Envelope:

        from lazybridge.core.types import TextContent, ThinkingConfig, ToolResultContent, ToolUseContent

        executor = self._make_executor()

        messages: list[Message] = []
        if memory:
            messages.extend(memory.messages())

        system = self.system or ""
        if env.context:
            system = f"{system}\n\nContext:\n{env.context}".strip() if system else f"Context:\n{env.context}"

        task_text = env.task or env.text()
        # Multimodal: when the Envelope carries images / audio, build a
        # content-block list instead of a plain string so the provider's
        # ``_messages_to_<provider>`` method emits the right wire shape.
        # Capability filtering (M1.5) drops attachments the resolved model
        # can't handle before they reach the wire — done in
        # ``_build_user_content`` so the LLM-loop ↔ provider boundary
        # stays tidy.
        user_content = self._build_user_content(task_text, env)
        messages.append(Message(role=Role.USER, content=user_content))

        tool_defs = [t.definition() for t in tools]
        tool_map = {t.name: t for t in tools}

        structured_cfg: StructuredOutputConfig | None = None
        # Activate structured output when the caller declared a non-str
        # output type — including generic forms like ``list[MyModel]``
        # or ``dict[str, MyModel]``.  ``isinstance(type)`` would reject
        # generics (``list[X]`` isn't a ``type``), so the check is
        # explicit identity against ``str`` / ``Any``.
        if output_type is not str and output_type is not Any:
            from typing import get_origin

            if isinstance(output_type, type) or get_origin(output_type) is not None:
                structured_cfg = StructuredOutputConfig(schema=output_type)

        thinking_cfg = ThinkingConfig(enabled=True) if self.thinking else None
        # "parallel" is our asyncio strategy; provider always gets "auto"
        provider_tc = self.tool_choice

        total_in = total_out = 0
        cost = 0.0
        # Nested aggregation: when a tool returns an ``Envelope`` (an
        # Agent wrapped via ``as_tool``), its own metadata is added to
        # these buckets so the outer Envelope reflects total pipeline
        # cost without double-counting with ``total_in`` / ``total_out``.
        nested_in = nested_out = 0
        nested_cost = 0.0
        model_used: str | None = None

        for turn in range(self.max_turns):
            if session:
                session.emit(EventType.LOOP_STEP, {"turn": turn, "messages": len(messages)}, run_id=run_id)

            req = CompletionRequest(
                messages=messages,
                system=system or None,
                temperature=self.temperature,
                tools=tool_defs,
                native_tools=self.native_tools,
                tool_choice=provider_tc if tool_defs else None,
                # Always pass structured_cfg regardless of whether tools are
                # present.  On turns where the model emits tool calls the
                # constraint is honoured but unused (the response carries
                # tool_use blocks, not text).  On the final turn — when the
                # model produces its answer — the provider enforces the schema
                # server-side so resp.parsed is populated and the post-hoc
                # _validate_and_retry path immediately succeeds.  Providers that
                # don't support the combination fall back gracefully to the
                # existing post-hoc validation in Agent._validate_and_retry.
                structured_output=structured_cfg,
                thinking=thinking_cfg,
                # ``getattr`` keeps this safe when an engine has been
                # built via ``LLMEngine.__new__`` (test factories, custom
                # subclasses) without going through ``__init__``.
                cache=getattr(self, "cache", None),
                stream=_stream_sink is not None,
            )

            if session:
                session.emit(
                    EventType.MODEL_REQUEST,
                    {"provider": executor._provider.__class__.__name__, "model": self.model, "turn": turn},
                    run_id=run_id,
                )

            if _stream_sink is not None:
                # Streaming path — accumulate full response while yielding tokens.
                # request_timeout intentionally does NOT wrap streaming:
                # a slow stream is still making progress, and wrapping the
                # whole iterator in ``wait_for`` would cancel on total
                # duration rather than stall duration.
                resp = await self._stream_turn(executor, req, _stream_sink)
            elif self.request_timeout is not None:
                resp = await asyncio.wait_for(executor.aexecute(req), timeout=self.request_timeout)
            else:
                resp = await executor.aexecute(req)

            model_used = resp.model or self.model
            total_in += resp.usage.input_tokens
            total_out += resp.usage.output_tokens
            if resp.usage.cost_usd:
                cost += resp.usage.cost_usd

            if session:
                session.emit(
                    EventType.MODEL_RESPONSE,
                    {
                        "content": resp.content[:500],
                        "input_tokens": resp.usage.input_tokens,
                        "output_tokens": resp.usage.output_tokens,
                        "cost_usd": resp.usage.cost_usd,
                        "stop_reason": resp.stop_reason,
                    },
                    run_id=run_id,
                )

            if not resp.tool_calls:
                payload: Any
                if structured_cfg and resp.parsed:
                    payload = resp.parsed
                else:
                    payload = resp.content

                if memory:
                    # Pass this turn's marginal token count, not the
                    # running cumulative total.  total_in+total_out is the
                    # pipeline total across all turns; Memory's compression
                    # threshold should be measured against per-turn cost.
                    memory.add(
                        task_text,
                        resp.content,
                        tokens=resp.usage.input_tokens + resp.usage.output_tokens,
                    )

                return Envelope(
                    task=env.task,
                    context=env.context,
                    payload=payload,
                    metadata=EnvelopeMetadata(
                        input_tokens=total_in,
                        output_tokens=total_out,
                        cost_usd=cost,
                        nested_input_tokens=nested_in,
                        nested_output_tokens=nested_out,
                        nested_cost_usd=nested_cost,
                        model=model_used,
                        provider=executor._provider.__class__.__name__,
                    ),
                )

            # Append assistant turn with tool calls
            assistant_blocks: list[Any] = []
            if resp.content:
                assistant_blocks.append(TextContent(text=resp.content))
            for tc in resp.tool_calls:
                assistant_blocks.append(ToolUseContent(id=tc.id, name=tc.name, input=tc.arguments))
            messages.append(Message(role=Role.ASSISTANT, content=assistant_blocks))

            # Execute tool calls concurrently.  A single tool call in a
            # turn is just a one-element gather; N calls run in parallel.
            # Tool-is-Tool uniformity: each ``tc`` may target a plain
            # function, an Agent wrapped via ``as_tool()``, or an Agent of
            # Agents — the engine does not special-case any of them.
            # When ``max_parallel_tools`` is set, a semaphore caps the
            # in-flight count to apply backpressure on wide fan-outs.
            sem = asyncio.Semaphore(self.max_parallel_tools) if self.max_parallel_tools is not None else None

            async def _run_one(tc: ToolCall, *, _sem: asyncio.Semaphore | None = sem) -> Any:
                if _sem is None:
                    return await self._exec_tool(tc, tool_map, session=session, run_id=run_id)
                async with _sem:
                    return await self._exec_tool(tc, tool_map, session=session, run_id=run_id)

            raw_results = await asyncio.gather(
                *[_run_one(tc) for tc in resp.tool_calls],
                return_exceptions=True,
            )

            result_blocks: list[Any] = []
            for tc, tr in zip(resp.tool_calls, raw_results):
                # Detect Envelope-returning tools (agent-as-tool,
                # verified tools): preserve their metadata and error
                # state, stringify via .text() for the provider.
                if isinstance(tr, Envelope):
                    # Aggregate usage. Include the nested Envelope's
                    # own ``nested_*`` so aggregation is transitive
                    # across arbitrarily-deep agent trees.
                    nm = tr.metadata
                    nested_in += nm.input_tokens + nm.nested_input_tokens
                    nested_out += nm.output_tokens + nm.nested_output_tokens
                    nested_cost += nm.cost_usd + nm.nested_cost_usd
                    content = tr.text()
                    is_err = not tr.ok
                elif isinstance(tr, ToolTimeoutError):
                    # Explicit timeout marker so the model can recognise
                    # cancellation distinct from a generic exception.
                    content = f"[TOOL_TIMEOUT] {tr}"
                    is_err = True
                elif isinstance(tr, Exception):
                    content = f"Tool error: {tr}"
                    is_err = True
                else:
                    content = str(tr)
                    is_err = False
                result_blocks.append(
                    ToolResultContent(
                        tool_use_id=tc.id,
                        content=content,
                        tool_name=tc.name,
                        is_error=is_err,
                    )
                )
            messages.append(Message(role=Role.USER, content=result_blocks))

        return Envelope(
            task=env.task,
            error=ErrorInfo(
                type="MaxTurnsExceeded",
                message=f"Reached max_turns={self.max_turns}",
                retryable=False,
            ),
            metadata=EnvelopeMetadata(
                input_tokens=total_in,
                output_tokens=total_out,
                cost_usd=cost,
                nested_input_tokens=nested_in,
                nested_output_tokens=nested_out,
                nested_cost_usd=nested_cost,
                model=model_used,
            ),
        )

    async def _stream_turn(
        self,
        executor: Executor,
        req: CompletionRequest,
        sink: asyncio.Queue[str | None],
    ) -> CompletionResponse:
        """Stream one LLM turn, push tokens to sink, return reconstructed CompletionResponse."""
        from lazybridge.core.types import CompletionResponse, UsageStats

        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        stop_reason = "end_turn"
        usage = UsageStats()
        model_out: str | None = None

        async for chunk in self._idle_guarded_stream(executor.astream(req)):
            if chunk.delta:
                content_parts.append(chunk.delta)
                await sink.put(chunk.delta)
            if chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)
            if chunk.stop_reason:
                stop_reason = chunk.stop_reason
            if chunk.usage:
                usage = chunk.usage
            if chunk.is_final:
                model_out = getattr(chunk, "model", None)

        return CompletionResponse(
            content="".join(content_parts),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
            model=model_out,
        )

    async def _idle_guarded_stream(self, agen: Any) -> AsyncGenerator[Any, None]:
        """Yield items from ``agen``, raising on inter-chunk idle timeout.

        Wraps each ``__anext__`` in ``asyncio.wait_for`` so a stalled
        provider stream raises ``StreamStallError`` instead of pinning
        a worker forever.  When ``stream_idle_timeout`` is ``None``
        this is a transparent passthrough.
        """
        if self.stream_idle_timeout is None:
            async for item in agen:
                yield item
            return
        aiter = agen.__aiter__()
        while True:
            try:
                item = await asyncio.wait_for(aiter.__anext__(), timeout=self.stream_idle_timeout)
            except StopAsyncIteration:
                return
            except TimeoutError as exc:
                raise StreamStallError(
                    f"Stream went idle for {self.stream_idle_timeout}s without "
                    "delivering a chunk (set LLMEngine(stream_idle_timeout=...) "
                    "to a higher value if streams legitimately pause that long)."
                ) from exc
            yield item

    async def _exec_tool(
        self,
        tc: ToolCall,
        tool_map: dict[str, Tool],
        *,
        session: Session | None,
        run_id: str,
    ) -> Any:
        if session:
            # ``tool_use_id`` is the provider-supplied call id; it lets
            # downstream consumers (OTel exporter, audit dashboards)
            # correlate a TOOL_CALL with its eventual TOOL_RESULT /
            # TOOL_ERROR even when N parallel invocations of the same
            # tool name are in flight in a single turn.
            session.emit(
                EventType.TOOL_CALL,
                {"tool": tc.name, "tool_use_id": tc.id, "arguments": tc.arguments},
                run_id=run_id,
            )

        # Loud surfacing of malformed tool-call arguments.
        # Provider ``_safe_json_loads`` tags un-parseable arg blobs with
        # ``_parse_error`` so we can short-circuit here with a structured
        # error instead of letting the tool fail later with a misleading
        # "missing required field" message — the model then has the
        # *real* failure (its JSON output was malformed) in the next
        # turn and can self-correct.
        parse_err = tc.arguments.get("_parse_error") if isinstance(tc.arguments, dict) else None
        if parse_err:
            raw = tc.arguments.get("_raw_arguments", "") if isinstance(tc.arguments, dict) else ""
            err = RuntimeError(
                f"Tool {tc.name!r} received malformed JSON arguments ({parse_err}). Raw arguments: {raw!r}"
            )
            if session:
                session.emit(
                    EventType.TOOL_ERROR,
                    {
                        "tool": tc.name,
                        "tool_use_id": tc.id,
                        "error": str(err),
                        "type": "ToolArgumentParseError",
                        "raw_arguments": raw,
                        "parse_error": parse_err,
                    },
                    run_id=run_id,
                )
            return err

        tool = tool_map.get(tc.name)
        if tool is None:
            err = RuntimeError(f"Unknown tool: {tc.name!r}")
            if session:
                session.emit(
                    EventType.TOOL_ERROR,
                    {"tool": tc.name, "tool_use_id": tc.id, "error": str(err)},
                    run_id=run_id,
                )
            return err

        try:
            if self.tool_timeout is not None:
                try:
                    result = await asyncio.wait_for(tool.run(**tc.arguments), timeout=self.tool_timeout)
                except TimeoutError:
                    timeout_err = ToolTimeoutError(f"Tool {tc.name!r} timed out after {self.tool_timeout}s")
                    if session:
                        # Distinct event type so operators can filter
                        # planned cancellations from genuine exceptions
                        # in dashboards / alerting.
                        session.emit(
                            EventType.TOOL_TIMEOUT,
                            {
                                "tool": tc.name,
                                "tool_use_id": tc.id,
                                "error": str(timeout_err),
                                "type": "ToolTimeoutError",
                                "timeout_s": self.tool_timeout,
                            },
                            run_id=run_id,
                        )
                    return timeout_err
            else:
                result = await tool.run(**tc.arguments)
            if session:
                session.emit(
                    EventType.TOOL_RESULT,
                    {"tool": tc.name, "tool_use_id": tc.id, "result": str(result)[:500]},
                    run_id=run_id,
                )
            return result
        except Exception as exc:
            if session:
                session.emit(
                    EventType.TOOL_ERROR,
                    {
                        "tool": tc.name,
                        "tool_use_id": tc.id,
                        "error": str(exc),
                        "type": type(exc).__name__,
                    },
                    run_id=run_id,
                )
            return exc

    # ------------------------------------------------------------------
    # stream() — full multi-turn loop with token streaming
    # ------------------------------------------------------------------

    async def stream(
        self,
        env: Envelope,
        *,
        tools: list[Tool],
        output_type: type,
        memory: Memory | None,
        session: Session | None,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from the full tool-calling loop.

        Yields str tokens as the LLM generates them. Tool calls between turns
        are executed silently; the next-turn response is then streamed.
        This means token output is continuous across tool-call boundaries.
        """
        run_id = str(uuid.uuid4())
        agent_name = getattr(self, "_agent_name", "agent")

        if session:
            session.emit(EventType.AGENT_START, {"agent_name": agent_name, "task": env.task}, run_id=run_id)

        # Bounded sink — see ``stream_buffer`` on ``__init__``.  The
        # producer ``await sink.put(token)`` naturally blocks when the
        # consumer falls behind; this is the only mechanism that keeps
        # an idle consumer from forcing unbounded memory growth as the
        # provider streams tokens.
        sink: asyncio.Queue[str | None] = asyncio.Queue(maxsize=self.stream_buffer)

        async def _run_loop() -> None:
            try:
                await self._loop(
                    env,
                    tools=tools,
                    output_type=output_type,
                    memory=memory,
                    session=session,
                    run_id=run_id,
                    _stream_sink=sink,
                )
            finally:
                await sink.put(None)  # sentinel — loop done

        task = asyncio.create_task(_run_loop())
        cancelled_by_us = False
        try:
            while True:
                token = await sink.get()
                if token is None:
                    break
                yield token
        finally:
            # If the consumer broke early (e.g. ``break`` out of the
            # ``async for``), cancel the background loop instead of
            # awaiting it — otherwise the provider keeps streaming
            # into a sink no one is reading, racking up cost and
            # tying up worker capacity for the lifetime of the turn.
            if not task.done():
                task.cancel()
                cancelled_by_us = True
            try:
                await task
            except asyncio.CancelledError:
                if not cancelled_by_us:
                    raise
            # Emit AGENT_FINISH regardless of how we exited so streaming
            # runs are observable end-to-end the same way ``run()``
            # invocations are.  The companion AGENT_START is emitted at
            # the top of this method.
            if session:
                session.emit(
                    EventType.AGENT_FINISH,
                    {"agent_name": agent_name, "cancelled": cancelled_by_us},
                    run_id=run_id,
                )
