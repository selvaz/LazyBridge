"""Stable public extension point for custom LLM providers.

Implement ``BaseProvider`` to integrate any LLM backend with LazyBridge.
The contract below is **stable** — breaking changes will follow a deprecation
cycle and a minor-version bump.

Minimal implementation::

    from lazybridge.core.providers.base import BaseProvider
    from lazybridge.core.types import CompletionRequest, CompletionResponse, StreamChunk

    class MyProvider(BaseProvider):
        default_model = "my-model-v1"

        def _init_client(self, **kwargs) -> None:
            import my_sdk
            self._client = my_sdk.Client(api_key=self.api_key)

        def complete(self, request: CompletionRequest) -> CompletionResponse:
            resp = self._client.generate(
                model=self._resolve_model(request),
                messages=[...],   # convert request.messages
            )
            return CompletionResponse(content=resp.text)

        def stream(self, request: CompletionRequest):
            for chunk in self._client.stream(...):
                yield StreamChunk(delta=chunk.text)
            yield StreamChunk(stop_reason="end_turn", is_final=True)

        async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
            ...   # async version of complete()

        async def astream(self, request: CompletionRequest):
            ...   # async generator version of stream()

Usage::

    from lazybridge import Agent, LLMEngine
    agent = Agent(engine=LLMEngine("my-model"))
    print(agent("hello").text())
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal

from lazybridge.core.types import (
    CompletionRequest,
    CompletionResponse,
    NativeTool,
    StreamChunk,
)

#: Canonical tier aliases recognised by ``Agent.from_provider(tier=...)``
#: and each provider's ``_TIER_ALIASES`` map.  Single source of truth for
#: the tier vocabulary — ordered cheapest → most capable.  A string not in
#: this set is treated as a literal model id (passthrough), so callers may
#: still pass an explicit model name where a ``Tier`` is annotated.
Tier = Literal["super_cheap", "cheap", "medium", "expensive", "top"]


class UnsupportedNativeToolError(ValueError):
    """Raised when a request asks for a :class:`NativeTool` the provider
    doesn't implement, and the provider is in strict mode
    (``strict_native_tools=True``).

    Subclasses :class:`ValueError` so existing call sites that broadly
    catch ``ValueError`` don't suddenly need a new branch — but the
    type is distinct so production code can intercept it precisely
    (e.g. to fail-over to a different provider).
    """


class UnsupportedFeatureError(ValueError):
    """Raised when a request asks for a multimodal modality (vision /
    audio) the resolved model does not support, and the engine was
    configured with ``strict_multimodal=True``.

    Distinct from :class:`UnsupportedNativeToolError` so call sites can
    react differently to "the API tier you asked for is missing" (native
    tool) versus "the input you sent is wrong shape for this model"
    (multimodal capability).
    """


class BaseProvider(ABC):
    """Stable abstract base class for all LLM providers.

    Subclass this to integrate any LLM backend with LazyBridge. Plug a
    custom provider in by constructing an ``LLMEngine`` that routes to it
    (see ``lazybridge/core/executor.py`` for resolution)::

        agent = Agent(engine=LLMEngine("my-model"))

    **Stability contract**
    The following are guaranteed stable across minor versions:

    - ``__init__(api_key, model, **kwargs)`` signature
    - ``_init_client(**kwargs)`` — override to initialise your SDK client
    - ``complete(request)`` — synchronous completion
    - ``stream(request)`` — synchronous streaming
    - ``acomplete(request)`` — async completion
    - ``astream(request)`` — async streaming generator
    - ``default_model: str`` — class-level default model name
    - ``supported_native_tools: frozenset[NativeTool]`` — declare web search etc.
    - ``get_default_max_tokens(model)`` — override to set per-model limits
    - ``_resolve_model(request)`` — helper: request.model → self.model → default_model
    - ``_compute_cost(model, input_tokens, output_tokens)`` — override for cost tracking
    - ``_check_native_tools(tools)`` — filters unsupported native tools with a warning

    **What you MUST implement**: ``complete``, ``stream``, ``acomplete``, ``astream``.

    **What you SHOULD override**: ``_init_client``, ``default_model``,
    ``get_default_max_tokens``, ``_compute_cost``.

    **What you MUST NOT do**:
    - Raise exceptions other than Python built-ins or your SDK's own error types.
      LazyBridge does not wrap provider exceptions — they propagate as-is.
    - Mutate ``request`` — it is shared and must be treated as read-only.
    - Block the event loop inside ``acomplete`` / ``astream`` — use ``await`` or
      ``asyncio.get_event_loop().run_in_executor`` for blocking SDK calls.
    """

    default_model: str | None = ""
    """Class-level default model identifier.  Used when neither the request
    nor the constructor ``model=`` argument specifies a model.
    Set to ``None`` on paid cloud providers to force explicit model selection
    and prevent silent fallback to an expensive flagship."""

    supported_native_tools: frozenset[NativeTool] = frozenset()
    """Declare which :class:`~lazybridge.core.types.NativeTool` values this
    provider supports (e.g. ``frozenset({NativeTool.WEB_SEARCH})``).
    Unsupported tools requested by the user are filtered and warned —
    or raised, when ``strict_native_tools=True`` is set on construction."""

    # Phase-3 Block I: declarative capability flags.  Subclasses override
    # these when they don't support a feature; the defaults are the common
    # case (every provider in this repo today supports all three).  Used
    # by ``lazybridge.matrix`` (planned) and by per-provider preflight
    # checks that need to refuse a request the SDK can't fulfil.

    supports_streaming: bool = True
    """Does this provider expose ``stream(...)`` / ``astream(...)``?"""

    supports_structured_output: bool = True
    """Does this provider accept ``request.structured_output`` (Pydantic
    model or JSON-schema dict)?"""

    supports_thinking: bool = True
    """Does this provider produce a ``thinking`` field on the response (or
    ``reasoning_tokens`` / ``thoughts_token_count`` on usage)?"""

    strict_native_tools: bool = False
    """When True, requesting an unsupported :class:`NativeTool` raises
    :class:`UnsupportedNativeToolError` instead of warning-and-dropping.
    Set on construction (``BaseProvider(..., strict_native_tools=True)``)
    or via the subclass.  Default ``False`` preserves the friendly
    pre-W5.1 behaviour for ad-hoc / interactive use.  Production setups
    should consider opting into strict mode so a misconfigured provider
    fails loud rather than degrading to a non-grounded reply."""

    # ------------------------------------------------------------------
    # Multimodal capability matrix — class-level so the LLM engine can
    # consult them without instantiating an SDK client (which would
    # require an API key just to read static metadata).
    # ------------------------------------------------------------------

    #: Substrings — when ANY appears in a model id, the provider is
    #: considered vision-capable for that model.  Subclasses override
    #: with their own canonical set; the base default is empty so a
    #: provider that doesn't opt in stays text-only.
    _VISION_CAPABLE_MODEL_PATTERNS: frozenset[str] = frozenset()

    #: Same shape, audio modality.
    _AUDIO_CAPABLE_MODEL_PATTERNS: frozenset[str] = frozenset()

    @classmethod
    def supports_vision(cls, model: str | None = None) -> bool:
        """Whether the resolved ``model`` accepts image input.

        Default implementation does a substring scan against
        :attr:`_VISION_CAPABLE_MODEL_PATTERNS`.  Override when the
        decision needs custom logic (e.g. version-range checks).

        Returns ``False`` for ``None`` / empty model because we don't
        know what the eventual default will be — caller can re-query
        once the model is resolved.
        """
        if not model:
            return False
        m = model.lower()
        return any(p in m for p in cls._VISION_CAPABLE_MODEL_PATTERNS)

    @classmethod
    def supports_audio(cls, model: str | None = None) -> bool:
        """Whether the resolved ``model`` accepts audio input.

        See :meth:`supports_vision` — same semantics, audio modality.
        """
        if not model:
            return False
        m = model.lower()
        return any(p in m for p in cls._AUDIO_CAPABLE_MODEL_PATTERNS)

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        fallback_model: str | None = None,
        strict_native_tools: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the provider.

        Parameters
        ----------
        api_key:
            Provider API key.  If ``None``, ``_init_client`` reads it from
            an environment variable (standard pattern for all built-in providers).
        model:
            Model identifier to use for all requests.  When ``None`` and
            ``default_model`` is also ``None`` (recommended for paid cloud
            providers), ``_resolve_model`` raises a clear ``ValueError``
            rather than silently falling back to an expensive flagship.
        fallback_model:
            Model to use when neither ``model=`` nor ``request.model`` is set.
            Two forms:
            - Explicit string, e.g. ``fallback_model="gpt-4o-mini"`` — used
              verbatim (tier aliases are resolved normally).
            - ``"cheapest"`` — automatically resolves to the cheapest tier
              alias available on this provider
              (``super_cheap`` → ``cheap`` → ``medium``, in that order).
            When ``None`` (default) and no model is configured, a
            ``ValueError`` is raised with guidance on how to fix it.
        strict_native_tools:
            When ``True``, requesting an unsupported :class:`NativeTool`
            raises :class:`UnsupportedNativeToolError`.  When ``None``
            (default) the class-level :attr:`strict_native_tools`
            attribute is used (typically ``False``).
        **kwargs:
            Forwarded verbatim to :meth:`_init_client`.
        """
        if api_key is not None and not api_key.strip():
            raise ValueError(
                f"{self.__class__.__name__}: api_key must not be an empty or "
                "whitespace-only string. Pass None to read from the environment "
                "variable, or provide a valid key."
            )
        self.api_key = api_key
        # Store the user-supplied model separately so _resolve_model can
        # distinguish "user didn't pass a model" from "class default applies".
        # self.model is the effective value for backward-compat reads (e.g.
        # executor.model); _resolve_model uses _user_model to decide when to
        # consult fallback_model before falling through to default_model.
        self._user_model: str | None = model
        self.model = model or self.default_model
        self.fallback_model = fallback_model
        if strict_native_tools is not None:
            # Per-instance override of the class-level default.
            self.strict_native_tools = bool(strict_native_tools)
        self._init_client(**kwargs)

    def _init_client(self, **kwargs: Any) -> None:  # noqa: B027
        """Initialise the provider SDK client.

        Override this to create your SDK client and store it on ``self``::

            def _init_client(self, **kwargs) -> None:
                import my_sdk
                self._client = my_sdk.Client(api_key=self.api_key, **kwargs)

        Called once at construction time.  Default implementation is a no-op.
        """
        pass

    def is_retryable(self, exc: BaseException) -> bool | None:
        """Classify a provider exception as retryable, non-retryable, or defer.

        The :class:`~lazybridge.core.executor.Executor` consults this hook
        before falling back to its generic status/string heuristic.  Override
        when the provider SDK raises structured exception types that encode
        retry semantics more precisely than HTTP status codes alone — for
        example a rate-limit exception that carries a ``retry_after`` attribute
        distinguishing "back off" (retryable) from "quota exhausted" (not).

        Return values:
          * ``True`` — retry with backoff.
          * ``False`` — do not retry; surface the exception.
          * ``None`` — no opinion; Executor falls back to its generic
            classifier (``core.executor._is_retryable``) that matches
            ``status_code in {429, 5xx}`` and common transient-error strings.

        Default implementation returns ``None`` so built-in providers fall
        through to the generic path with no behaviour change.
        """
        return None

    @abstractmethod
    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Execute a synchronous completion and return a unified response.

        Parameters
        ----------
        request:
            Fully assembled :class:`~lazybridge.core.types.CompletionRequest`.
            Treat as **read-only** — do not mutate.

        Returns
        -------
        CompletionResponse
            At minimum, ``content`` must be set to the model's text reply.
            Populate ``usage``, ``model``, ``tool_calls``, ``stop_reason``
            when available.  Set ``raw`` to the original SDK response object
            to allow callers to access provider-specific fields.

        Raises
        ------
        Any exception from your SDK is acceptable — LazyBridge propagates them
        as-is and handles retry logic in :class:`~lazybridge.core.executor.Executor`.
        """
        ...

    @abstractmethod
    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        """Stream a completion, yielding :class:`~lazybridge.core.types.StreamChunk` objects.

        The final chunk **must** have ``is_final=True`` and ``stop_reason`` set.
        Token usage should be reported on the final chunk when available.

        Parameters
        ----------
        request:
            Same as :meth:`complete`. Treat as read-only.

        Yields
        ------
        StreamChunk
            Intermediate chunks: ``delta`` contains the new text fragment.
            Final chunk: ``is_final=True``, ``stop_reason`` set, ``usage`` populated.

        Example skeleton::

            def stream(self, request):
                for raw_chunk in self._client.stream(...):
                    yield StreamChunk(delta=raw_chunk.text)
                yield StreamChunk(
                    delta="",
                    stop_reason="end_turn",
                    is_final=True,
                    usage=UsageStats(input_tokens=..., output_tokens=...),
                )
        """
        ...

    @abstractmethod
    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        """Async version of :meth:`complete`.

        Semantics and return contract are identical. Use ``await`` for all
        blocking operations — never call ``time.sleep`` or blocking I/O here.
        """
        ...

    @abstractmethod
    def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        """Async streaming generator — async version of :meth:`stream`.

        Implement as an ``async def`` generator::

            async def astream(self, request):
                async for raw_chunk in self._client.astream(...):
                    yield StreamChunk(delta=raw_chunk.text)
                yield StreamChunk(stop_reason="end_turn", is_final=True, usage=...)

        The same final-chunk contract as :meth:`stream` applies.
        """
        ...

    # ------------------------------------------------------------------
    # Helpers — stable, callable from subclasses
    # ------------------------------------------------------------------

    #: Tier aliases.  Each provider populates this with the concrete
    #: model it considers "top"/"expensive"/"medium"/"cheap"/
    #: "super_cheap" (the :data:`Tier` vocabulary) so users can write
    #: ``Agent.from_provider("anthropic", tier="cheap")`` without
    #: hard-coding preview / date-pinned names.  A string not in this
    #: dict is treated as a literal model name (passthrough).
    _TIER_ALIASES: dict[str, str] = {}

    #: Optional fallback chain.  If a concrete model in the key is
    #: unavailable (provider returns 404 / "model not found"), the
    #: provider *would* try each model in the list in order.
    #:
    #: .. warning::
    #:
    #:    **Not implemented.**  ``_FALLBACKS`` is currently dead data:
    #:    no code path reads it at runtime.  The provider price-table
    #:    and tier-alias tests (``tests/unit/test_provider_static_paths.py``)
    #:    keep the structure honest so the data is ready when a retry
    #:    path lands, but users should not rely on automatic fallback
    #:    today.  An explicit ``model=`` argument or wrapping
    #:    ``Agent.run(...)`` in your own try/except is the only
    #:    supported resilience pattern in 0.7.9.
    _FALLBACKS: dict[str, list[str]] = {}

    def _cheapest_tier(self) -> str | None:
        """Return the concrete model for the cheapest available tier alias.

        Walks ``super_cheap`` → ``cheap`` → ``medium`` and returns the first
        hit.  Returns ``None`` if none of those tiers are defined on this
        provider.
        """
        for tier in ("super_cheap", "cheap", "medium"):
            model = self._TIER_ALIASES.get(tier)
            if model:
                return model
        return None

    def _resolve_model(self, request: CompletionRequest) -> str:
        """Return the effective model for this request.

        Resolution order:
        1. ``request.model``         — per-call override (highest priority)
        2. ``self._user_model``      — explicitly passed at construction time
        3. ``self.fallback_model``   — explicit fallback or ``"cheapest"``
        4. ``self.default_model``    — class-level default (free/cheap providers only)
        5. raise ``ValueError``      — no silent expensive surprises

        Tier aliases (``"top"``, ``"cheap"``, ``"super_cheap"`` etc.) are
        resolved at the end via ``_TIER_ALIASES``; everything else passes
        through verbatim.  Always call this method instead of reading
        ``self.model`` directly so per-request overrides and tier tables
        are respected.
        """
        # 1. Per-request override.
        if request.model:
            return self._TIER_ALIASES.get(request.model, request.model)

        # 2. User-supplied model at construction time.  getattr for backward
        # compat with __new__-bypassed instances (tests, legacy subclasses)
        # that never called __init__ and lack _user_model; fall back to
        # self.model which those callers set directly.  _user_model=None
        # means "user passed no model= argument" and is distinct from
        # "attribute absent", so fallback_model is correctly consulted even
        # when a class default_model exists (e.g. LMStudioProvider with
        # fallback_model="cheapest").
        user_model: str | None = getattr(self, "_user_model", self.model)
        if user_model:
            return self._TIER_ALIASES.get(user_model, user_model)

        # 3. fallback_model — opt-in safety net, checked BEFORE default_model
        # so LMStudioProvider(fallback_model="cheapest") takes effect even
        # when the class defines a default_model.
        fb: str | None = getattr(self, "fallback_model", None)
        if fb == "cheapest":
            cheapest = self._cheapest_tier()
            if cheapest is None:
                raise ValueError(
                    f"{type(self).__name__}: fallback_model='cheapest' requested "
                    f"but no cheap/super_cheap/medium tier alias is defined.\n"
                    f"  Available tier aliases: {sorted(self._TIER_ALIASES)}"
                )
            return self._TIER_ALIASES.get(cheapest, cheapest)
        if fb:
            return self._TIER_ALIASES.get(fb, fb)

        # 4. Class-level default (free/cheap providers only; paid providers
        # set default_model = None to force explicit selection).
        default = self.default_model
        if default:
            return self._TIER_ALIASES.get(default, default)

        raise ValueError(
            f"{type(self).__name__}: no model configured.\n"
            f"  Fix options:\n"
            f"  1. Pass model= explicitly:      LLMEngine('gpt-4o-mini')\n"
            f"  2. Set on the provider:          OpenAIProvider(model='gpt-4o-mini')\n"
            f"  3. Explicit fallback:            OpenAIProvider(fallback_model='gpt-4o-mini')\n"
            f"  4. Cheapest-tier fallback:       OpenAIProvider(fallback_model='cheapest')\n"
            f"  Available tier aliases: {sorted(self._TIER_ALIASES)} "
            f"(or pass an explicit model id)."
        )

    def _check_native_tools(self, tools: list[NativeTool]) -> list[NativeTool]:
        """Filter ``tools`` to only those declared in ``supported_native_tools``.

        By default (``strict_native_tools=False`` — see ``__init__``):
        unsupported tools are dropped with a :class:`UserWarning` so an
        accidental mis-routing (e.g. asking DeepSeek for ``WEB_SEARCH``)
        doesn't silently corrupt a request.  When ``strict_native_tools``
        is set, the same condition raises :class:`UnsupportedNativeToolError`
        — useful in production where a missing capability should fail
        loud rather than degrade silently.

        Either way the warning / error message lists which native tools
        the provider DOES support, so the user can see alternatives
        without hunting through documentation.
        """
        supported_set = self.supported_native_tools
        supported_list: list[NativeTool] = []
        for tool in tools:
            if tool in supported_set:
                supported_list.append(tool)
                continue
            available = (
                ", ".join(sorted(t.value for t in supported_set))
                if supported_set
                else "(none — this provider does not implement any server-side native tools)"
            )
            msg = (
                f"{self.__class__.__name__} does not support native tool "
                f"{tool.value!r}.  Supported: {available}.  Use "
                f"plain function tools instead, or switch to a provider "
                f"that supports this capability."
            )
            if getattr(self, "strict_native_tools", False):
                raise UnsupportedNativeToolError(msg)
            warnings.warn(msg, UserWarning, stacklevel=3)
        return supported_list

    def _compute_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
    ) -> float | None:
        """Return estimated cost in USD, or ``None`` if unknown.

        Override to enable cost tracking in ``CompletionResponse.usage.cost_usd``.

        ``cached_input_tokens`` is the subset of ``input_tokens`` served from
        the provider's prompt cache.  Override callers that don't model
        cache pricing can ignore the parameter (it defaults to 0)::

            _PRICES = {"my-model-v1": (0.50, 1.50)}  # ($/1M input, $/1M output)

            def _compute_cost(self, model, input_tokens, output_tokens, cached_input_tokens=0):
                for key, (inp, out) in self._PRICES.items():
                    if key in model:
                        return (input_tokens * inp + output_tokens * out) / 1_000_000
                return None
        """
        del cached_input_tokens  # base default ignores cache pricing
        return None

    def get_default_max_tokens(self, model: str | None = None) -> int:
        """Return the default ``max_tokens`` cap for the given model.

        Override when your model has a limit lower or higher than 4096.
        LazyBridge calls this when ``max_tokens`` is not set explicitly.
        """
        return 4096
