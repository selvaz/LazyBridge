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

from lazybridge.core.types import (
    CompletionRequest,
    CompletionResponse,
    NativeTool,
    StreamChunk,
)


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

    default_model: str = ""
    """Class-level default model identifier.  Used when neither the request
    nor the constructor ``model=`` argument specifies a model."""

    supported_native_tools: frozenset[NativeTool] = frozenset()
    """Declare which :class:`~lazybridge.core.types.NativeTool` values this
    provider supports (e.g. ``frozenset({NativeTool.WEB_SEARCH})``).
    Unsupported tools requested by the user are filtered and warned —
    or raised, when ``strict_native_tools=True`` is set on construction."""

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
        strict_native_tools: bool | None = None,
        **kwargs,
    ):
        """Initialise the provider.

        Parameters
        ----------
        api_key:
            Provider API key.  If ``None``, ``_init_client`` should read it from
            an environment variable (standard pattern for all built-in providers).
        model:
            Model identifier to use for all requests.  Falls back to
            ``default_model`` when ``None``.
        strict_native_tools:
            When ``True``, requesting an unsupported :class:`NativeTool`
            raises :class:`UnsupportedNativeToolError`.  When ``None``
            (default) the class-level :attr:`strict_native_tools`
            attribute is used (typically ``False``).
        **kwargs:
            Forwarded verbatim to :meth:`_init_client`.
        """
        self.api_key = api_key
        self.model = model or self.default_model
        if strict_native_tools is not None:
            # Per-instance override of the class-level default.
            self.strict_native_tools = bool(strict_native_tools)
        self._init_client(**kwargs)

    def _init_client(self, **kwargs) -> None:  # noqa: B027
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
    #: "super_cheap" so users can write ``Agent("anthropic",
    #: model="cheap")`` without hard-coding preview / date-pinned names.
    #: A string not in this dict is treated as a literal model name
    #: (passthrough).
    _TIER_ALIASES: dict[str, str] = {}

    #: Optional fallback chain.  If a concrete model in the key is
    #: unavailable (provider returns 404 / "model not found"), the
    #: provider should try each model in the list in order.  Wiring
    #: into a retry path is per-provider and not yet active; the tables
    #: are populated so the data is ready when that path lands.
    _FALLBACKS: dict[str, list[str]] = {}

    def _resolve_model(self, request: CompletionRequest) -> str:
        """Return the effective model: request → instance → class default.

        Tier aliases (``"top"``, ``"cheap"`` etc.) are resolved here via
        ``_TIER_ALIASES``; everything else is a passthrough.  Always use
        this inside ``complete`` / ``stream`` instead of reading
        ``self.model`` directly so that per-request overrides and
        per-provider tier tables are respected.
        """
        name = request.model or self.model or self.default_model
        # Empty / None defaults to class default.
        if not name:
            return name
        # Tier alias?  Resolve to the concrete model.
        resolved = self._TIER_ALIASES.get(name, name)
        return resolved

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

    def _compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float | None:
        """Return estimated cost in USD, or ``None`` if unknown.

        Override to enable cost tracking in ``CompletionResponse.usage.cost_usd``::

            _PRICES = {"my-model-v1": (0.50, 1.50)}  # ($/1M input, $/1M output)

            def _compute_cost(self, model, input_tokens, output_tokens):
                for key, (inp, out) in self._PRICES.items():
                    if key in model:
                        return (input_tokens * inp + output_tokens * out) / 1_000_000
                return None
        """
        return None

    def get_default_max_tokens(self, model: str | None = None) -> int:
        """Return the default ``max_tokens`` cap for the given model.

        Override when your model has a limit lower or higher than 4096.
        LazyBridge calls this when ``max_tokens`` is not set explicitly.
        """
        return 4096
