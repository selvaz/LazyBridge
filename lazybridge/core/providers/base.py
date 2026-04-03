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

    from lazybridge import LazyAgent
    agent = LazyAgent(MyProvider(api_key="..."))
    print(agent.chat("hello").content)
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


class BaseProvider(ABC):
    """Stable abstract base class for all LLM providers.

    Subclass this to integrate any LLM backend with LazyBridge.
    Pass an instance directly as the first argument of ``LazyAgent``::

        agent = LazyAgent(MyProvider(api_key="..."))

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
    Unsupported tools requested by the user are silently filtered and warned."""

    def __init__(self, api_key: str | None = None, model: str | None = None, **kwargs):
        """Initialise the provider.

        Parameters
        ----------
        api_key:
            Provider API key.  If ``None``, ``_init_client`` should read it from
            an environment variable (standard pattern for all built-in providers).
        model:
            Model identifier to use for all requests.  Falls back to
            ``default_model`` when ``None``.
        **kwargs:
            Forwarded verbatim to :meth:`_init_client`.
        """
        self.api_key = api_key
        self.model = model or self.default_model
        self._init_client(**kwargs)

    def _init_client(self, **kwargs) -> None:
        """Initialise the provider SDK client.

        Override this to create your SDK client and store it on ``self``::

            def _init_client(self, **kwargs) -> None:
                import my_sdk
                self._client = my_sdk.Client(api_key=self.api_key, **kwargs)

        Called once at construction time.  Default implementation is a no-op.
        """
        pass

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

    def _resolve_model(self, request: CompletionRequest) -> str:
        """Return the effective model: request → instance → class default.

        Always use this inside ``complete`` / ``stream`` instead of reading
        ``self.model`` directly so that per-request overrides are respected.
        """
        return request.model or self.model or self.default_model

    def _check_native_tools(self, tools: list[NativeTool]) -> list[NativeTool]:
        """Filter ``tools`` to only those declared in ``supported_native_tools``.

        Unsupported tools are dropped with a :class:`UserWarning`.
        Call this at the start of ``complete`` / ``acomplete`` if your provider
        supports any native tools.
        """
        supported = []
        for tool in tools:
            if tool in self.supported_native_tools:
                supported.append(tool)
            else:
                warnings.warn(
                    f"{self.__class__.__name__} does not support native tool "
                    f"'{tool.value}'. Skipping.",
                    UserWarning,
                    stacklevel=3,
                )
        return supported

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
