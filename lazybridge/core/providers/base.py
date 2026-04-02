"""Abstract base provider for uniAI."""

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
    """Abstract base class for all LLM providers."""

    default_model: str = ""
    supported_native_tools: frozenset[NativeTool] = frozenset()

    def __init__(self, api_key: str | None = None, model: str | None = None, **kwargs):
        self.api_key = api_key
        self.model = model or self.default_model
        self._init_client(**kwargs)

    def _init_client(self, **kwargs) -> None:
        """Initialize the provider SDK client. Override in subclasses."""
        pass

    @abstractmethod
    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Send a completion request and return a unified response."""
        ...

    @abstractmethod
    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        """Stream a completion request, yielding chunks."""
        ...

    @abstractmethod
    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        """Async version of complete()."""
        ...

    @abstractmethod
    def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        """Async version of stream(). Implementations should be async generators."""
        ...

    def _check_native_tools(self, tools: list[NativeTool]) -> list[NativeTool]:
        """Filter and warn about unsupported native tools for this provider."""
        supported = []
        for tool in tools:
            if tool in self.supported_native_tools:
                supported.append(tool)
            else:
                warnings.warn(
                    f"{self.__class__.__name__} does not support native tool '{tool.value}'. Skipping.",
                    UserWarning,
                    stacklevel=3,
                )
        return supported

    def _resolve_model(self, request: CompletionRequest) -> str:
        return request.model or self.model or self.default_model

    def _compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float | None:
        """Return cost in USD, or None if the model is not in the price table.

        Override in concrete providers that maintain a ``_PRICE_TABLE``.
        """
        return None

    def get_default_max_tokens(self, model: str | None = None) -> int:
        return 4096
