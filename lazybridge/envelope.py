"""Envelope — the single data type flowing between all agents and engines.

Generic over its payload so ``Envelope[SearchResult]`` narrows the type
seen by mypy / pyright without changing runtime behaviour.  Writing
``Envelope(...)`` without a type parameter is equivalent to
``Envelope[Any](...)`` and remains the zero-friction default.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    pass

T = TypeVar("T")


class EnvelopeMetadata(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    model: str | None = None
    provider: str | None = None
    run_id: str | None = None
    # Aggregation buckets for nested agent-as-tool calls.  When Agent A
    # calls B through ``as_tool``, B's per-run metadata accumulates here
    # so the outer Envelope's metadata reflects total pipeline cost.
    nested_input_tokens: int = 0
    nested_output_tokens: int = 0
    nested_cost_usd: float = 0.0


class ErrorInfo(BaseModel):
    type: str
    message: str
    retryable: bool = False


class Envelope(BaseModel, Generic[T]):
    """Typed envelope carrying a payload of type ``T``.

    ``Envelope[str]`` → payload is a string.  ``Envelope[MyModel]`` →
    payload is an instance of ``MyModel``.  ``Envelope`` (no parameter)
    defaults to ``T = Any`` for maximum flexibility.

    Multimodal attachments (``images=`` / ``audio=``) ride alongside
    ``task`` and reach the LLMEngine's user-message builder verbatim.
    Steps in a Plan see them only on step 0; downstream steps receive
    upstream output (text), not the original attachments.
    """

    # ``ImageContent`` / ``AudioContent`` are dataclasses living in
    # ``core/types``.  Pydantic v2 needs the explicit allowance to
    # validate them as field values without us having to mirror the
    # whole type ladder as Pydantic models.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task: str | None = None
    context: str | None = None
    images: list[Any] | None = None  # list[ImageContent] — typed weakly to dodge runtime import
    audio: Any | None = None  # AudioContent — single clip per call (model convention)
    payload: T | None = None
    metadata: EnvelopeMetadata = Field(default_factory=EnvelopeMetadata)
    error: ErrorInfo | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def text(self) -> str:
        if self.payload is None:
            return ""
        if isinstance(self.payload, str):
            return self.payload
        if isinstance(self.payload, BaseModel):
            return self.payload.model_dump_json()
        return json.dumps(self.payload, default=str)

    def __str__(self) -> str:
        """Stringification falls through to :meth:`text`.

        Needed because tools that return an ``Envelope`` (agent-as-tool)
        cross back into content blocks expected by the LLM API, which
        serialise the value via ``str(...)``.  Without this, any such
        tool would produce ``"task=…  context=…"`` garbage instead of
        the agent's actual answer.
        """
        return self.text()

    @classmethod
    def from_task(cls, task: str, context: str | None = None) -> Envelope:
        return cls(task=task, context=context, payload=task)  # type: ignore[arg-type]

    @classmethod
    def error_envelope(cls, exc: BaseException, *, retryable: bool = False) -> Envelope:
        return cls(
            error=ErrorInfo(
                type=type(exc).__name__,
                message=str(exc),
                retryable=retryable,
            )
        )
