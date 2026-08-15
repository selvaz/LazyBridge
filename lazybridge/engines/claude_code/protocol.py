"""Small SDK-facing protocol used by the prototype and its fake transport.

Keeping this boundary local makes all deterministic tests independent from a
Claude login and lets the future LazyBridge integration load the real Agent SDK
only when the optional extra is installed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class McpTool:
    """A tool exposed to Claude through the in-process MCP boundary."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[Any]]


@dataclass(frozen=True)
class ClaudeSdkOptions:
    """Small, engine-relevant subset of the Agent SDK options."""

    cwd: str | None = None
    model: str | None = None
    fallback_model: str | None = None
    reasoning_effort: str | None = None
    thinking: dict[str, Any] | None = None
    system_prompt: str | None = None
    max_turns: int = 20
    resume: str | None = None
    allowed_tools: tuple[str, ...] = ()
    builtin_tools: tuple[str, ...] = ()
    file_roots: tuple[str, ...] = ()
    mcp_server_name: str = "lazybridge"
    mcp_tools: tuple[McpTool, ...] = ()
    include_partial_messages: bool = False
    #: Native structured output, in the Messages API shape the Agent SDK
    #: expects: ``{"type": "json_schema", "schema": {...}}``. The SDK turns
    #: this into the CLI's ``--json-schema`` flag and returns the parsed
    #: object on ``ResultMessage.structured_output``.
    output_format: dict[str, Any] | None = None


@dataclass(frozen=True)
class ClaudeSdkResult:
    """Final result returned by a Claude Agent SDK conversation."""

    text: str
    session_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str | None = None


@dataclass(frozen=True)
class ClaudeSdkStreamEvent:
    """One text chunk or the final metadata event from an SDK stream."""

    text: str = ""
    session_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    final: bool = False


class ClaudeSdkClient(Protocol):
    """Minimal boundary implemented by the real and fake SDK clients.

    ``attachments`` are Anthropic content blocks (e.g.
    ``{"type": "image", "source": {"type": "base64", ...}}``) appended after
    the prompt's text block; empty for a text-only run.
    """

    async def run(
        self, prompt: str, *, options: ClaudeSdkOptions, attachments: tuple[dict[str, Any], ...] = ()
    ) -> ClaudeSdkResult: ...

    # Deliberately not ``async def``: implementations are async *generator*
    # functions, so calling ``stream(...)`` returns the ``AsyncIterator``
    # directly rather than a coroutine that resolves to one. Declaring this
    # ``async`` would type it as ``Coroutine[Any, Any, AsyncIterator[...]]``,
    # which no async-generator implementation actually satisfies.
    def stream(
        self, prompt: str, *, options: ClaudeSdkOptions, attachments: tuple[dict[str, Any], ...] = ()
    ) -> AsyncIterator[ClaudeSdkStreamEvent]: ...
