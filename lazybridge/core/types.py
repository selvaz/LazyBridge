"""Shared types and data structures for lazybridge."""

from __future__ import annotations

import base64
import mimetypes
import warnings
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ContentType(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"


# Magic-byte signatures used by ``_detect_image_mime`` to recover a
# media-type when the caller passes raw bytes without one.  The full
# stdlib ``imghdr`` module is deprecated in 3.13 and gone in 3.14, so
# we keep our own minimal table — covers the formats every provider
# accepts inline (PNG, JPEG, GIF, WebP).
_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


_VALID_IMAGE_MIMES: frozenset[str] = frozenset(
    {
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
        "image/bmp",
        "image/tiff",
    }
)

_VALID_AUDIO_MIMES: frozenset[str] = frozenset(
    {
        "audio/wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/flac",
        "audio/ogg",
        "audio/webm",
        "audio/aac",
        "audio/mp4",
        "audio/x-wav",
        "audio/x-mpeg",
    }
)


def _detect_image_mime(data: bytes) -> str | None:
    """Best-effort image media-type from the leading magic bytes.

    Returns ``None`` when the signature is unrecognised; callers either
    fall back to a sensible default or raise.  WebP needs a two-step
    check because the magic is split: ``RIFF....WEBP`` (the four-byte
    file size sits between the markers).
    """
    for magic, mime in _IMAGE_MAGIC:
        if data.startswith(magic):
            return mime
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _detect_audio_mime(data: bytes) -> str | None:
    """Best-effort audio media-type from the leading magic bytes.

    Covers the four formats every multimodal provider accepts inline:
    WAV (RIFF...WAVE), MP3 (ID3 tag or 0xFF frame sync), FLAC, OGG.
    Returns ``None`` for anything else — caller falls back to a default.
    """
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"
    if data.startswith(b"ID3") or (len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return "audio/mpeg"
    if data.startswith(b"fLaC"):
        return "audio/flac"
    if data.startswith(b"OggS"):
        return "audio/ogg"
    return None


class NativeTool(StrEnum):
    """Provider-native server-side tools (run on provider infrastructure)."""

    WEB_SEARCH = "web_search"
    CODE_EXECUTION = "code_execution"
    FILE_SEARCH = "file_search"  # OpenAI
    COMPUTER_USE = "computer_use"
    IMAGE_GENERATION = "image_generation"  # OpenAI Responses API (gpt-image-2 family)
    GOOGLE_SEARCH = "google_search"  # Gemini grounding
    GOOGLE_MAPS = "google_maps"  # Gemini grounding


@dataclass
class TextContent:
    text: str
    type: ContentType = ContentType.TEXT


@dataclass
class ImageContent:
    url: str | None = None
    base64_data: str | None = None
    media_type: str = "image/jpeg"
    type: ContentType = ContentType.IMAGE

    def __post_init__(self) -> None:
        if self.media_type not in _VALID_IMAGE_MIMES:
            warnings.warn(
                f"ImageContent: unrecognised media_type {self.media_type!r}. "
                f"Known types: {sorted(_VALID_IMAGE_MIMES)}. "
                "The request may be rejected by some providers.",
                UserWarning,
                stacklevel=3,
            )

    @classmethod
    def from_url(cls, url: str, *, media_type: str | None = None) -> ImageContent:
        # ``mimetypes.guess_type`` doesn't strip URL schemes — parse the
        # path component first so ``https://x.com/cat.png`` correctly
        # resolves to ``image/png`` instead of falling through to the
        # JPEG default.
        from urllib.parse import urlparse

        path = urlparse(url).path or url
        guessed = media_type or mimetypes.guess_type(path)[0] or "image/jpeg"
        return cls(url=url, media_type=guessed)

    @classmethod
    def from_path(cls, path: str | Path, *, media_type: str | None = None) -> ImageContent:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"image not found: {p}")
        data = p.read_bytes()
        guessed = media_type or _detect_image_mime(data) or mimetypes.guess_type(p.name)[0] or "image/jpeg"
        return cls(base64_data=base64.b64encode(data).decode("ascii"), media_type=guessed)

    @classmethod
    def from_bytes(cls, data: bytes, media_type: str | None = None) -> ImageContent:
        guessed = media_type or _detect_image_mime(data) or "image/jpeg"
        return cls(base64_data=base64.b64encode(data).decode("ascii"), media_type=guessed)

    @classmethod
    def from_data_uri(cls, data_uri: str) -> ImageContent:
        """Parse ``data:image/png;base64,<...>`` style URIs."""
        if not data_uri.startswith("data:"):
            raise ValueError("expected a data: URI")
        header, _, body = data_uri[5:].partition(",")
        if not body:
            raise ValueError("malformed data URI: missing payload")
        media_type, _, encoding = header.partition(";")
        media_type = media_type or "image/jpeg"
        if encoding == "base64":
            return cls(base64_data=body, media_type=media_type)
        return cls(base64_data=base64.b64encode(body.encode()).decode("ascii"), media_type=media_type)


@dataclass
class AudioContent:
    """Audio attachment for multimodal LLM input.

    Provider support varies by model — see
    :meth:`BaseProvider.supports_audio`.  Anthropic / OpenAI accept
    base64 only; Google Gemini accepts both URL and base64.
    """

    url: str | None = None
    base64_data: str | None = None
    media_type: str = "audio/wav"
    type: ContentType = ContentType.AUDIO

    def __post_init__(self) -> None:
        if self.media_type not in _VALID_AUDIO_MIMES:
            warnings.warn(
                f"AudioContent: unrecognised media_type {self.media_type!r}. "
                f"Known types: {sorted(_VALID_AUDIO_MIMES)}. "
                "The request may be rejected by some providers.",
                UserWarning,
                stacklevel=3,
            )

    @classmethod
    def from_url(cls, url: str, *, media_type: str | None = None) -> AudioContent:
        from urllib.parse import urlparse

        path = urlparse(url).path or url
        guessed = media_type or mimetypes.guess_type(path)[0] or "audio/wav"
        return cls(url=url, media_type=guessed)

    @classmethod
    def from_path(cls, path: str | Path, *, media_type: str | None = None) -> AudioContent:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"audio not found: {p}")
        data = p.read_bytes()
        guessed = media_type or _detect_audio_mime(data) or mimetypes.guess_type(p.name)[0] or "audio/wav"
        return cls(base64_data=base64.b64encode(data).decode("ascii"), media_type=guessed)

    @classmethod
    def from_bytes(cls, data: bytes, media_type: str | None = None) -> AudioContent:
        guessed = media_type or _detect_audio_mime(data) or "audio/wav"
        return cls(base64_data=base64.b64encode(data).decode("ascii"), media_type=guessed)

    @classmethod
    def from_data_uri(cls, data_uri: str) -> AudioContent:
        """Parse ``data:audio/flac;base64,<...>`` style URIs."""
        if not data_uri.startswith("data:"):
            raise ValueError("expected a data: URI")
        header, _, body = data_uri[5:].partition(",")
        if not body:
            raise ValueError("malformed data URI: missing payload")
        media_type, _, encoding = header.partition(";")
        media_type = media_type or "audio/wav"
        if encoding == "base64":
            return cls(base64_data=body, media_type=media_type)
        return cls(base64_data=base64.b64encode(body.encode()).decode("ascii"), media_type=media_type)


@dataclass
class ToolUseContent:
    id: str
    name: str
    input: dict[str, Any]
    thought_signature: Any = None  # Gemini thinking models: raw SDK Part, re-emitted as-is
    type: ContentType = ContentType.TOOL_USE


@dataclass
class ToolResultContent:
    tool_use_id: str
    content: str | list[Any]
    tool_name: str | None = None  # function name — required for Gemini function_response
    is_error: bool = False
    type: ContentType = ContentType.TOOL_RESULT


@dataclass
class ThinkingContent:
    thinking: str
    type: ContentType = ContentType.THINKING


ContentBlock = TextContent | ImageContent | AudioContent | ToolUseContent | ToolResultContent | ThinkingContent


def _coerce_image(x: Any) -> ImageContent:
    """Best-effort coerce ``x`` into an :class:`ImageContent`.

    Accepted forms:

    * ``ImageContent`` — passthrough.
    * ``str`` — ``http(s)://`` → URL; ``data:image/...`` → data URI; else
      treated as a filesystem path (must exist).
    * ``Path`` — filesystem path.
    * ``bytes`` — raw bytes; media-type sniffed from magic bytes.
    * ``dict`` — unpacked as ``ImageContent(**x)``.

    Raises :class:`TypeError` for anything else.  Use this at API
    boundaries (``Agent.run(images=[...])``) so callers can pass the
    most natural form for their context.
    """
    if isinstance(x, ImageContent):
        return x
    if isinstance(x, Path):
        return ImageContent.from_path(x)
    if isinstance(x, str):
        if x.startswith(("http://", "https://")):
            return ImageContent.from_url(x)
        if x.startswith("data:"):
            return ImageContent.from_data_uri(x)
        return ImageContent.from_path(x)
    if isinstance(x, bytes):
        return ImageContent.from_bytes(x)
    if isinstance(x, dict):
        return ImageContent(**x)
    raise TypeError(f"cannot coerce {type(x).__name__!r} to ImageContent — pass a URL, Path, bytes, or ImageContent")


def _coerce_audio(x: Any) -> AudioContent:
    """Best-effort coerce ``x`` into an :class:`AudioContent`.

    See :func:`_coerce_image` for the accepted shapes.  Audio URL
    handling is provider-dependent (Google supports it, Anthropic /
    OpenAI do not — see :meth:`BaseProvider.supports_audio`).
    """
    if isinstance(x, AudioContent):
        return x
    if isinstance(x, Path):
        return AudioContent.from_path(x)
    if isinstance(x, str):
        if x.startswith(("http://", "https://")):
            return AudioContent.from_url(x)
        return AudioContent.from_path(x)
    if isinstance(x, bytes):
        return AudioContent.from_bytes(x)
    if isinstance(x, dict):
        return AudioContent(**x)
    raise TypeError(f"cannot coerce {type(x).__name__!r} to AudioContent — pass a URL, Path, bytes, or AudioContent")


@dataclass
class Message:
    role: Role
    content: str | list[ContentBlock]

    def to_text(self) -> str:
        """Extract plain text from message content."""
        if isinstance(self.content, str):
            return self.content
        parts = []
        for block in self.content:
            if isinstance(block, TextContent):
                parts.append(block.text)
            elif isinstance(block, ThinkingContent):
                parts.append(block.thinking)
        return "\n".join(parts)

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role=Role.USER, content=text)

    @classmethod
    def assistant(cls, text: str) -> Message:
        return cls(role=Role.ASSISTANT, content=text)

    @classmethod
    def system(cls, text: str) -> Message:
        return cls(role=Role.SYSTEM, content=text)


@dataclass
class ToolDefinition:
    """Unified tool/function definition (JSON Schema based)."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object
    strict: bool = False


class StructuredOutputParseError(Exception):
    """Raised when model output cannot be parsed as valid JSON."""

    def __init__(self, message: str, *, provider: str | None = None, model: str | None = None, raw: str | None = None):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.raw = raw


class StructuredOutputValidationError(Exception):
    """Raised when parsed JSON fails Pydantic or schema validation."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        raw: str | None = None,
        parsed: Any = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.raw = raw
        self.parsed = parsed


@dataclass
class StructuredOutputConfig:
    """Config for constrained JSON output."""

    schema: type | dict[str, Any]  # Pydantic model class or raw JSON schema dict
    strict: bool = True  # pass strict=True to APIs that support it


@dataclass
class ThinkingConfig:
    """Reasoning/thinking configuration."""

    enabled: bool = True
    display: str | None = None  # Anthropic 4.6+: e.g. "omitted" to hide thinking text in streams
    effort: str = "high"  # "low"|"medium"|"high"|"xhigh" — maps to provider equivalent
    # For older Anthropic models only (budget_tokens deprecated on claude-*-4-6):
    budget_tokens: int | None = None


@dataclass
class SkillsConfig:
    """Anthropic Skills — server-side domain-expert packages."""

    skills: list[str]  # e.g. ["pdf", "excel", "powerpoint", "word"]


@dataclass
class CacheConfig:
    """Mark the static prefix of a request (system prompt + tool
    definitions) as cacheable.

    Providers with explicit prompt caching (Anthropic) need a
    ``cache_control`` marker on the last block of each cached
    segment; this makes it a one-flag opt-in instead of asking
    callers to hand-craft provider-specific content lists.

    Provider-specific behaviour:

    * **Anthropic** — the system prompt is upgraded from a string to a
      ``[{type: "text", text, cache_control}]`` block; a cache
      breakpoint is also placed on the last tool definition if tools
      are present.  Cache hits cost ~10% of input tokens; writes cost
      ~25% more.  TTL options: ``"5m"`` (default) or ``"1h"``.
    * **OpenAI** — automatic for system prompts >1024 tokens; no
      user-visible opt-in is required.  This config is a no-op but
      accepted for forward-compat.
    * **Google Gemini** — explicit Context Caching uses a different
      lifecycle (create a cache resource, reference by name).  Not
      auto-wired from this config; pass via ``extra`` if needed.
    * **DeepSeek** — automatic; no-op.
    """

    enabled: bool = True
    #: Anthropic-only: ``"5m"`` (default) or ``"1h"``.  Other providers
    #: ignore this field.
    ttl: str = "5m"


#: Sentinel used by config-object unpacking to detect "user passed a
#: flat kwarg explicitly" vs "left it at the default".  Internal only —
#: never exposed in public signatures.
_UNSET: Any = object()


@dataclass
class ResilienceConfig:
    """Bundle of reliability / performance knobs shareable across Agents.

    Wraps the resilience kwargs on ``Agent`` so a fleet of agents in a
    production pipeline can share a single retry / timeout / cache
    policy instead of copy-pasting seven kwargs at every call site.

    Flat ``Agent`` kwargs still win when both are passed — the config
    is the default, an explicit kwarg is the override.
    """

    timeout: float | None = None
    max_retries: int = 3
    retry_delay: float = 1.0
    cache: bool | CacheConfig = False
    max_output_retries: int = 2
    output_validator: Any = None  # Callable[[Any], Any] | None
    #: Forward-ref to ``Agent`` — typed ``Any`` to avoid a circular import.
    #: Actual validation happens inside ``Agent.__init__``.
    fallback: Any = None


@dataclass
class ObservabilityConfig:
    """Bundle of identity / tracing knobs shareable across Agents.

    ``session`` is the single biggest win: binding a fleet of agents to
    the same ``Session`` in one object beats threading it through every
    constructor call.
    """

    verbose: bool = False
    session: Any = None  # Session | None
    name: str | None = None
    description: str | None = None


@dataclass
class AgentRuntimeConfig:
    """Composite — carries both resilience and observability.

    Convenience for configuration injection: one object passed to every
    ``Agent`` in a factory instead of two.
    """

    resilience: ResilienceConfig | None = None
    observability: ObservabilityConfig | None = None


#: Provider-agnostic meta-keywords accepted as ``tool_choice``.  Anything
#: outside this set is interpreted as a tool NAME and validated against
#: the ``tools`` list on the request, so typos fail fast at request
#: construction instead of being silently ignored by the provider API
#: (or worse, triggering a cryptic server-side error several RTTs in).
_TOOL_CHOICE_KEYWORDS: frozenset[str] = frozenset({"auto", "required", "none", "any"})


@dataclass
class CompletionRequest:
    """Unified request object passed to any provider."""

    messages: list[Message]
    model: str | None = None
    system: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    tools: list[ToolDefinition] = field(default_factory=list)
    tool_choice: str | None = None  # "auto" | "required" | "none" | "any" | tool_name
    native_tools: list[NativeTool] = field(default_factory=list)
    structured_output: StructuredOutputConfig | None = None
    thinking: ThinkingConfig | None = None
    skills: SkillsConfig | None = None  # Anthropic only
    #: Opt-in prompt caching for the static prefix (system + tools).
    #: ``None`` = caching disabled; ``CacheConfig()`` = default enabled.
    #: See :class:`CacheConfig` for per-provider semantics.
    cache: CacheConfig | None = None
    stream: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate ``tool_choice`` up-front: non-keyword values must
        # name an actual tool in ``self.tools``.  Without this check a
        # typo would pass silently to the provider API, where it
        # either errored cryptically or was ignored.
        if self.tool_choice is None:
            return
        if self.tool_choice in _TOOL_CHOICE_KEYWORDS:
            return
        tool_names = {t.name for t in self.tools}
        if self.tool_choice in tool_names:
            return
        # Either no tools at all, or the named tool isn't in the list.
        if not tool_names:
            raise ValueError(
                f"CompletionRequest: tool_choice={self.tool_choice!r} names a "
                f"specific tool, but this request has no tools.  Either pass "
                f"tools=[...] or use one of "
                f"{sorted(_TOOL_CHOICE_KEYWORDS)}."
            )
        raise ValueError(
            f"CompletionRequest: tool_choice={self.tool_choice!r} does not "
            f"match any tool in tools=.  Known tools: "
            f"{sorted(tool_names)}.  Pick one of those names, or use one of "
            f"{sorted(_TOOL_CHOICE_KEYWORDS)} for the provider's default "
            f"behaviour."
        )


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    thought_signature: Any = None  # Gemini thinking models: raw SDK Part, re-emitted as-is


@dataclass
class UsageStats:
    """Token-usage and cost telemetry for one request.

    ``thinking_tokens`` reports the *reasoning* portion of the model's
    output for providers that expose it.  It is informational and may
    overlap with ``output_tokens`` depending on the provider:

    * **OpenAI**: ``output_tokens`` (== API ``completion_tokens``) is
      the *total* output INCLUDING reasoning, and ``thinking_tokens``
      is the reasoning *subset* of that total.
    * **Anthropic / DeepSeek**: ``thinking_tokens`` is reported as a
      separate field; whether it is already inside ``output_tokens``
      varies by model and SDK version.

    ``cost_usd`` is computed off ``output_tokens`` (which the
    provider-side billing always uses) plus ``input_tokens`` /
    ``cached_input_tokens``, so cost is correct regardless of the
    overlap.  Don't sum ``output_tokens + thinking_tokens`` for a
    cost dashboard — it would double-count for OpenAI.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cached_input_tokens: int = 0
    cost_usd: float | None = None


@dataclass
class GroundingSource:
    """A web source returned by search grounding (Anthropic, OpenAI, Google Gemini)."""

    url: str
    title: str | None = None
    snippet: str | None = None


@dataclass
class CompletionResponse:
    """Unified response from any provider."""

    content: str
    thinking: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    model: str | None = None
    usage: UsageStats = field(default_factory=UsageStats)
    raw: Any = None  # Original provider response object
    parsed: Any = None  # Parsed structured output (Pydantic model instance or dict)
    validation_error: str | None = None  # Set when parse/validation fails; None on success
    validated: bool | None = (
        None  # None if structured output was not requested; True on success; False on parse/validation failure
    )
    grounding_sources: list[GroundingSource] = field(default_factory=list)  # Web search citations
    web_search_queries: list[str] = field(default_factory=list)  # Queries issued by the grounding tool
    search_entry_point: str | None = (
        None  # Google's rendered HTML attribution widget (required by ToS when displaying grounded results)
    )
    verify_log: list[str] = field(
        default_factory=list
    )  # Judge verdicts that rejected (in order); empty if approved first try or verify=None

    def raise_if_failed(self) -> None:
        """Raise an exception if structured output validation failed.

        Checks ``validation_error``: if set, raises the appropriate exception.
        No-op when ``validation_error`` is None (SO not requested, or succeeded).

        Raises:
            StructuredOutputParseError:      when validation_error starts with
                                             "JSON parse error".
            StructuredOutputValidationError: for all other validation failures.
        """
        if not self.validation_error:
            return
        if self.validation_error.startswith("JSON parse error"):
            raise StructuredOutputParseError(self.validation_error, raw=self.content)
        raise StructuredOutputValidationError(self.validation_error, raw=self.content, parsed=self.parsed)


@dataclass
class StreamChunk:
    """A single chunk from a streaming response."""

    delta: str = ""
    thinking_delta: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    usage: UsageStats | None = None
    is_final: bool = False
    parsed: Any = None  # populated on is_final=True if output_schema was set
    validation_error: str | None = None  # set on is_final=True if parse/validation failed
    validated: bool | None = (
        None  # None if structured output was not requested; True on success; False on parse/validation failure
    )
    grounding_sources: list[GroundingSource] = field(
        default_factory=list
    )  # Web search citations (populated on is_final=True)
    web_search_queries: list[str] = field(
        default_factory=list
    )  # Queries issued by the grounding tool (populated on is_final=True)
    search_entry_point: str | None = None  # Google's rendered HTML attribution widget (populated on is_final=True)


# ---------------------------------------------------------------------------
# Verifier Protocol — type-safe verify parameter for loop()/aloop()
# ---------------------------------------------------------------------------


@runtime_checkable
class Verifier(Protocol):
    """Protocol for verify judges used in ``loop()`` / ``aloop()``.

    Any object with a ``text(messages) -> str`` method satisfies this
    protocol — including ``Agent`` itself.  Alternatively, pass a
    plain ``Callable[[str, str], str]`` to the verify= parameter.

    Return a string starting with ``"approved"`` (case-insensitive) to
    accept the answer; anything else triggers a retry with the feedback.
    """

    def text(self, messages: str) -> str: ...
