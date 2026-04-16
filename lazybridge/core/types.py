"""Shared types and data structures for lazybridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ContentType(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"


class NativeTool(StrEnum):
    """Provider-native server-side tools (run on provider infrastructure)."""

    WEB_SEARCH = "web_search"
    CODE_EXECUTION = "code_execution"
    FILE_SEARCH = "file_search"  # OpenAI
    COMPUTER_USE = "computer_use"
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


ContentBlock = TextContent | ImageContent | ToolUseContent | ToolResultContent | ThinkingContent


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
class CompletionRequest:
    """Unified request object passed to any provider."""

    messages: list[Message]
    model: str | None = None
    system: str | None = None
    max_tokens: int = 4096
    temperature: float | None = None
    tools: list[ToolDefinition] = field(default_factory=list)
    tool_choice: str | None = None  # "auto" | "required" | "none" | tool_name
    native_tools: list[NativeTool] = field(default_factory=list)
    structured_output: StructuredOutputConfig | None = None
    thinking: ThinkingConfig | None = None
    skills: SkillsConfig | None = None  # Anthropic only
    stream: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    thought_signature: Any = None  # Gemini thinking models: raw SDK Part, re-emitted as-is


@dataclass
class UsageStats:
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
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
