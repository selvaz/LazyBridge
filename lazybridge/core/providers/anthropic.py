"""Anthropic (Claude) provider for LazyBridge."""

from __future__ import annotations

import os
import warnings
from collections.abc import AsyncIterator, Iterator
from typing import Any

from lazybridge.core.providers.base import BaseProvider
from lazybridge.core.types import (
    CompletionRequest,
    CompletionResponse,
    GroundingSource,
    NativeTool,
    Role,
    StreamChunk,
    ToolCall,
    UsageStats,
)

# Lazy imports — only required if Anthropic provider is used
try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None  # type: ignore


# Maps NativeTool enum → Anthropic tool type strings
_NATIVE_TOOL_MAP: dict[NativeTool, dict[str, Any]] = {
    NativeTool.WEB_SEARCH: {
        "type": "web_search_20260209",
        "name": "web_search",
        "allowed_callers": ["direct"],
    },
    NativeTool.CODE_EXECUTION: {"type": "code_execution_20260120", "name": "code_execution"},
    NativeTool.COMPUTER_USE: {"type": "computer_use_20250124", "name": "computer"},
}

# Beta headers required per feature
_BETA_WEB_SEARCH = "web-search-2025-03-05"
_BETA_CODE_EXEC = "code-execution-2025-08-25"
_BETA_SKILLS = "skills-2025-10-02"
_BETA_FILES = "files-api-2025-04-14"
_BETA_COMPUTER_USE = "computer-use-2025-01-24"
_FORCE_STREAM_MAX_TOKENS = 20_000

# Price per 1M tokens (input, output). Approximate; verify at console.anthropic.com/pricing.
# Ordering matters: more-specific keys MUST appear before less-specific ones.
_PRICE_TABLE: dict[str, tuple[float, float]] = {
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (15.0, 75.0),
    "claude-opus-4-5": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (0.80, 4.0),
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-3-5-haiku": (0.80, 4.0),
    "claude-3-opus": (15.0, 75.0),
    "claude-3-sonnet": (3.0, 15.0),
    "claude-3-haiku": (0.25, 1.25),
}

# Models where temperature/top_p/top_k are not supported (returns 400)
_NO_SAMPLING_MODELS = frozenset({"claude-opus-4-7"})

# Models that use adaptive thinking only (no budget_tokens)
_ADAPTIVE_ONLY_MODELS = frozenset({"claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6"})


class AnthropicProvider(BaseProvider):
    """Anthropic Claude provider.

    Supports:
    - Adaptive thinking (claude-opus-4-7 / claude-opus-4-6 / claude-sonnet-4-6)
    - Extended thinking with budget_tokens (older models: 3.x, 4.5)
    - Structured output via messages.parse() + Pydantic
    - Tool use via beta tool_runner or manual loop
    - Native tools: web_search, code_execution, computer_use
    - Anthropic Skills (domain-expert server-side packages)
    - Streaming

    Model-specific behavior:
    - Opus 4.7: adaptive thinking only, no temperature/sampling params
    - Opus 4.6 / Sonnet 4.6: adaptive thinking, full sampling support
    - Older models: extended thinking with budget_tokens
    """

    default_model = "claude-sonnet-4-6"

    # Tier aliases — ``LazyAgent("anthropic", model="top")`` resolves here.
    # Update this table when new models ship; the matrix in
    # lazy_wiki/human/agents.md mirrors it (audit F2).
    _TIER_ALIASES = {
        "top":         "claude-opus-4-7",
        "expensive":   "claude-opus-4-6",
        "medium":      "claude-sonnet-4-6",
        "cheap":       "claude-haiku-4-5",
        "super_cheap": "claude-3-haiku",
    }
    _FALLBACKS = {
        "claude-opus-4-7":   ["claude-opus-4-6", "claude-sonnet-4-6"],
        "claude-opus-4-6":   ["claude-opus-4-5", "claude-sonnet-4-6"],
        "claude-sonnet-4-6": ["claude-sonnet-4-5", "claude-3-5-sonnet"],
        "claude-haiku-4-5":  ["claude-3-5-haiku"],
    }
    supported_native_tools: frozenset[NativeTool] = frozenset(
        {
            NativeTool.WEB_SEARCH,
            NativeTool.CODE_EXECUTION,
            NativeTool.COMPUTER_USE,
        }
    )

    def _compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float | None:
        # Substring matching against model.lower() lets a fully-qualified model
        # string like "claude-sonnet-4-6-20250514" match the short key
        # "claude-sonnet-4-6" without having to enumerate every version suffix.
        #
        # Ordering is critical: more-specific keys MUST appear before less-specific
        # ones in _PRICE_TABLE (e.g. "claude-3-5-sonnet" before "claude-3-sonnet")
        # so that a longer, specific key matches first.  Python 3.7+ dicts preserve
        # insertion order, which makes the match deterministic.
        model_l = model.lower()
        for key, (in_price, out_price) in _PRICE_TABLE.items():
            if key in model_l:
                return (input_tokens * in_price + output_tokens * out_price) / 1_000_000
        return None

    def get_default_max_tokens(self, model: str | None = None) -> int:
        """Return the default max_tokens for a given Anthropic model."""
        resolved = (model or self.model or self.default_model or "").lower()
        if "opus-4-7" in resolved or "opus-4-6" in resolved:
            return 128_000
        if any(x in resolved for x in ("sonnet-4-6", "haiku-4-5", "opus-4-5", "sonnet-4-5")):
            return 64_000
        if "haiku" in resolved:
            return 4_096
        if "sonnet" in resolved or "opus" in resolved:
            return 8_192
        return 8_192

    def _init_client(self, **kwargs) -> None:
        if _anthropic is None:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "Anthropic API key not found. Set the ANTHROPIC_API_KEY environment "
                "variable, or pass api_key= to LazyAgent/AnthropicProvider."
            )
        # Allow callers to override beta header versions and the streaming threshold.
        self._beta_overrides: dict[str, str] = kwargs.pop("beta_overrides", {}) or {}
        self._force_stream_threshold: int = kwargs.pop("force_stream_threshold", _FORCE_STREAM_MAX_TOKENS)
        self._client = _anthropic.Anthropic(api_key=key, **kwargs)
        self._async_client = _anthropic.AsyncAnthropic(api_key=key, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _messages_to_anthropic(self, request: CompletionRequest) -> list[dict]:
        """Convert unified Messages to Anthropic format."""
        result: list[dict[str, Any]] = []
        for msg in request.messages:
            if msg.role == Role.SYSTEM:
                continue  # system handled separately
            if isinstance(msg.content, str):
                api_role = "user" if msg.role == Role.TOOL else msg.role.value
                result.append({"role": api_role, "content": msg.content})
            else:
                blocks: list[dict[str, Any]] = []
                for block in msg.content:
                    from lazybridge.core.types import (
                        ImageContent,
                        TextContent,
                        ThinkingContent,
                        ToolResultContent,
                        ToolUseContent,
                    )

                    if isinstance(block, TextContent):
                        blocks.append({"type": "text", "text": block.text})
                    elif isinstance(block, ThinkingContent):
                        blocks.append({"type": "thinking", "thinking": block.thinking})
                    elif isinstance(block, ImageContent):
                        if block.url:
                            blocks.append(
                                {
                                    "type": "image",
                                    "source": {"type": "url", "url": block.url},
                                }
                            )
                        elif block.base64_data:
                            blocks.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": block.media_type,
                                        "data": block.base64_data,
                                    },
                                }
                            )
                    elif isinstance(block, ToolUseContent):
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input,
                            }
                        )
                    elif isinstance(block, ToolResultContent):
                        blocks.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.tool_use_id,
                                "content": block.content,
                                "is_error": block.is_error,
                            }
                        )
                api_role = "user" if msg.role == Role.TOOL else msg.role.value
                result.append({"role": api_role, "content": blocks})
        return result

    def _get_system(self, request: CompletionRequest) -> str | None:
        """Extract system prompt from request or system messages."""
        if request.system:
            return request.system
        for msg in request.messages:
            if msg.role == Role.SYSTEM:
                return msg.to_text()
        return None

    def _build_tools(self, request: CompletionRequest) -> list[dict]:
        """Build tool definitions list for Anthropic API."""
        tools = []
        for t in request.tools:
            tool_def: dict[str, Any] = {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            if t.strict:
                tool_def["strict"] = True
            tools.append(tool_def)
        # Add native tools
        for nt in self._check_native_tools(request.native_tools):
            tools.append(_NATIVE_TOOL_MAP[nt])
        return tools

    def _build_betas(self, request: CompletionRequest) -> list[str]:
        # getattr with default {} because __new__-based test construction can
        # bypass __init__, leaving _beta_overrides unset on the instance.
        overrides = getattr(self, "_beta_overrides", {})
        betas: list[str] = []
        # Each feature requires its own beta header string.  overrides allows
        # callers to pin a specific version or substitute a newer beta header
        # without subclassing the provider (e.g. for early-access preview access).
        if any(nt == NativeTool.WEB_SEARCH for nt in request.native_tools):
            betas.append(overrides.get("web_search", _BETA_WEB_SEARCH))
        if any(nt == NativeTool.CODE_EXECUTION for nt in request.native_tools):
            betas.append(overrides.get("code_execution", _BETA_CODE_EXEC))
        if any(nt == NativeTool.COMPUTER_USE for nt in request.native_tools):
            betas.append(overrides.get("computer_use", _BETA_COMPUTER_USE))
        if request.skills:
            # Skills require three beta headers simultaneously: the skills package
            # header, code execution (skills can execute code), and files API (skills
            # may read/write files).  Duplicates are removed below.
            betas.extend(
                [
                    overrides.get("skills", _BETA_SKILLS),
                    overrides.get("code_execution", _BETA_CODE_EXEC),
                    overrides.get("files", _BETA_FILES),
                ]
            )
        # dict.fromkeys preserves insertion order while removing duplicates.
        # This is needed when both skills and code_execution are requested —
        # both contribute _BETA_CODE_EXEC and the API rejects repeated headers.
        return list(dict.fromkeys(betas))

    def _beta_kwargs(self, betas: list[str]) -> dict[str, Any]:
        return {"betas": betas} if betas else {}

    def _build_thinking(self, request: CompletionRequest) -> dict | None:
        if not request.thinking or not request.thinking.enabled:
            return None
        model = self._resolve_model(request)

        # Check if this model uses adaptive-only thinking
        is_adaptive = any(key in model for key in _ADAPTIVE_ONLY_MODELS)

        if is_adaptive:
            if request.thinking.budget_tokens is not None:
                import warnings

                warnings.warn(
                    f"ThinkingConfig.budget_tokens is ignored for model '{model}'. "
                    "Use 'effort' instead (adaptive thinking only).",
                    DeprecationWarning,
                    stacklevel=4,
                )
            thinking: dict[str, Any] = {"type": "adaptive"}
            if request.thinking.display:
                thinking["display"] = request.thinking.display
            return thinking

        # Older models (3.x, 4.5): explicit budget required
        budget = request.thinking.budget_tokens or 8000
        return {"type": "enabled", "budget_tokens": budget}

    def _build_params(self, request: CompletionRequest) -> dict[str, Any]:
        model = self._resolve_model(request)
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "messages": self._messages_to_anthropic(request),
        }
        system = self._get_system(request)
        if system:
            params["system"] = system
        # Opus 4.7 does not support temperature/top_p/top_k — but we now
        # warn rather than drop silently so users aren't surprised when
        # their temperature setting has no effect (audit M7).
        no_sampling = any(key in model for key in _NO_SAMPLING_MODELS)
        if request.temperature is not None:
            if no_sampling:
                warnings.warn(
                    f"Anthropic model {model!r} does not support the temperature "
                    "parameter; your temperature= value is being ignored. "
                    "Drop it from the call or pick a different model to suppress "
                    "this warning.",
                    UserWarning,
                    stacklevel=4,
                )
            else:
                params["temperature"] = request.temperature
        tools = self._build_tools(request)
        if tools:
            params["tools"] = tools
        if request.tool_choice:
            if request.tool_choice in ("auto", "none", "required"):
                params["tool_choice"] = {"type": request.tool_choice}
            else:
                params["tool_choice"] = {"type": "tool", "name": request.tool_choice}
        thinking = self._build_thinking(request)
        if thinking:
            params["thinking"] = thinking
        return params

    def _parse_response(self, response: Any) -> CompletionResponse:
        """Convert Anthropic response to unified CompletionResponse."""
        from lazybridge.core.types import GroundingSource

        content = ""
        thinking = None
        tool_calls: list[ToolCall] = []
        grounding_sources: list[GroundingSource] = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "thinking":
                thinking = (thinking or "") + block.thinking
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )
            elif block.type == "web_search_result":
                grounding_sources.append(
                    GroundingSource(
                        url=getattr(block, "url", "") or "",
                        title=getattr(block, "title", None),
                    )
                )

        usage = UsageStats(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        # reasoning_tokens is an optional field added to the usage object when
        # the model was run in thinking/reasoning mode.  getattr with None as the
        # default avoids AttributeError on SDK versions that predate this field.
        _tt = getattr(response.usage, "reasoning_tokens", None)
        if _tt is not None:
            usage.thinking_tokens = _tt
        # Price lookup by substring match — see _compute_cost for ordering notes.
        usage.cost_usd = self._compute_cost(response.model, usage.input_tokens, usage.output_tokens)

        return CompletionResponse(
            content=content,
            thinking=thinking,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "end_turn",
            model=response.model,
            usage=usage,
            raw=response,
            grounding_sources=grounding_sources,
        )

    def _should_force_streaming(self, request: CompletionRequest) -> bool:
        threshold = getattr(self, "_force_stream_threshold", _FORCE_STREAM_MAX_TOKENS)
        if request.max_tokens and request.max_tokens > threshold:
            import logging as _logging

            _logging.getLogger(__name__).debug(
                "AnthropicProvider: auto-forcing streaming because max_tokens=%d > threshold=%d.",
                request.max_tokens,
                threshold,
            )
            return True
        return False

    def _collect_streamed_response(self, request: CompletionRequest) -> CompletionResponse:
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        final_chunk: StreamChunk | None = None
        for chunk in self.stream(request):
            if chunk.delta:
                content_parts.append(chunk.delta)
            if chunk.thinking_delta:
                thinking_parts.append(chunk.thinking_delta)
            if chunk.is_final:
                final_chunk = chunk
        final_usage = final_chunk.usage if final_chunk and final_chunk.usage else UsageStats()
        return CompletionResponse(
            content="".join(content_parts),
            thinking="".join(thinking_parts) or None,
            tool_calls=final_chunk.tool_calls if final_chunk else [],
            stop_reason=final_chunk.stop_reason or "end_turn" if final_chunk else "end_turn",
            model=self._resolve_model(request),
            usage=final_usage,
            raw=None,  # not available in force-streamed mode
            parsed=final_chunk.parsed if final_chunk else None,
            validation_error=final_chunk.validation_error if final_chunk else None,
            validated=final_chunk.validated if final_chunk else None,
        )

    async def _acollect_streamed_response(self, request: CompletionRequest) -> CompletionResponse:
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        final_chunk: StreamChunk | None = None
        async for chunk in self.astream(request):
            if chunk.delta:
                content_parts.append(chunk.delta)
            if chunk.thinking_delta:
                thinking_parts.append(chunk.thinking_delta)
            if chunk.is_final:
                final_chunk = chunk
        final_usage = final_chunk.usage if final_chunk and final_chunk.usage else UsageStats()
        return CompletionResponse(
            content="".join(content_parts),
            thinking="".join(thinking_parts) or None,
            tool_calls=final_chunk.tool_calls if final_chunk else [],
            stop_reason=final_chunk.stop_reason or "end_turn" if final_chunk else "end_turn",
            model=self._resolve_model(request),
            usage=final_usage,
            raw=None,  # not available in force-streamed mode
            parsed=final_chunk.parsed if final_chunk else None,
            validation_error=final_chunk.validation_error if final_chunk else None,
            validated=final_chunk.validated if final_chunk else None,
        )

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Execute a synchronous Anthropic Messages API call."""
        if self._should_force_streaming(request):
            return self._collect_streamed_response(request)
        betas = self._build_betas(request)
        params = self._build_params(request)

        if request.structured_output and request.structured_output.schema:
            from lazybridge.core.structured import (
                apply_structured_validation,
                normalize_json_schema,
            )

            schema = request.structured_output.schema
            if isinstance(schema, dict):
                schema = normalize_json_schema(schema)
                params["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
                response = self._client.beta.messages.create(**params, **self._beta_kwargs(betas))
                resp = self._parse_response(response)
                apply_structured_validation(resp, resp.content, schema)
            else:
                use_native_parse = not request.thinking and not request.native_tools and not betas
                if use_native_parse:
                    try:
                        response = self._client.messages.parse(
                            **params,
                            output_format=schema,
                        )
                        resp = self._parse_response(response)
                        if response.parsed_output is not None:
                            resp.parsed = response.parsed_output
                            resp.validated = True
                        else:
                            apply_structured_validation(resp, resp.content, schema)
                    except (AttributeError, NotImplementedError) as _pe:
                        # messages.parse() not available on this SDK version — fall back.
                        # Log once at DEBUG so users diagnosing "why is validation
                        # different?" can find the signal (audit L7).
                        import logging as _logging
                        _logging.getLogger(__name__).debug(
                            "Anthropic SDK lacks messages.parse(); falling back to "
                            "manual JSON parse for schema %r (%s)",
                            getattr(schema, "__name__", schema), _pe,
                        )
                        response = self._client.messages.create(**params)
                        resp = self._parse_response(response)
                        apply_structured_validation(resp, resp.content, schema)
                else:
                    response = self._client.messages.create(**params)
                    resp = self._parse_response(response)
                    apply_structured_validation(resp, resp.content, schema)
            return resp

        if betas:
            response = self._client.beta.messages.create(**params, **self._beta_kwargs(betas))
        else:
            response = self._client.messages.create(**params)

        return self._parse_response(response)

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        """Stream a completion, yielding StreamChunk objects."""
        betas = self._build_betas(request)
        params = self._build_params(request)

        ctx: Any
        if betas:
            ctx = self._client.beta.messages.stream(**params, **self._beta_kwargs(betas))
        else:
            ctx = self._client.messages.stream(**params)

        # text_accum collects all text deltas so structured output validation
        # can be run against the complete response at message_stop time.
        text_accum = ""
        with ctx as s:
            for event in s:
                if event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        text_accum += delta.text
                        yield StreamChunk(delta=delta.text)
                    elif delta.type == "thinking_delta":
                        # Thinking deltas are streamed separately; they do not
                        # contribute to text_accum (thinking tokens are not
                        # included in the JSON output for structured validation).
                        yield StreamChunk(thinking_delta=delta.thinking)
                elif event.type == "message_stop":
                    # message_stop is the terminal event.  get_final_message()
                    # returns the fully-assembled response (including usage stats)
                    # after the stream has been drained.
                    final = s.get_final_message()
                    usage = UsageStats(
                        input_tokens=final.usage.input_tokens,
                        output_tokens=final.usage.output_tokens,
                    )
                    # reasoning_tokens is only present for thinking-mode responses.
                    _thinking_tokens = getattr(final.usage, "reasoning_tokens", None)
                    if _thinking_tokens is not None:
                        usage.thinking_tokens = _thinking_tokens
                    usage.cost_usd = self._compute_cost(
                        getattr(final, "model", ""), usage.input_tokens, usage.output_tokens
                    )
                    tool_calls = [
                        ToolCall(id=b.id, name=b.name, arguments=b.input) for b in final.content if b.type == "tool_use"
                    ]
                    grounding_sources = [
                        GroundingSource(
                            url=getattr(b, "url", "") or "",
                            title=getattr(b, "title", None),
                        )
                        for b in final.content
                        if b.type == "web_search_result"
                    ]
                    final_chunk = StreamChunk(
                        stop_reason=final.stop_reason,
                        usage=usage,
                        tool_calls=tool_calls,
                        is_final=True,
                        grounding_sources=grounding_sources,
                    )
                    if request.structured_output:
                        from lazybridge.core.structured import apply_structured_validation

                        apply_structured_validation(final_chunk, text_accum, request.structured_output.schema)
                    yield final_chunk

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        """Async completion."""
        if self._should_force_streaming(request):
            return await self._acollect_streamed_response(request)
        betas = self._build_betas(request)
        params = self._build_params(request)

        if request.structured_output and request.structured_output.schema:
            from lazybridge.core.structured import (
                apply_structured_validation,
                normalize_json_schema,
            )

            schema = request.structured_output.schema
            if isinstance(schema, dict):
                # Always use beta endpoint — output_config requires it.
                schema = normalize_json_schema(schema)
                params["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
                response = await self._async_client.beta.messages.create(**params, **self._beta_kwargs(betas))
                resp = self._parse_response(response)
                apply_structured_validation(resp, resp.content, schema)
            else:
                use_native_parse = not request.thinking and not request.native_tools and not betas
                if use_native_parse:
                    try:
                        response = await self._async_client.messages.parse(
                            **params,
                            output_format=schema,
                        )
                        resp = self._parse_response(response)
                        if response.parsed_output is not None:
                            resp.parsed = response.parsed_output
                            resp.validated = True
                        else:
                            apply_structured_validation(resp, resp.content, schema)
                    except (AttributeError, NotImplementedError):
                        # messages.parse() not available on this SDK version — fall back
                        response = await self._async_client.messages.create(**params)
                        resp = self._parse_response(response)
                        apply_structured_validation(resp, resp.content, schema)
                else:
                    response = await self._async_client.messages.create(**params)
                    resp = self._parse_response(response)
                    apply_structured_validation(resp, resp.content, schema)
            return resp

        if betas:
            response = await self._async_client.beta.messages.create(**params, **self._beta_kwargs(betas))
        else:
            response = await self._async_client.messages.create(**params)

        return self._parse_response(response)

    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        """Async streaming completion."""
        betas = self._build_betas(request)
        params = self._build_params(request)

        ctx: Any
        if betas:
            ctx = self._async_client.beta.messages.stream(**params, **self._beta_kwargs(betas))
        else:
            ctx = self._async_client.messages.stream(**params)

        # text_accum collects all text deltas for end-of-stream validation.
        # See the sync stream() method for detailed inline commentary.
        text_accum = ""
        async with ctx as s:
            async for event in s:
                if event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        text_accum += delta.text
                        yield StreamChunk(delta=delta.text)
                    elif delta.type == "thinking_delta":
                        yield StreamChunk(thinking_delta=delta.thinking)
                elif event.type == "message_stop":
                    # get_final_message() is awaited here (async context manager).
                    final = await s.get_final_message()
                    usage = UsageStats(
                        input_tokens=final.usage.input_tokens,
                        output_tokens=final.usage.output_tokens,
                    )
                    # reasoning_tokens only present for thinking-mode responses.
                    _thinking_tokens = getattr(final.usage, "reasoning_tokens", None)
                    if _thinking_tokens is not None:
                        usage.thinking_tokens = _thinking_tokens
                    usage.cost_usd = self._compute_cost(
                        getattr(final, "model", ""), usage.input_tokens, usage.output_tokens
                    )
                    tool_calls = [
                        ToolCall(id=b.id, name=b.name, arguments=b.input) for b in final.content if b.type == "tool_use"
                    ]
                    grounding_sources = [
                        GroundingSource(
                            url=getattr(b, "url", "") or "",
                            title=getattr(b, "title", None),
                        )
                        for b in final.content
                        if b.type == "web_search_result"
                    ]
                    final_chunk = StreamChunk(
                        stop_reason=final.stop_reason,
                        usage=usage,
                        tool_calls=tool_calls,
                        is_final=True,
                        grounding_sources=grounding_sources,
                    )
                    if request.structured_output:
                        from lazybridge.core.structured import apply_structured_validation

                        apply_structured_validation(final_chunk, text_accum, request.structured_output.schema)
                    yield final_chunk
