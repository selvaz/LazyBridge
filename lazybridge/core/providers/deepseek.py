"""DeepSeek provider for LazyBridge.

DeepSeek's API is fully compatible with the OpenAI SDK — it uses the same
client with a custom base_url and different model names.

Models (April 2026):
  Flagship:    deepseek-v4-pro    (1.6T/49B active, 1M ctx, 384K out, optional thinking)
  Fast/Cheap:  deepseek-v4-flash  (284B/13B active, 1M ctx, 384K out, optional thinking)

Deprecated (retire 2026-07-24 — currently routed to deepseek-v4-flash by the API):
  deepseek-reasoner  → deepseek-v4-flash (thinking mode)
  deepseek-chat      → deepseek-v4-flash (non-thinking mode)

Thinking mode (V4 models):
  - Activated by passing ThinkingConfig to the request.
  - The API receives extra_body={"thinking": {"type": "enabled"}}.
  - Chain-of-thought surfaces in the ``reasoning_content`` field (streaming and non-streaming).
  - temperature, top_p, presence_penalty, frequency_penalty are ignored in thinking mode.
  - tool_choice is not supported in thinking mode.

Thinking mode (legacy deepseek-reasoner):
  - Always in thinking mode; reasoning_content is always present.

Native tools:
  - No provider-native server-side tools (web search etc.) via API.
  - Standard OpenAI-compatible function calling supported.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator

from lazybridge.core.providers.openai import OpenAIProvider, _safe_json_loads
from lazybridge.core.types import (
    CompletionRequest,
    CompletionResponse,
    NativeTool,
    StreamChunk,
    ToolCall,
    UsageStats,
)

_logger = logging.getLogger(__name__)

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_DEEPSEEK_ENV_KEY = "DEEPSEEK_API_KEY"

# Legacy model permanently in thinking/reasoning mode.
_REASONING_MODELS = frozenset({"deepseek-reasoner"})

# V4 models that support optional thinking via ThinkingConfig.
_THINKING_CAPABLE_MODELS = frozenset({"deepseek-v4-pro", "deepseek-v4-flash"})

# Parameters silently ignored by the API when thinking mode is active.
_THINKING_SUPPRESSED_PARAMS = frozenset({"temperature", "top_p", "presence_penalty", "frequency_penalty"})

# Price per 1M tokens (input, output). Approximate; verify at platform.deepseek.com/api-docs/pricing.
_PRICE_TABLE: dict[str, tuple[float, float]] = {
    "deepseek-v4-pro": (1.74, 3.48),
    "deepseek-v4-flash": (0.14, 0.28),
    # Deprecated 2026-07-24; currently API-routed to deepseek-v4-flash.
    "deepseek-reasoner": (0.14, 0.28),
    "deepseek-chat": (0.14, 0.28),
}


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek provider — extends OpenAI provider with DeepSeek-specific handling.

    Supports:
    - deepseek-v4-flash: fast/cheap general chat + optional thinking + function calling
    - deepseek-v4-pro: flagship, higher quality + optional thinking + function calling
    - deepseek-reasoner (deprecated): always-on reasoning via reasoning_content field
    - Streaming with reasoning extraction
    - JSON mode structured output (response_format: json_object)
    """

    default_model = "deepseek-v4-flash"

    _TIER_ALIASES = {
        "top": "deepseek-v4-pro",
        "expensive": "deepseek-v4-pro",
        "medium": "deepseek-v4-flash",
        "cheap": "deepseek-v4-flash",
        "super_cheap": "deepseek-v4-flash",
    }
    _FALLBACKS = {
        "deepseek-v4-pro": ["deepseek-v4-flash"],
        "deepseek-reasoner": ["deepseek-v4-flash"],
    }
    supported_native_tools: frozenset[NativeTool] = frozenset()  # No native server tools

    def _compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float | None:
        model_l = model.lower()
        for key, (in_price, out_price) in _PRICE_TABLE.items():
            if key in model_l:
                return (input_tokens * in_price + output_tokens * out_price) / 1_000_000
        return None

    def get_default_max_tokens(self, model: str | None = None) -> int:
        """Return the default max_tokens for the given model."""
        resolved = (model or self.model or self.default_model or "").lower()
        if "v4" in resolved:
            return 64_000  # Conservative default; V4 models support up to 384K
        if "reasoner" in resolved:
            return 64_000
        return 8_000

    def _init_client(self, **kwargs) -> None:
        import os

        try:
            import openai as _openai
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai") from None

        key = self.api_key or os.environ.get(_DEEPSEEK_ENV_KEY)
        if not key:
            raise ValueError(
                f"DeepSeek API key not found. Set the {_DEEPSEEK_ENV_KEY} environment variable "
                "or pass api_key= to the provider."
            )
        self._client = _openai.OpenAI(
            api_key=key,
            base_url=_DEEPSEEK_BASE_URL,
        )
        self._async_client = _openai.AsyncOpenAI(
            api_key=key,
            base_url=_DEEPSEEK_BASE_URL,
        )

    def _is_reasoning_model(self, model: str) -> bool:
        return model in _REASONING_MODELS

    def _is_thinking_active(self, request: CompletionRequest, model: str) -> bool:
        """True when the request will run in thinking/reasoning mode."""
        if model in _REASONING_MODELS:
            return True
        return (
            model in _THINKING_CAPABLE_MODELS
            and bool(request.thinking and request.thinking.enabled)
        )

    def _resolve_thinking(self, request: CompletionRequest) -> CompletionRequest:
        """Validate that thinking is only requested on models that support it.

        Previously the provider auto-switched to ``deepseek-reasoner``
        with only a warning — changing the model underneath the caller
        is too surprising (audit M8).  The caller now has to pick the
        reasoning model explicitly.
        """
        if request.thinking and request.thinking.enabled:
            model = self._resolve_model(request)
            if model not in (_REASONING_MODELS | _THINKING_CAPABLE_MODELS):
                raise ValueError(
                    f"DeepSeek: thinking was requested but model {model!r} does "
                    "not support reasoning. Use 'deepseek-v4-pro' or 'deepseek-v4-flash' "
                    "for thinking, or drop thinking= for this call."
                )
        return request

    def _apply_thinking_params(self, params: dict, model: str, request: CompletionRequest) -> None:
        """Mutate params in-place to activate thinking mode on V4 models."""
        if model not in _THINKING_CAPABLE_MODELS:
            return
        if not (request.thinking and request.thinking.enabled):
            return
        params.setdefault("extra_body", {})["thinking"] = {"type": "enabled"}
        # Strip params the API silently ignores in thinking mode to avoid confusion.
        for p in _THINKING_SUPPRESSED_PARAMS:
            params.pop(p, None)

    # ------------------------------------------------------------------
    # Override: extract reasoning_content from DeepSeek responses
    # ------------------------------------------------------------------

    def _parse_deepseek_chat_response(self, response, model: str) -> CompletionResponse:
        if not response.choices:
            _logger.warning("DeepSeek response has no choices (content filter, quota, or API error).")
            usage = UsageStats(
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
            )
            usage.cost_usd = self._compute_cost(model, usage.input_tokens, usage.output_tokens)
            return CompletionResponse(
                content="",
                tool_calls=[],
                stop_reason="error",
                model=model,
                usage=usage,
                raw=response,
            )
        choice = response.choices[0]
        msg = choice.message
        content = msg.content or ""
        thinking = getattr(msg, "reasoning_content", None)
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=_safe_json_loads(tc.function.arguments),
                    )
                )
        usage = UsageStats(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )
        usage = self._populate_reasoning_tokens(usage, response.usage)
        usage.cost_usd = self._compute_cost(model, usage.input_tokens, usage.output_tokens)
        return CompletionResponse(
            content=content,
            thinking=thinking,
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason or "end_turn",
            model=model,
            usage=usage,
            raw=response,
        )

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Execute a synchronous completion."""
        request = self._resolve_thinking(request)
        model = self._resolve_model(request)
        params = self._build_chat_params(request)

        if self._is_thinking_active(request, model):
            params.pop("tool_choice", None)

        self._apply_thinking_params(params, model, request)

        # DeepSeek structured output: JSON mode only (not full schema enforcement).
        # Skip when tools are present — tool calls return empty content, which
        # breaks JSON parsing. The json() caller uses loop() + repair instead.
        has_tools = bool(params.get("tools"))
        if request.structured_output and not has_tools:
            params["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**params)
        resp = self._parse_deepseek_chat_response(response, model)

        if request.structured_output and not resp.tool_calls:
            from lazybridge.core.structured import apply_structured_validation

            apply_structured_validation(resp, resp.content, request.structured_output.schema)

        return resp

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        """Stream a completion, yielding StreamChunk objects."""
        request = self._resolve_thinking(request)
        model = self._resolve_model(request)
        params = self._build_chat_params(request)
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}

        if self._is_thinking_active(request, model):
            params.pop("tool_choice", None)

        self._apply_thinking_params(params, model, request)

        text_accum = ""
        tool_call_accum: dict[int, dict] = {}
        for chunk in self._client.chat.completions.create(**params):
            choice = chunk.choices[0] if chunk.choices else None
            if choice:
                delta = choice.delta
                reasoning_delta = getattr(delta, "reasoning_content", None) or ""
                content_delta = delta.content or ""

                if reasoning_delta:
                    yield StreamChunk(thinking_delta=reasoning_delta)
                if content_delta:
                    text_accum += content_delta
                    yield StreamChunk(delta=content_delta)

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_call_accum:
                            tool_call_accum[idx] = {"id": tc.id or "", "name": tc.function.name or "", "args": ""}
                        if tc.id:
                            tool_call_accum[idx]["id"] = tc.id
                        if tc.function.name:
                            tool_call_accum[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_call_accum[idx]["args"] += tc.function.arguments

                if choice.finish_reason:
                    tool_calls = [
                        ToolCall(id=v["id"], name=v["name"], arguments=_safe_json_loads(v["args"]))
                        for v in tool_call_accum.values()
                    ]
                    usage = None
                    if chunk.usage:
                        usage = UsageStats(
                            input_tokens=chunk.usage.prompt_tokens,
                            output_tokens=chunk.usage.completion_tokens,
                        )
                        usage = self._populate_reasoning_tokens(usage, chunk.usage)
                        usage.cost_usd = self._compute_cost(
                            getattr(chunk, "model", "") or "", usage.input_tokens, usage.output_tokens
                        )
                    final_chunk = StreamChunk(
                        stop_reason=choice.finish_reason,
                        tool_calls=tool_calls,
                        usage=usage,
                        is_final=True,
                    )
                    if request.structured_output:
                        from lazybridge.core.structured import apply_structured_validation

                        apply_structured_validation(final_chunk, text_accum, request.structured_output.schema)
                    yield final_chunk

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        """Async completion."""
        request = self._resolve_thinking(request)
        model = self._resolve_model(request)
        params = self._build_chat_params(request)

        if self._is_thinking_active(request, model):
            params.pop("tool_choice", None)

        self._apply_thinking_params(params, model, request)

        has_tools = bool(params.get("tools"))
        if request.structured_output and not has_tools:
            params["response_format"] = {"type": "json_object"}

        response = await self._async_client.chat.completions.create(**params)
        resp = self._parse_deepseek_chat_response(response, model)

        if request.structured_output and not resp.tool_calls:
            from lazybridge.core.structured import apply_structured_validation

            apply_structured_validation(resp, resp.content, request.structured_output.schema)

        return resp

    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        """Async streaming completion."""
        request = self._resolve_thinking(request)
        model = self._resolve_model(request)
        params = self._build_chat_params(request)
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}

        if self._is_thinking_active(request, model):
            params.pop("tool_choice", None)

        self._apply_thinking_params(params, model, request)

        text_accum = ""
        tool_call_accum: dict[int, dict] = {}
        async for chunk in await self._async_client.chat.completions.create(**params):
            choice = chunk.choices[0] if chunk.choices else None
            if choice:
                delta = choice.delta
                reasoning_delta = getattr(delta, "reasoning_content", None) or ""
                content_delta = delta.content or ""
                if reasoning_delta:
                    yield StreamChunk(thinking_delta=reasoning_delta)
                if content_delta:
                    text_accum += content_delta
                    yield StreamChunk(delta=content_delta)

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_call_accum:
                            tool_call_accum[idx] = {"id": tc.id or "", "name": tc.function.name or "", "args": ""}
                        if tc.id:
                            tool_call_accum[idx]["id"] = tc.id
                        if tc.function.name:
                            tool_call_accum[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_call_accum[idx]["args"] += tc.function.arguments

                if choice.finish_reason:
                    tool_calls = [
                        ToolCall(id=v["id"], name=v["name"], arguments=_safe_json_loads(v["args"]))
                        for v in tool_call_accum.values()
                    ]
                    usage = None
                    if chunk.usage:
                        usage = UsageStats(
                            input_tokens=chunk.usage.prompt_tokens,
                            output_tokens=chunk.usage.completion_tokens,
                        )
                        usage = self._populate_reasoning_tokens(usage, chunk.usage)
                        usage.cost_usd = self._compute_cost(
                            getattr(chunk, "model", "") or "", usage.input_tokens, usage.output_tokens
                        )
                    final_chunk = StreamChunk(
                        stop_reason=choice.finish_reason,
                        tool_calls=tool_calls,
                        usage=usage,
                        is_final=True,
                    )
                    if request.structured_output:
                        from lazybridge.core.structured import apply_structured_validation

                        apply_structured_validation(final_chunk, text_accum, request.structured_output.schema)
                    yield final_chunk
