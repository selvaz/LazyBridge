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
from typing import Any

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

# Price per 1M tokens (input, output). Verify at platform.deepseek.com/api-docs/pricing.
# Cache-hit input rates (automatic for repeated prefixes ≥1024 tokens, same account):
#   deepseek-v4-pro:   $0.145/M   deepseek-v4-flash: $0.028/M
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

    # DeepSeek's standard API (deepseek-chat / deepseek-v4-*) is
    # text-only.  DeepSeek-VL2 ships separately and is not routed
    # through this provider — override OpenAIProvider's vision matrix
    # explicitly so a model id like ``gpt-4o`` (impossible here, but
    # MRO-inherited) can't accidentally turn the matcher truthy.
    _VISION_CAPABLE_MODEL_PATTERNS = frozenset()

    # DeepSeek does not accept audio input on the public API.
    _AUDIO_CAPABLE_MODEL_PATTERNS = frozenset()

    def _compute_cost(
        self, model: str, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0
    ) -> float | None:
        # Match the OpenAIProvider supertype signature (cached_input_tokens
        # ignored — DeepSeek pricing is the same regardless of caching).
        del cached_input_tokens  # unused; kept for signature compatibility
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
        self._structured_drop_warned: bool = False

    def _warn_structured_drop_once(self) -> None:
        """Warn the first time ``structured_output`` is dropped because
        ``tools`` are present in the same request.  Stamped on the
        instance so a long-running provider doesn't spam the log every
        turn."""
        if getattr(self, "_structured_drop_warned", False):
            return
        import warnings

        warnings.warn(
            "DeepSeek: structured_output is silently disabled when tools "
            "are present (the tool-loop returns empty content which can't "
            "be JSON-parsed).  Validate the final response yourself, or "
            "drop ``tools`` for the structured-output call.",
            UserWarning,
            stacklevel=4,
        )
        self._structured_drop_warned = True  # type: ignore[attr-defined]

    def _is_reasoning_model(self, model: str) -> bool:
        return model in _REASONING_MODELS

    def _is_thinking_active(self, request: CompletionRequest, model: str) -> bool:
        """True when the request will run in thinking/reasoning mode."""
        if model in _REASONING_MODELS:
            return True
        return model in _THINKING_CAPABLE_MODELS and bool(request.thinking and request.thinking.enabled)

    def _resolve_thinking(self, request: CompletionRequest) -> CompletionRequest:
        """Validate that thinking is only requested on models that support it.

        The caller has to pick the reasoning model explicitly —
        auto-switching the model underneath them would be surprising.
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
        """Mutate params in-place to control thinking mode on V4 models.

        deepseek-v4-flash and deepseek-v4-pro activate thinking by default
        when the API decides to.  An explicit disable is required to prevent
        the model from returning ``reasoning_content``, which would cause a
        400 error on the next tool-call turn ("reasoning_content must be
        passed back to the API").
        """
        if model not in _THINKING_CAPABLE_MODELS:
            return
        if request.thinking and request.thinking.enabled:
            params.setdefault("extra_body", {})["thinking"] = {"type": "enabled"}
            # Strip params the API silently ignores in thinking mode.
            for p in _THINKING_SUPPRESSED_PARAMS:
                params.pop(p, None)
        else:
            # Explicitly disable so the API never returns reasoning_content
            # in non-thinking calls — avoids passback errors on multi-turn
            # tool-calling loops.
            params.setdefault("extra_body", {})["thinking"] = {"type": "disabled"}

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

    @staticmethod
    def _ensure_json_word_in_prompt(params: dict, schema: Any = None) -> None:
        """DeepSeek requires the literal word 'json' somewhere in the prompt
        when response_format=json_object is used, or the API returns 400.
        Also injects the expected JSON schema so the model produces the right shape.
        """
        import json as _json

        if schema is not None:
            try:
                if hasattr(schema, "model_json_schema"):
                    schema_str = _json.dumps(schema.model_json_schema(), indent=2)
                elif isinstance(schema, dict):
                    schema_str = _json.dumps(schema, indent=2)
                else:
                    schema_str = str(schema)
            except Exception:
                schema_str = str(schema)
            instruction = f"Respond with a valid JSON object matching this schema exactly:\n```json\n{schema_str}\n```"
        else:
            messages: list[dict] = params.get("messages", [])
            has_json = any("json" in str(m.get("content", "")).lower() for m in messages)
            if has_json:
                return
            instruction = "Respond with a valid JSON object."

        messages = params.get("messages", [])
        if messages and messages[0].get("role") == "system":
            messages[0] = {**messages[0], "content": messages[0]["content"] + "\n" + instruction}
        else:
            params["messages"] = [{"role": "system", "content": instruction}] + messages

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Execute a synchronous completion."""
        request = self._resolve_thinking(request)
        model = self._resolve_model(request)
        params = self._build_chat_params(request)

        if self._is_thinking_active(request, model):
            params.pop("tool_choice", None)

        self._apply_thinking_params(params, model, request)

        # DeepSeek structured output: JSON mode only (not full schema
        # enforcement).  V4 models (deepseek-v4-pro / deepseek-v4-flash)
        # support response_format + tools simultaneously via strict
        # mode; legacy models (deepseek-chat / deepseek-reasoner) return
        # empty content on tool-call turns which breaks JSON parsing,
        # so the JSON request is dropped (with a one-shot warning).
        has_tools = bool(params.get("tools"))
        supports_structured_with_tools = model in _THINKING_CAPABLE_MODELS
        if request.structured_output:
            if has_tools and not supports_structured_with_tools:
                self._warn_structured_drop_once()
            else:
                params["response_format"] = {"type": "json_object"}
                self._ensure_json_word_in_prompt(params, schema=request.structured_output.schema)

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
        supports_structured_with_tools = model in _THINKING_CAPABLE_MODELS
        if request.structured_output:
            if has_tools and not supports_structured_with_tools:
                self._warn_structured_drop_once()
            else:
                params["response_format"] = {"type": "json_object"}
            self._ensure_json_word_in_prompt(params, schema=request.structured_output.schema)

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
