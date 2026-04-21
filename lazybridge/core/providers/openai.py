"""OpenAI provider for LazyBridge.

Routes all requests through the Responses API (OpenAI's recommended path since 2025).
Chat Completions is retained only for Pydantic structured output (requires beta.parse()).

Default model: gpt-5.4. Pass model= to override.
"""

from __future__ import annotations

import json
import logging
import os
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

_logger = logging.getLogger(__name__)

try:
    import openai as _openai
except ImportError:
    _openai = None  # type: ignore

# Reasoning models that use `reasoning_effort` instead of `temperature`
_REASONING_MODELS = frozenset(
    {
        "o1",
        "o1-mini",
        "o1-pro",
        "o3",
        "o3-mini",
        "o3-pro",
        "o4-mini",
    }
)

# Responses API native tool type strings
_RESPONSES_NATIVE_MAP: dict[NativeTool, dict] = {
    NativeTool.WEB_SEARCH: {"type": "web_search_preview"},
    NativeTool.CODE_EXECUTION: {"type": "code_interpreter"},
    NativeTool.FILE_SEARCH: {"type": "file_search"},
    NativeTool.COMPUTER_USE: {"type": "computer_use_preview"},
}

# Effort level mapping: unified → OpenAI reasoning_effort
_EFFORT_MAP = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "xhigh",
}

# Price per 1M tokens (input, output). Approximate; verify at platform.openai.com/docs/pricing.
_PRICE_TABLE: dict[str, tuple[float, float]] = {
    "gpt-5": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.0),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4": (30.0, 60.0),
    "gpt-3.5": (0.50, 1.50),
    "o4-mini": (1.10, 4.40),
    "o3-mini": (1.10, 4.40),
    "o3": (10.0, 40.0),
    "o1-mini": (3.0, 12.0),
    "o1-pro": (150.0, 600.0),
    "o1": (15.0, 60.0),
}


def _safe_json_loads(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        _logger.warning(
            "Failed to parse tool-call arguments as JSON (%s). "
            "Raw arguments stored under '_raw_arguments'. Raw: %.200s",
            exc,
            raw,
        )
        return {"_raw_arguments": raw}


class OpenAIProvider(BaseProvider):
    """OpenAI provider.

    Supports:
    - Chat Completions (standard + function calling + structured outputs)
    - Responses API (native tools: web_search, code_interpreter, file_search, computer_use)
    - Reasoning effort control for o-series and gpt-5.4+ models
    - Streaming
    """

    default_model = "gpt-5.4"

    # Tier aliases (audit F2) — see the matrix in lazy_wiki/human/agents.md.
    _TIER_ALIASES = {
        "top": "gpt-5.4",
        "expensive": "gpt-5",
        "medium": "gpt-4o",
        "cheap": "gpt-4o-mini",
        "super_cheap": "gpt-3.5-turbo",
    }
    _FALLBACKS = {
        "gpt-5.4": ["gpt-5", "gpt-4o"],
        "gpt-5": ["gpt-4o", "gpt-4-turbo"],
        "gpt-4o": ["gpt-4-turbo", "gpt-3.5-turbo"],
        "gpt-4o-mini": ["gpt-3.5-turbo"],
    }
    supported_native_tools: frozenset[NativeTool] = frozenset(
        {
            NativeTool.WEB_SEARCH,
            NativeTool.CODE_EXECUTION,
            NativeTool.FILE_SEARCH,
            NativeTool.COMPUTER_USE,
        }
    )

    def _compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float | None:
        model_l = model.lower()
        for key, (in_price, out_price) in _PRICE_TABLE.items():
            if key in model_l:
                return (input_tokens * in_price + output_tokens * out_price) / 1_000_000
        return None

    def get_default_max_tokens(self, model: str | None = None) -> int:
        """Return the default max_tokens for the given model."""
        resolved = (model or self.model or self.default_model or "").lower()
        if resolved.startswith("gpt-5"):
            return 128_000
        if resolved.startswith("gpt-4.1"):
            return 32_768
        if resolved.startswith("gpt-4o"):
            return 16_384
        if self._is_reasoning_model(resolved):
            if resolved.startswith("o1") and "mini" in resolved:
                return 65_536  # o1-mini
            return 100_000  # o1, o1-pro, o3, o3-mini, o4-mini
        return 16_384

    def _init_client(self, **kwargs) -> None:
        if _openai is None:
            raise ImportError("openai package not installed. Run: pip install openai")
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key not found. Set the OPENAI_API_KEY environment "
                "variable, or pass api_key= to OpenAIProvider."
            )
        base_url = kwargs.pop("base_url", None)
        self._client = _openai.OpenAI(api_key=key, base_url=base_url, **kwargs)
        self._async_client = _openai.AsyncOpenAI(api_key=key, base_url=base_url, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_reasoning_model(self, model: str) -> bool:
        return model in _REASONING_MODELS or model.startswith(("o1", "o3", "o4", "gpt-5"))

    @staticmethod
    def _populate_reasoning_tokens(usage: UsageStats, raw_usage: Any) -> UsageStats:
        if raw_usage and hasattr(raw_usage, "completion_tokens_details"):
            details = raw_usage.completion_tokens_details
            if details and hasattr(details, "reasoning_tokens"):
                usage.thinking_tokens = details.reasoning_tokens or 0
        return usage

    def _messages_to_openai(self, request: CompletionRequest) -> list[dict]:
        messages: list[dict[str, Any]] = []
        # Prepend system prompt as system message if provided
        if request.system:
            messages.append({"role": "system", "content": request.system})
        for msg in request.messages:
            if msg.role == Role.SYSTEM:
                # Already handled above, or inline
                if not request.system:
                    messages.append({"role": "system", "content": msg.to_text()})
                continue
            if isinstance(msg.content, str):
                messages.append({"role": msg.role.value, "content": msg.content})
            else:
                from lazybridge.core.types import (
                    ImageContent,
                    TextContent,
                    ThinkingContent,
                    ToolResultContent,
                    ToolUseContent,
                )

                # Build content list for multimodal messages
                parts: list[dict[str, Any]] = []
                tool_calls_in_msg: list[dict[str, Any]] = []
                tool_results_in_msg: list[dict[str, Any]] = []
                for block in msg.content:
                    if isinstance(block, TextContent):
                        parts.append({"type": "text", "text": block.text})
                    elif isinstance(block, ThinkingContent):
                        pass  # OpenAI doesn't expose reasoning in messages
                    elif isinstance(block, ImageContent):
                        if block.url:
                            parts.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": block.url},
                                }
                            )
                        elif block.base64_data:
                            parts.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{block.media_type};base64,{block.base64_data}"},
                                }
                            )
                    elif isinstance(block, ToolUseContent):
                        tool_calls_in_msg.append(
                            {
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": json.dumps(block.input),
                                },
                            }
                        )
                    elif isinstance(block, ToolResultContent):
                        tool_results_in_msg.append(
                            {
                                "tool_call_id": block.tool_use_id,
                                "content": block.content
                                if isinstance(block.content, str)
                                else json.dumps(block.content),
                            }
                        )

                if tool_results_in_msg:
                    if parts:  # text present in the same message: emit it first
                        messages.append({"role": msg.role.value, "content": parts})
                    for tr in tool_results_in_msg:
                        messages.append({"role": "tool", **tr})
                elif tool_calls_in_msg:
                    m: dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls_in_msg}
                    if parts:
                        m["content"] = parts
                    messages.append(m)
                else:
                    messages.append(
                        {
                            "role": msg.role.value,
                            "content": parts[0]["text"]
                            if (len(parts) == 1 and parts[0].get("type") == "text")
                            else (parts if parts else ""),
                        }
                    )
        return messages

    def _build_function_tools(self, request: CompletionRequest) -> list[dict]:
        tools = []
        for t in request.tools:
            tool_def: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            if t.strict:
                tool_def["function"]["strict"] = True
            tools.append(tool_def)
        return tools

    def _build_chat_params(self, request: CompletionRequest) -> dict[str, Any]:
        model = self._resolve_model(request)
        params: dict[str, Any] = {
            "model": model,
            "messages": self._messages_to_openai(request),
        }
        # max_tokens vs max_completion_tokens
        if self._is_reasoning_model(model):
            params["max_completion_tokens"] = request.max_tokens
        else:
            params["max_tokens"] = request.max_tokens
            if request.temperature is not None:
                params["temperature"] = request.temperature

        # Reasoning effort
        if request.thinking and request.thinking.enabled:
            effort = _EFFORT_MAP.get(request.thinking.effort, request.thinking.effort)
            params["reasoning_effort"] = effort

        tools = self._build_function_tools(request)
        if tools:
            params["tools"] = tools
            if request.tool_choice:
                if request.tool_choice in ("auto", "none", "required"):
                    params["tool_choice"] = request.tool_choice
                else:
                    params["tool_choice"] = {
                        "type": "function",
                        "function": {"name": request.tool_choice},
                    }

        return params

    def _messages_to_responses_input(self, request: CompletionRequest) -> list[dict]:
        """Convert conversation history to Responses API input format.

        The Responses API uses a flat list where tool calls and results are
        top-level typed items, NOT role-based messages like Chat Completions:
          - function call (model → tool):  {"type": "function_call", "call_id", "name", "arguments"}
          - function result (tool → model): {"type": "function_call_output", "call_id", "output"}
          - text messages: {"role": "user|assistant|system", "content": ...}
        """
        from lazybridge.core.types import (
            ImageContent,
            TextContent,
            ThinkingContent,
            ToolResultContent,
            ToolUseContent,
        )

        items: list[dict[str, Any]] = []
        if request.system:
            items.append({"role": "system", "content": request.system})
        for msg in request.messages:
            if msg.role == Role.SYSTEM:
                if not request.system:
                    items.append({"role": "system", "content": msg.to_text()})
                continue
            if isinstance(msg.content, str):
                items.append({"role": msg.role.value, "content": msg.content})
                continue

            text_parts: list[dict] = []
            for block in msg.content:
                if isinstance(block, TextContent):
                    text_parts.append({"type": "input_text", "text": block.text})
                elif isinstance(block, ThinkingContent):
                    pass  # not exposed in Responses API history
                elif isinstance(block, ImageContent):
                    if block.url:
                        text_parts.append({"type": "input_image", "image_url": block.url})
                    elif block.base64_data:
                        text_parts.append(
                            {
                                "type": "input_image",
                                "image_url": f"data:{block.media_type};base64,{block.base64_data}",
                            }
                        )
                elif isinstance(block, ToolUseContent):
                    # Model decided to call a function — emit any preceding text first
                    if text_parts:
                        items.append(
                            {
                                "role": msg.role.value,
                                "content": text_parts[0]["text"] if len(text_parts) == 1 else text_parts,
                            }
                        )
                        text_parts = []
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": block.id,
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        }
                    )
                elif isinstance(block, ToolResultContent):
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": block.tool_use_id,
                            "output": block.content if isinstance(block.content, str) else json.dumps(block.content),
                        }
                    )

            if text_parts:
                items.append(
                    {
                        "role": msg.role.value,
                        "content": text_parts[0]["text"] if len(text_parts) == 1 else text_parts,
                    }
                )
        return items

    def _build_responses_params(self, request: CompletionRequest) -> dict[str, Any]:
        """Build params for the Responses API (OpenAI's recommended path)."""
        model = self._resolve_model(request)
        native = self._check_native_tools(request.native_tools)

        tools: list[dict] = [_RESPONSES_NATIVE_MAP[nt] for nt in native]
        # User-defined function tools (flattened format required by Responses API)
        for t in request.tools:
            tools.append(
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                    **({"strict": True} if t.strict else {}),
                }
            )

        params: dict[str, Any] = {
            "model": model,
            "input": self._messages_to_responses_input(request),
        }
        if tools:
            params["tools"] = tools
            if request.tool_choice:
                if request.tool_choice in ("auto", "none", "required"):
                    params["tool_choice"] = request.tool_choice
                else:
                    # Responses API specific tool: {"type": "function", "name": "..."}
                    params["tool_choice"] = {"type": "function", "name": request.tool_choice}
        if request.max_tokens:
            params["max_output_tokens"] = request.max_tokens
        if request.thinking and request.thinking.enabled:
            params["reasoning"] = {"effort": _EFFORT_MAP.get(request.thinking.effort, "high")}
        elif request.temperature is not None:
            params["temperature"] = request.temperature
        if "store" in request.extra:
            params["store"] = request.extra["store"]
        if "previous_response_id" in request.extra:
            params["previous_response_id"] = request.extra["previous_response_id"]

        # Structured output via Responses API text.format
        if request.structured_output and isinstance(request.structured_output.schema, dict):
            from lazybridge.core.structured import normalize_json_schema

            schema = normalize_json_schema(request.structured_output.schema)
            strict = request.structured_output.strict
            params["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "output",
                    "schema": schema,
                    "strict": strict,
                }
            }
        return params

    def _parse_chat_response(self, response: Any) -> CompletionResponse:
        model_name = getattr(response, "model", "") or ""
        if not response.choices:
            _logger.warning("OpenAI response has no choices (content filter, quota, or API error).")
            usage = UsageStats(
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
            )
            usage.cost_usd = self._compute_cost(model_name, usage.input_tokens, usage.output_tokens)
            return CompletionResponse(
                content="",
                tool_calls=[],
                stop_reason="error",
                model=getattr(response, "model", None),
                usage=usage,
                raw=response,
            )
        choice = response.choices[0]
        msg = choice.message
        content = msg.content or ""
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
        usage.cost_usd = self._compute_cost(model_name, usage.input_tokens, usage.output_tokens)
        return CompletionResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason or "end_turn",
            model=response.model,
            usage=usage,
            raw=response,
        )

    def _parse_responses_response(self, response: Any) -> CompletionResponse:
        """Parse Responses API output."""
        content = ""
        tool_calls = []
        for item in response.output:
            if item.type == "message":
                for block in item.content:
                    if block.type == "output_text":
                        content += block.text
            elif item.type == "function_call":
                tool_calls.append(
                    ToolCall(
                        id=item.call_id,
                        name=item.name,
                        arguments=_safe_json_loads(item.arguments),
                    )
                )
        usage = UsageStats(
            input_tokens=getattr(response.usage, "input_tokens", 0),
            output_tokens=getattr(response.usage, "output_tokens", 0),
        )
        usage = self._populate_reasoning_tokens(usage, response.usage)
        model_name = getattr(response, "model", "") or ""
        usage.cost_usd = self._compute_cost(model_name, usage.input_tokens, usage.output_tokens)
        return CompletionResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason=self._responses_stop_reason(response, tool_calls),
            model=getattr(response, "model", None),
            usage=usage,
            raw=response,
            grounding_sources=self._extract_grounding_from_output(response.output),
        )

    @staticmethod
    def _extract_grounding_from_output(output: Any) -> list[GroundingSource]:
        """Extract url_citation annotations from Responses API output items."""
        sources: list[GroundingSource] = []
        for item in output or []:
            if getattr(item, "type", None) == "message":
                for block in getattr(item, "content", []) or []:
                    if getattr(block, "type", None) == "output_text":
                        for ann in getattr(block, "annotations", []) or []:
                            if getattr(ann, "type", None) == "url_citation":
                                sources.append(
                                    GroundingSource(
                                        url=getattr(ann, "url", "") or "",
                                        title=getattr(ann, "title", None),
                                    )
                                )
        return sources

    @staticmethod
    def _responses_stop_reason(response: Any, tool_calls: list) -> str:
        """Map OpenAI Responses API status/stop fields to a normalised stop_reason."""
        if tool_calls:
            return "tool_use"
        status = getattr(response, "status", None)
        if status == "incomplete":
            return "max_tokens"
        if status == "failed":
            return "error"
        return "end_turn"

    def _stream_responses_api(self, params: dict) -> Iterator[StreamChunk]:
        """Stream from the Responses API with parsing of official event types."""
        fc_names: dict[str, str] = {}  # call_id → function name
        fc_args: dict[str, str] = {}  # call_id → accumulated args JSON string
        completed_response = None

        for event in self._client.responses.create(**params):
            etype = getattr(event, "type", None)

            if etype == "response.output_text.delta":
                delta = getattr(event, "delta", "") or ""
                if delta:
                    yield StreamChunk(delta=delta)

            elif etype == "response.output_item.added":
                # Capture function call metadata (name, call_id) as soon as the
                # item is added, before any argument deltas arrive.
                item = getattr(event, "item", None)
                if item and getattr(item, "type", None) == "function_call":
                    call_id = getattr(item, "call_id", None) or ""
                    name = getattr(item, "name", "") or ""
                    if call_id:
                        fc_names[call_id] = name
                        fc_args.setdefault(call_id, "")

            elif etype == "response.function_call_arguments.delta":
                call_id = getattr(event, "call_id", None) or ""
                delta = getattr(event, "delta", "") or ""
                if call_id:
                    fc_args[call_id] = fc_args.get(call_id, "") + delta

            elif etype == "response.completed" or etype == "response.failed":
                completed_response = getattr(event, "response", None)

        # Build final chunk
        tool_calls = []
        for call_id, args_str in fc_args.items():
            name = fc_names.get(call_id, call_id)
            arguments = _safe_json_loads(args_str) if args_str else {}
            tool_calls.append(ToolCall(id=call_id, name=name, arguments=arguments))

        usage = None
        grounding_sources = []
        if completed_response is not None:
            u = getattr(completed_response, "usage", None)
            if u:
                usage = UsageStats(
                    input_tokens=getattr(u, "input_tokens", 0) or 0,
                    output_tokens=getattr(u, "output_tokens", 0) or 0,
                )
                usage = self._populate_reasoning_tokens(usage, u)
                usage.cost_usd = self._compute_cost(
                    getattr(completed_response, "model", "") or "",
                    usage.input_tokens,
                    usage.output_tokens,
                )
            grounding_sources = self._extract_grounding_from_output(getattr(completed_response, "output", []))

        yield StreamChunk(
            stop_reason=self._responses_stop_reason(completed_response, tool_calls),
            tool_calls=tool_calls,
            usage=usage,
            is_final=True,
            grounding_sources=grounding_sources,
        )

    async def _astream_responses_api(self, params: dict) -> AsyncIterator[StreamChunk]:
        """Async version of _stream_responses_api."""
        fc_names: dict[str, str] = {}
        fc_args: dict[str, str] = {}
        completed_response = None

        async for event in await self._async_client.responses.create(**params):
            etype = getattr(event, "type", None)

            if etype == "response.output_text.delta":
                delta = getattr(event, "delta", "") or ""
                if delta:
                    yield StreamChunk(delta=delta)

            elif etype == "response.output_item.added":
                item = getattr(event, "item", None)
                if item and getattr(item, "type", None) == "function_call":
                    call_id = getattr(item, "call_id", None) or ""
                    name = getattr(item, "name", "") or ""
                    if call_id:
                        fc_names[call_id] = name
                        fc_args.setdefault(call_id, "")

            elif etype == "response.function_call_arguments.delta":
                call_id = getattr(event, "call_id", None) or ""
                delta = getattr(event, "delta", "") or ""
                if call_id:
                    fc_args[call_id] = fc_args.get(call_id, "") + delta

            elif etype == "response.completed" or etype == "response.failed":
                completed_response = getattr(event, "response", None)

        tool_calls = []
        for call_id, args_str in fc_args.items():
            name = fc_names.get(call_id, call_id)
            arguments = _safe_json_loads(args_str) if args_str else {}
            tool_calls.append(ToolCall(id=call_id, name=name, arguments=arguments))

        usage = None
        grounding_sources = []
        if completed_response is not None:
            u = getattr(completed_response, "usage", None)
            if u:
                usage = UsageStats(
                    input_tokens=getattr(u, "input_tokens", 0) or 0,
                    output_tokens=getattr(u, "output_tokens", 0) or 0,
                )
                usage = self._populate_reasoning_tokens(usage, u)
                usage.cost_usd = self._compute_cost(
                    getattr(completed_response, "model", "") or "",
                    usage.input_tokens,
                    usage.output_tokens,
                )
            grounding_sources = self._extract_grounding_from_output(getattr(completed_response, "output", []))

        yield StreamChunk(
            stop_reason=self._responses_stop_reason(completed_response, tool_calls),
            tool_calls=tool_calls,
            usage=usage,
            is_final=True,
            grounding_sources=grounding_sources,
        )

    def _use_responses_api(self, request: CompletionRequest) -> bool:
        # Responses API is the default for all requests.
        # Exception: Pydantic structured output stays on Chat Completions because
        # it requires beta.chat.completions.parse() for native object hydration.
        return not (request.structured_output and not isinstance(request.structured_output.schema, dict))

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Execute a synchronous completion."""
        if self._use_responses_api(request):
            params = self._build_responses_params(request)
            response = self._client.responses.create(**params)
            resp = self._parse_responses_response(response)
            if request.structured_output:
                from lazybridge.core.structured import apply_structured_validation

                apply_structured_validation(resp, resp.content, request.structured_output.schema)
            return resp

        params = self._build_chat_params(request)

        # Pydantic schema: use native beta.parse() on Chat Completions
        if request.structured_output and not isinstance(request.structured_output.schema, dict):
            from lazybridge.core.structured import apply_structured_validation

            schema = request.structured_output.schema
            params["response_format"] = schema
            response = self._client.beta.chat.completions.parse(**params)
            resp = self._parse_chat_response(response)
            native_parsed = response.choices[0].message.parsed if response.choices else None
            if native_parsed is not None:
                resp.parsed = native_parsed
                resp.validated = True
            else:
                apply_structured_validation(resp, resp.content, schema)
            return resp

        response = self._client.chat.completions.create(**params)
        return self._parse_chat_response(response)

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        """Stream a completion, yielding StreamChunk objects."""
        if self._use_responses_api(request):
            params = self._build_responses_params(request)
            params["stream"] = True
            text_accum = ""
            for chunk in self._stream_responses_api(params):
                if not chunk.is_final:
                    text_accum += chunk.delta
                    yield chunk
                else:
                    if request.structured_output:
                        from lazybridge.core.structured import apply_structured_validation

                        apply_structured_validation(chunk, text_accum, request.structured_output.schema)
                    yield chunk
            return

        params = self._build_chat_params(request)
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}

        # For Pydantic schema streaming, instruct the model to output JSON.
        # (beta.chat.completions.parse() is not available in streaming mode.)
        # The advertised schema is lost on this path — the model emits some
        # JSON, and structured.apply_structured_validation re-validates
        # against our subset validator on the final chunk.  Warn the
        # caller so this "best-effort" behaviour isn't a surprise
        # (audit M13).
        if request.structured_output and not isinstance(request.structured_output.schema, dict):
            import warnings as _warnings

            _warnings.warn(
                "OpenAI streaming with a Pydantic output_schema is best-effort: "
                "the schema is enforced at validation time, not by the model's "
                "response_format. Expect occasional parse/validation failures on "
                "long completions.  Use stream=False for strict Pydantic parsing.",
                UserWarning,
                stacklevel=4,
            )
            params["response_format"] = {"type": "json_object"}

        text_accum = ""
        tool_call_accum: dict[int, dict] = {}
        final_chunk: StreamChunk | None = None
        final_usage: UsageStats | None = None
        for chunk in self._client.chat.completions.create(**params):
            if chunk.usage:
                final_usage = UsageStats(
                    input_tokens=chunk.usage.prompt_tokens,
                    output_tokens=chunk.usage.completion_tokens,
                )
                final_usage = self._populate_reasoning_tokens(final_usage, chunk.usage)
                final_usage.cost_usd = self._compute_cost(
                    getattr(chunk, "model", "") or "", final_usage.input_tokens, final_usage.output_tokens
                )
            choice = chunk.choices[0] if chunk.choices else None
            if choice and choice.delta.content:
                text_accum += choice.delta.content
                yield StreamChunk(delta=choice.delta.content)
            if choice and choice.delta.tool_calls:
                for tc in choice.delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_call_accum:
                        tool_call_accum[idx] = {"id": tc.id or "", "name": tc.function.name or "", "args": ""}
                    if tc.id:
                        tool_call_accum[idx]["id"] = tc.id
                    if tc.function.name:
                        tool_call_accum[idx]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_call_accum[idx]["args"] += tc.function.arguments
            if choice and choice.finish_reason:
                tool_calls = [
                    ToolCall(id=v["id"], name=v["name"], arguments=_safe_json_loads(v["args"]))
                    for v in tool_call_accum.values()
                ]
                final_chunk = StreamChunk(
                    stop_reason=choice.finish_reason,
                    tool_calls=tool_calls,
                    usage=final_usage,
                    is_final=True,
                )
                if request.structured_output:
                    from lazybridge.core.structured import apply_structured_validation

                    apply_structured_validation(final_chunk, text_accum, request.structured_output.schema)
        if final_chunk is None:
            # Stream ended without a finish_reason (interrupted / truncated).
            # Emit a best-effort final chunk so consumers that depend on the
            # is_final marker (usage, tool_calls, structured-output validation)
            # don't hang forever waiting for it.
            tool_calls = [
                ToolCall(id=v["id"], name=v["name"], arguments=_safe_json_loads(v["args"]))
                for v in tool_call_accum.values()
            ]
            final_chunk = StreamChunk(
                stop_reason="incomplete",
                tool_calls=tool_calls,
                usage=final_usage,
                is_final=True,
            )
            if request.structured_output:
                from lazybridge.core.structured import apply_structured_validation

                apply_structured_validation(final_chunk, text_accum, request.structured_output.schema)
        final_chunk.usage = final_chunk.usage or final_usage
        yield final_chunk

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        """Async completion."""
        if self._use_responses_api(request):
            params = self._build_responses_params(request)
            response = await self._async_client.responses.create(**params)
            resp = self._parse_responses_response(response)
            if request.structured_output:
                from lazybridge.core.structured import apply_structured_validation

                apply_structured_validation(resp, resp.content, request.structured_output.schema)
            return resp

        params = self._build_chat_params(request)

        # Pydantic schema: use native beta.parse() on Chat Completions
        if request.structured_output and not isinstance(request.structured_output.schema, dict):
            from lazybridge.core.structured import apply_structured_validation

            schema = request.structured_output.schema
            params["response_format"] = schema
            response = await self._async_client.beta.chat.completions.parse(**params)
            resp = self._parse_chat_response(response)
            native_parsed = response.choices[0].message.parsed if response.choices else None
            if native_parsed is not None:
                resp.parsed = native_parsed
                resp.validated = True
            else:
                apply_structured_validation(resp, resp.content, schema)
            return resp

        response = await self._async_client.chat.completions.create(**params)
        return self._parse_chat_response(response)

    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        """Async streaming completion."""
        if self._use_responses_api(request):
            params = self._build_responses_params(request)
            params["stream"] = True
            text_accum = ""
            async for chunk in self._astream_responses_api(params):
                if not chunk.is_final:
                    text_accum += chunk.delta
                    yield chunk
                else:
                    if request.structured_output:
                        from lazybridge.core.structured import apply_structured_validation

                        apply_structured_validation(chunk, text_accum, request.structured_output.schema)
                    yield chunk
            return

        params = self._build_chat_params(request)
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}

        if request.structured_output and not isinstance(request.structured_output.schema, dict):
            params["response_format"] = {"type": "json_object"}

        text_accum = ""
        tool_call_accum: dict[int, dict] = {}
        final_chunk: StreamChunk | None = None
        final_usage: UsageStats | None = None
        async for chunk in await self._async_client.chat.completions.create(**params):
            if chunk.usage:
                final_usage = UsageStats(
                    input_tokens=chunk.usage.prompt_tokens,
                    output_tokens=chunk.usage.completion_tokens,
                )
                final_usage = self._populate_reasoning_tokens(final_usage, chunk.usage)
                final_usage.cost_usd = self._compute_cost(
                    getattr(chunk, "model", "") or "", final_usage.input_tokens, final_usage.output_tokens
                )
            choice = chunk.choices[0] if chunk.choices else None
            if choice and choice.delta.content:
                text_accum += choice.delta.content
                yield StreamChunk(delta=choice.delta.content)
            if choice and choice.delta.tool_calls:
                for tc in choice.delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_call_accum:
                        tool_call_accum[idx] = {"id": tc.id or "", "name": tc.function.name or "", "args": ""}
                    if tc.id:
                        tool_call_accum[idx]["id"] = tc.id
                    if tc.function.name:
                        tool_call_accum[idx]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_call_accum[idx]["args"] += tc.function.arguments
            if choice and choice.finish_reason:
                tool_calls = [
                    ToolCall(id=v["id"], name=v["name"], arguments=_safe_json_loads(v["args"]))
                    for v in tool_call_accum.values()
                ]
                final_chunk = StreamChunk(
                    stop_reason=choice.finish_reason,
                    tool_calls=tool_calls,
                    usage=final_usage,
                    is_final=True,
                )
                if request.structured_output:
                    from lazybridge.core.structured import apply_structured_validation

                    apply_structured_validation(final_chunk, text_accum, request.structured_output.schema)
        if final_chunk is None:
            # Interrupted / truncated async stream — emit a best-effort final
            # chunk so awaiters don't hang.
            tool_calls = [
                ToolCall(id=v["id"], name=v["name"], arguments=_safe_json_loads(v["args"]))
                for v in tool_call_accum.values()
            ]
            final_chunk = StreamChunk(
                stop_reason="incomplete",
                tool_calls=tool_calls,
                usage=final_usage,
                is_final=True,
            )
            if request.structured_output:
                from lazybridge.core.structured import apply_structured_validation

                apply_structured_validation(final_chunk, text_accum, request.structured_output.schema)
        final_chunk.usage = final_chunk.usage or final_usage
        yield final_chunk
