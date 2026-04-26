"""LiteLLM bridge — single provider mapping onto LiteLLM's 100+ model catalog.

Users opt in via the ``litellm/`` model prefix::

    Agent("litellm/groq/llama-3.3-70b-versatile")
    Agent("litellm/mistral/mistral-large-latest")
    Agent("litellm/ollama/llama3")

The prefix is stripped before the request hits ``litellm.completion()``, so
LiteLLM sees its own native ``provider/model`` syntax.  The native LazyBridge
providers (Anthropic, OpenAI, Google, DeepSeek) still handle their models
directly and keep their power features (thinking, prompt caching, native
web search).  LiteLLM is the catch-all for everything else.

Trade-offs you accept when routing through this bridge:

* **Native tools** (``WEB_SEARCH``, ``CODE_EXECUTION``, ...) — NOT forwarded.
  LiteLLM normalises to the OpenAI shape which has no way to express them.
  Ask for them and they're silently dropped with a warning.
* **Thinking / extended reasoning** — not wired through; use the native
  Anthropic / OpenAI adapters if you need it.
* **Prompt caching** — not wired through; same reason.
* **Async** — first-class: we call ``litellm.acompletion()`` directly.
* **Cost tracking** — forwarded from ``response._hidden_params["response_cost"]``
  when LiteLLM can compute it; ``None`` otherwise.

Install: ``pip install lazybridge[litellm]``.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import AsyncIterator, Iterator
from typing import Any

from lazybridge.core.providers.base import BaseProvider
from lazybridge.core.types import (
    CompletionRequest,
    CompletionResponse,
    ContentBlock,
    ImageContent,
    Role,
    StreamChunk,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolDefinition,
    ToolResultContent,
    ToolUseContent,
    UsageStats,
)

_logger = logging.getLogger(__name__)

#: Model-string prefix users type to route a request through this bridge.
#: Stripped before the model name reaches ``litellm.completion``.
_PREFIX = "litellm/"


def _strip_prefix(model: str) -> str:
    """Drop the ``litellm/`` prefix so LiteLLM sees its native model syntax."""
    if model.startswith(_PREFIX):
        return model[len(_PREFIX) :]
    return model


def _safe_json_loads(raw: str) -> dict[str, Any]:
    """Parse tool-call arguments. Return raw string on failure — never raise."""
    if not raw:
        return {}
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        _logger.warning(
            "LiteLLM tool-call arguments failed JSON parse (%s); storing under _raw_arguments. Raw: %.200s",
            exc,
            raw,
        )
        return {"_raw_arguments": raw}
    return result if isinstance(result, dict) else {"_raw_arguments": raw}


def _content_blocks_to_openai_parts(
    blocks: list[ContentBlock],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split a content-block list into text/image parts, assistant tool_calls, and tool_results.

    Returned tuple ``(parts, tool_calls, tool_results)`` mirrors the three
    distinct OpenAI message shapes a single content list can produce:
      * ``parts``        — multimodal content array (text + images)
      * ``tool_calls``   — assistant-side emitted calls (one dict per call)
      * ``tool_results`` — tool-side results (one dict per result)
    """
    parts: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, TextContent):
            parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ThinkingContent):
            # OpenAI-shape wire format has no slot for reasoning blocks.
            # LiteLLM normalises to OpenAI shape so we drop them too.
            pass
        elif isinstance(block, ImageContent):
            if block.url:
                parts.append({"type": "image_url", "image_url": {"url": block.url}})
            elif block.base64_data:
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{block.media_type};base64,{block.base64_data}"},
                    }
                )
        elif isinstance(block, ToolUseContent):
            tool_calls.append(
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
            tool_results.append(
                {
                    "tool_call_id": block.tool_use_id,
                    "content": block.content if isinstance(block.content, str) else json.dumps(block.content),
                }
            )
    return parts, tool_calls, tool_results


def _messages_to_openai(request: CompletionRequest) -> list[dict[str, Any]]:
    """Convert LazyBridge's Message list into the OpenAI wire format LiteLLM consumes."""
    messages: list[dict[str, Any]] = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    for msg in request.messages:
        if msg.role == Role.SYSTEM:
            if not request.system:
                messages.append({"role": "system", "content": msg.to_text()})
            continue
        if isinstance(msg.content, str):
            # Role.TOOL with a plain string has no tool_call_id to thread
            # back to an assistant call — demote to user so the text is at
            # least visible to the model.
            role = "user" if msg.role == Role.TOOL else msg.role.value
            messages.append({"role": role, "content": msg.content})
            continue

        parts, tool_calls, tool_results = _content_blocks_to_openai_parts(msg.content)

        if tool_results:
            # Tool results MUST be separate role=tool messages in OpenAI
            # format. If there was also text in the same LazyBridge message,
            # emit it first as a user message (can't mix on role=tool).
            if parts:
                messages.append({"role": msg.role.value, "content": parts})
            for tr in tool_results:
                messages.append({"role": "tool", **tr})
        elif tool_calls:
            out: dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls}
            if parts:
                out["content"] = parts
            messages.append(out)
        elif parts:
            # Flatten ``[{type:text, text:"x"}]`` → ``"x"`` for the common case —
            # a few providers behind LiteLLM still choke on unnecessary arrays.
            if len(parts) == 1 and parts[0].get("type") == "text":
                messages.append({"role": msg.role.value, "content": parts[0]["text"]})
            else:
                messages.append({"role": msg.role.value, "content": parts})
        else:
            messages.append({"role": msg.role.value, "content": ""})
    return messages


def _tools_to_openai(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Emit OpenAI function-tool schema. LiteLLM forwards this shape unchanged."""
    result = []
    for t in tools:
        entry: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        if t.strict:
            entry["function"]["strict"] = True
        result.append(entry)
    return result


def _tool_choice_to_openai(choice: str, tools: list[ToolDefinition]) -> Any:
    """Map LazyBridge tool_choice keyword / tool-name → OpenAI tool_choice field."""
    if choice in ("auto", "none", "required"):
        return choice
    if choice == "any":
        # OpenAI spells this "required".
        return "required"
    # Must be a tool name (validated up-front by CompletionRequest.__post_init__).
    return {"type": "function", "function": {"name": choice}}


class LiteLLMProvider(BaseProvider):
    """Bridge LazyBridge to LiteLLM's 100+ provider catalog.

    Constructed implicitly when the model string starts with ``litellm/``::

        Agent("litellm/mistral/mistral-large-latest")

    The prefix is stripped before ``litellm.completion`` sees the model.
    Install ``pip install lazybridge[litellm]`` to pull the dependency.
    """

    default_model = "litellm/gpt-4o-mini"

    #: No native-tools through the bridge — LiteLLM normalises to the
    #: OpenAI wire shape which has no slot for WEB_SEARCH / CODE_EXECUTION.
    #: Callers that need those should use a native provider.
    supported_native_tools = frozenset()

    def _init_client(self, **kwargs: Any) -> None:
        try:
            import litellm
        except ImportError as exc:
            raise ImportError(
                "LiteLLMProvider requires the litellm package. Install with: pip install 'lazybridge[litellm]'"
            ) from exc
        self._litellm = litellm
        # Forward extra kwargs to litellm module-level config (e.g.
        # drop_params, set_verbose, telemetry). Anything unknown is
        # stored silently on the module; callers are responsible for
        # knowing litellm's surface.
        for key, value in kwargs.items():
            if hasattr(litellm, key):
                setattr(litellm, key, value)

    # ------------------------------------------------------------------
    # Request → LiteLLM params
    # ------------------------------------------------------------------

    def _build_params(self, request: CompletionRequest) -> dict[str, Any]:
        """Translate a CompletionRequest into litellm.completion kwargs."""
        if request.native_tools:
            warnings.warn(
                f"LiteLLMProvider: native_tools={list(request.native_tools)!r} "
                f"is not forwarded through the bridge (LiteLLM normalises to "
                f"the OpenAI shape). Use a native provider for these.",
                UserWarning,
                stacklevel=3,
            )

        resolved = _strip_prefix(self._resolve_model(request))
        params: dict[str, Any] = {
            "model": resolved,
            "messages": _messages_to_openai(request),
            "max_tokens": request.max_tokens,
        }
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.tools:
            params["tools"] = _tools_to_openai(request.tools)
            if request.tool_choice is not None:
                params["tool_choice"] = _tool_choice_to_openai(
                    request.tool_choice,
                    request.tools,
                )
        if self.api_key:
            # When set explicitly, hand LiteLLM the key. Otherwise it reads
            # from the appropriate env var (OPENAI_API_KEY, ANTHROPIC_API_KEY,
            # GROQ_API_KEY, ...) based on the model's inferred provider.
            params["api_key"] = self.api_key
        if request.extra:
            # Escape hatch for provider-specific kwargs LazyBridge doesn't
            # model yet (e.g. top_p, presence_penalty, frequency_penalty,
            # seed, response_format, stop, user).
            params.update(request.extra)
        return params

    # ------------------------------------------------------------------
    # Response → CompletionResponse
    # ------------------------------------------------------------------

    def _convert_response(self, resp: Any) -> CompletionResponse:
        """Convert a litellm ModelResponse into the unified CompletionResponse."""
        choices = getattr(resp, "choices", []) or []
        if not choices:
            return CompletionResponse(content="", raw=resp)
        choice = choices[0]
        message = getattr(choice, "message", None)
        content = getattr(message, "content", "") or "" if message else ""
        stop_reason = getattr(choice, "finish_reason", "end_turn") or "end_turn"

        tool_calls: list[ToolCall] = []
        for raw_tc in getattr(message, "tool_calls", None) or []:
            fn = getattr(raw_tc, "function", None)
            if fn is None:
                continue
            tool_calls.append(
                ToolCall(
                    id=getattr(raw_tc, "id", ""),
                    name=getattr(fn, "name", ""),
                    arguments=_safe_json_loads(getattr(fn, "arguments", "") or ""),
                )
            )

        usage = self._extract_usage(resp)
        return CompletionResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            model=getattr(resp, "model", None),
            usage=usage,
            raw=resp,
        )

    def _extract_usage(self, resp: Any) -> UsageStats:
        raw_usage = getattr(resp, "usage", None)
        if raw_usage is None:
            return UsageStats()
        usage = UsageStats(
            input_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
        )
        # LiteLLM stuffs the computed cost under _hidden_params when it can
        # price the model; fall back to _compute_cost override otherwise.
        hidden = getattr(resp, "_hidden_params", None) or {}
        cost = hidden.get("response_cost") if isinstance(hidden, dict) else None
        if cost is not None:
            try:
                usage.cost_usd = float(cost)
            except (TypeError, ValueError):
                usage.cost_usd = None
        return usage

    # ------------------------------------------------------------------
    # Streaming chunk aggregation
    # ------------------------------------------------------------------

    def _reduce_stream_chunk(
        self,
        chunk: Any,
        partial_tool_calls: dict[int, dict[str, Any]],
    ) -> tuple[str, str | None, UsageStats | None]:
        """Read one LiteLLM stream chunk; update ``partial_tool_calls`` in place.

        Returns ``(delta_text, finish_reason, usage_on_this_chunk)``.
        Tool-call accumulation is shared across chunks via the mutable dict —
        each chunk carries deltas that we stitch together by index.
        """
        choices = getattr(chunk, "choices", []) or []
        if not choices:
            # Usage-only final chunk (some providers emit one after the
            # content stream completes).
            u = getattr(chunk, "usage", None)
            if u:
                return (
                    "",
                    None,
                    UsageStats(
                        input_tokens=getattr(u, "prompt_tokens", 0) or 0,
                        output_tokens=getattr(u, "completion_tokens", 0) or 0,
                    ),
                )
            return "", None, None

        choice = choices[0]
        delta_obj = getattr(choice, "delta", None)
        delta_text = (getattr(delta_obj, "content", None) or "") if delta_obj else ""
        finish_reason = getattr(choice, "finish_reason", None)

        for tc_delta in (getattr(delta_obj, "tool_calls", None) or []) if delta_obj else []:
            idx = getattr(tc_delta, "index", 0) or 0
            slot = partial_tool_calls.setdefault(idx, {"id": "", "name": "", "args": ""})
            tc_id = getattr(tc_delta, "id", None)
            if tc_id:
                slot["id"] = tc_id
            fn = getattr(tc_delta, "function", None)
            if fn is not None:
                name = getattr(fn, "name", None)
                if name:
                    slot["name"] = name
                args = getattr(fn, "arguments", None)
                if args:
                    slot["args"] += args

        usage = None
        u = getattr(chunk, "usage", None)
        if u:
            usage = UsageStats(
                input_tokens=getattr(u, "prompt_tokens", 0) or 0,
                output_tokens=getattr(u, "completion_tokens", 0) or 0,
            )
        return delta_text, finish_reason, usage

    def _finalise_stream(
        self,
        partial_tool_calls: dict[int, dict[str, Any]],
        finish_reason: str | None,
        usage: UsageStats | None,
    ) -> StreamChunk:
        """Build the final StreamChunk from accumulated state."""
        tool_calls = [
            ToolCall(
                id=slot["id"] or f"tc_{idx}",
                name=slot["name"],
                arguments=_safe_json_loads(slot["args"]),
            )
            for idx, slot in sorted(partial_tool_calls.items())
            if slot["name"]  # guard against malformed partials
        ]
        return StreamChunk(
            delta="",
            tool_calls=tool_calls,
            stop_reason=finish_reason or "end_turn",
            usage=usage,
            is_final=True,
        )

    # ------------------------------------------------------------------
    # Abstract-method implementations
    # ------------------------------------------------------------------

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        params = self._build_params(request)
        params["stream"] = False
        resp = self._litellm.completion(**params)
        return self._convert_response(resp)

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        params = self._build_params(request)
        params["stream"] = True
        iterator = self._litellm.completion(**params)
        partial: dict[int, dict[str, Any]] = {}
        last_usage: UsageStats | None = None
        last_finish: str | None = None
        for chunk in iterator:
            delta_text, finish_reason, usage = self._reduce_stream_chunk(chunk, partial)
            if delta_text:
                yield StreamChunk(delta=delta_text)
            if finish_reason:
                last_finish = finish_reason
            if usage is not None:
                last_usage = usage
        yield self._finalise_stream(partial, last_finish, last_usage)

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        params = self._build_params(request)
        params["stream"] = False
        resp = await self._litellm.acompletion(**params)
        return self._convert_response(resp)

    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        params = self._build_params(request)
        params["stream"] = True
        async_iterator = await self._litellm.acompletion(**params)
        partial: dict[int, dict[str, Any]] = {}
        last_usage: UsageStats | None = None
        last_finish: str | None = None
        async for chunk in async_iterator:
            delta_text, finish_reason, usage = self._reduce_stream_chunk(chunk, partial)
            if delta_text:
                yield StreamChunk(delta=delta_text)
            if finish_reason:
                last_finish = finish_reason
            if usage is not None:
                last_usage = usage
        yield self._finalise_stream(partial, last_finish, last_usage)
