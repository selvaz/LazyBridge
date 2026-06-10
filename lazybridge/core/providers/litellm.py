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
    AudioContent,
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

# MIME → short format string expected by OpenAI-shape audio input blocks.
_AUDIO_FORMAT: dict[str, str] = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
    "audio/ogg": "ogg",
    "audio/opus": "opus",
    "audio/webm": "webm",
    "audio/aac": "aac",
}

#: Model-string prefix users type to route a request through this bridge.
#: Stripped before the model name reaches ``litellm.completion``.
_PREFIX = "litellm/"


def _strip_prefix(model: str) -> str:
    """Drop the ``litellm/`` prefix so LiteLLM sees its native model syntax."""
    if model.startswith(_PREFIX):
        return model[len(_PREFIX) :]
    return model


def _safe_json_loads(raw: str) -> dict[str, Any]:
    """Parse tool-call arguments. Return a tagged dict on failure — never raise.

    On JSON-decode failure (or a non-object payload) we tag the result
    with ``_parse_error`` so :meth:`LLMEngine._exec_tool` surfaces a
    structured ``TOOL_ERROR`` instead of letting the tool fail later
    with a misleading "missing required field" message.
    """
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
        return {"_raw_arguments": raw, "_parse_error": str(exc)}
    if not isinstance(result, dict):
        return {
            "_raw_arguments": raw,
            "_parse_error": (f"tool-call arguments parsed as {type(result).__name__}, expected object"),
        }
    return result


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
        elif isinstance(block, AudioContent):
            if block.base64_data:
                fmt = _AUDIO_FORMAT.get((block.media_type or "").lower(), "mp3")
                parts.append(
                    {
                        "type": "input_audio",
                        "input_audio": {"data": block.base64_data, "format": fmt},
                    }
                )
            elif block.url:
                warnings.warn(
                    "LiteLLM (OpenAI wire format) audio requires base64 — URL "
                    f"audio ({block.url!r}) is not supported and was skipped. "
                    "Pass AudioContent.from_path() / from_bytes() instead.",
                    UserWarning,
                    stacklevel=3,
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
                messages.append({"role": "user", "content": parts})
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


def _tool_choice_to_openai(choice: str) -> Any:
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

    #: No class default — the bridge fronts paid cloud backends, so we
    #: follow the base-class convention for paid providers: force an
    #: explicit model choice (``_resolve_model`` raises a ValueError with
    #: fix options) instead of silently falling back to a flagship.
    default_model = None

    #: No native-tools through the bridge — LiteLLM normalises to the
    #: OpenAI wire shape which has no slot for WEB_SEARCH / CODE_EXECUTION.
    #: Callers that need those should use a native provider.
    supported_native_tools = frozenset()

    # LiteLLM normalises to the OpenAI wire shape, so streaming + structured
    # output via JSON schema work for any backend that accepts them.
    # Thinking is highly model-dependent (only some Anthropic / Gemini
    # routes pipe ``reasoning_content`` through); we declare ``False`` so
    # capability-aware callers know not to assume the field exists.
    supports_thinking: bool = False

    # LiteLLM forwards images / audio through the OpenAI wire shape;
    # whether the eventual backend honours them depends on which
    # provider the model id resolves to.  We take the optimistic stance
    # and let the backend reject malformed input — the framework can't
    # know LiteLLM's evolving capability matrix without the litellm
    # package itself imported, which is why the optional install gates
    # the whole provider.

    @classmethod
    def supports_vision(cls, model: str | None = None) -> bool:
        return True

    @classmethod
    def supports_audio(cls, model: str | None = None) -> bool:
        return True

    def _init_client(self, **kwargs: Any) -> None:
        try:
            import litellm
        except ImportError as exc:
            raise ImportError(
                "LiteLLMProvider requires the litellm package. Install with: pip install 'lazybridge[litellm]'"
            ) from exc
        self._litellm = litellm
        # Forward extra kwargs to litellm module-level config (e.g.
        # drop_params, set_verbose, telemetry).
        #
        # NOTE: this configuration is GLOBAL to the litellm module —
        # litellm exposes these as module attributes, so the values are
        # shared by every LiteLLMProvider instance (and any other code
        # importing litellm) in the process.  Unknown keys are NOT set;
        # they raise a UserWarning so typos (e.g. ``drop_parms=True``)
        # don't get silently discarded.
        for key, value in kwargs.items():
            if hasattr(litellm, key):
                setattr(litellm, key, value)
            else:
                warnings.warn(
                    f"LiteLLMProvider: unknown litellm config kwarg {key!r} "
                    f"ignored — the litellm module has no such attribute. "
                    f"Check for typos against litellm's module-level settings "
                    f"(e.g. drop_params, set_verbose, telemetry).",
                    UserWarning,
                    stacklevel=3,
                )

    # ------------------------------------------------------------------
    # Request → LiteLLM params
    # ------------------------------------------------------------------

    def _build_params(self, request: CompletionRequest) -> dict[str, Any]:
        """Translate a CompletionRequest into litellm.completion kwargs."""
        if request.native_tools:
            # Warn-and-drop by default; raises UnsupportedNativeToolError
            # when the provider was built with strict_native_tools=True.
            self._check_native_tools(request.native_tools)

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
                params["tool_choice"] = _tool_choice_to_openai(request.tool_choice)
        if request.structured_output:
            # JSON mode — litellm translates response_format for the
            # backends that support it.  Validation against the schema
            # happens client-side in complete/stream/acomplete/astream.
            params["response_format"] = {"type": "json_object"}
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

    @staticmethod
    def _usage_from_raw(raw_usage: Any) -> UsageStats:
        """Build UsageStats from an OpenAI-shape usage object.

        Also picks up the optional detail blocks when the backend reports
        them: ``completion_tokens_details.reasoning_tokens`` (thinking) and
        ``prompt_tokens_details.cached_tokens`` (prompt-cache hits).
        """
        usage = UsageStats(
            input_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
        )
        completion_details = getattr(raw_usage, "completion_tokens_details", None)
        if completion_details is not None:
            usage.thinking_tokens = getattr(completion_details, "reasoning_tokens", 0) or 0
        prompt_details = getattr(raw_usage, "prompt_tokens_details", None)
        if prompt_details is not None:
            usage.cached_input_tokens = getattr(prompt_details, "cached_tokens", 0) or 0
        return usage

    def _extract_usage(self, resp: Any) -> UsageStats:
        raw_usage = getattr(resp, "usage", None)
        if raw_usage is None:
            return UsageStats()
        usage = self._usage_from_raw(raw_usage)
        # LiteLLM stuffs the computed cost under _hidden_params when it can
        # price the model; fall back to _compute_cost override otherwise.
        hidden = getattr(resp, "_hidden_params", None) or {}
        cost = hidden.get("response_cost") if isinstance(hidden, dict) else None
        if cost is not None:
            try:
                usage.cost_usd = float(cost)
            except (TypeError, ValueError):
                usage.cost_usd = None
        if usage.cost_usd is None:
            usage.cost_usd = self._compute_cost(
                getattr(resp, "model", None) or "",
                usage.input_tokens,
                usage.output_tokens,
                usage.cached_input_tokens,
            )
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
            # Usage-only trailer chunk: with stream_options.include_usage
            # the OpenAI spec sends a final chunk with choices=[] carrying
            # the usage block.  Read it even though there is no choice.
            u = getattr(chunk, "usage", None)
            if u:
                return "", None, self._usage_from_raw(u)
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
            usage = self._usage_from_raw(u)
        return delta_text, finish_reason, usage

    def _finalise_stream(
        self,
        request: CompletionRequest,
        partial_tool_calls: dict[int, dict[str, Any]],
        finish_reason: str | None,
        usage: UsageStats | None,
        text_accum: str,
    ) -> StreamChunk:
        """Build the final StreamChunk from accumulated state.

        Always produced — even when the stream died without ever sending a
        finish_reason, in which case ``stop_reason`` is ``"incomplete"`` so
        the is_final=True contract from base.py holds.
        """
        tool_calls = [
            ToolCall(
                id=slot["id"] or f"tc_{idx}",
                name=slot["name"],
                arguments=_safe_json_loads(slot["args"]),
            )
            for idx, slot in sorted(partial_tool_calls.items())
            if slot["name"]  # guard against malformed partials
        ]
        final = StreamChunk(
            delta="",
            tool_calls=tool_calls,
            stop_reason=finish_reason or "incomplete",
            usage=usage,
            is_final=True,
        )
        # Tool-call turns have empty/irrelevant text by design — validating
        # them would record a spurious validation_error.
        if request.structured_output and not tool_calls:
            from lazybridge.core.structured import apply_structured_validation

            apply_structured_validation(final, text_accum, request.structured_output.schema)
        return final

    # ------------------------------------------------------------------
    # Abstract-method implementations
    # ------------------------------------------------------------------

    def _apply_structured_validation(self, request: CompletionRequest, resp: CompletionResponse) -> CompletionResponse:
        """Validate response content against request.structured_output, if set.

        Tool-call turns are skipped — their content is empty by design and
        validating it would record a spurious validation_error.
        """
        if request.structured_output and not resp.tool_calls:
            from lazybridge.core.structured import apply_structured_validation

            apply_structured_validation(resp, resp.content, request.structured_output.schema)
        return resp

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        params = self._build_params(request)
        params["stream"] = False
        resp = self._litellm.completion(**params)
        return self._apply_structured_validation(request, self._convert_response(resp))

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        params = self._build_params(request)
        params["stream"] = True
        # OpenAI-spec opt-in: ask for the usage trailer chunk (choices=[]).
        params["stream_options"] = {"include_usage": True}
        iterator = self._litellm.completion(**params)
        partial: dict[int, dict[str, Any]] = {}
        last_usage: UsageStats | None = None
        last_finish: str | None = None
        text_accum = ""
        for chunk in iterator:
            delta_text, finish_reason, usage = self._reduce_stream_chunk(chunk, partial)
            if delta_text:
                text_accum += delta_text
                yield StreamChunk(delta=delta_text)
            if finish_reason:
                last_finish = finish_reason
            if usage is not None:
                last_usage = usage
        yield self._finalise_stream(request, partial, last_finish, last_usage, text_accum)

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        params = self._build_params(request)
        params["stream"] = False
        resp = await self._litellm.acompletion(**params)
        return self._apply_structured_validation(request, self._convert_response(resp))

    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        params = self._build_params(request)
        params["stream"] = True
        # OpenAI-spec opt-in: ask for the usage trailer chunk (choices=[]).
        params["stream_options"] = {"include_usage": True}
        async_iterator = await self._litellm.acompletion(**params)
        partial: dict[int, dict[str, Any]] = {}
        last_usage: UsageStats | None = None
        last_finish: str | None = None
        text_accum = ""
        async for chunk in async_iterator:
            delta_text, finish_reason, usage = self._reduce_stream_chunk(chunk, partial)
            if delta_text:
                text_accum += delta_text
                yield StreamChunk(delta=delta_text)
            if finish_reason:
                last_finish = finish_reason
            if usage is not None:
                last_usage = usage
        yield self._finalise_stream(request, partial, last_finish, last_usage, text_accum)
