"""LazyAgent — the single public entry point for LLM interaction.

Replaces both LazyLayer and LazySession from the old codebase.
The only difference between a stateless and a stateful agent is the ``session`` parameter.

Quick start::

    from lazybridge import LazyAgent

    # Stateless (zero boilerplate)
    ai = LazyAgent("anthropic")
    resp = ai.chat("hello")
    print(resp.content)

    # With tools
    from lazybridge import LazyTool

    def add(a: int, b: int) -> int:
        return a + b

    tool = LazyTool.from_function(add)
    resp = ai.loop("what is 3 + 4?", tools=[tool])

    # With session (shared state, tracking, graph)
    from lazybridge import LazySession

    sess = LazySession()
    researcher = LazyAgent("anthropic", name="researcher", session=sess)
    writer     = LazyAgent("openai",    name="writer",     session=sess)

    researcher.loop("find top AI news", tools=[web_search])
    writer.chat("write a summary", context=LazyContext.from_agent(researcher))

    # Expose as a tool for another agent
    research_tool = researcher.as_tool("researcher", "researches a topic")
    orchestrator.loop("plan and execute", tools=[research_tool])
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
import warnings
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, Literal, overload

from lazybridge.core.executor import Executor
from lazybridge.core.types import (
    CompletionRequest,
    CompletionResponse,
    Message,
    NativeTool,
    Role,
    SkillsConfig,
    StreamChunk,
    StructuredOutputConfig,
    TextContent,
    ThinkingConfig,
    ThinkingContent,
    ToolCall,
    ToolDefinition,
    ToolResultContent,
    ToolUseContent,
    Verifier,
)
from lazybridge.guardrails import Guard, GuardError
from lazybridge.lazy_context import LazyContext
from lazybridge.lazy_session import Event, EventLog, LazySession, TrackLevel
from lazybridge.lazy_tool import LazyTool, NormalizedToolSet
from lazybridge.memory import Memory

_logger = logging.getLogger(__name__)

# Sentinel used to distinguish "caller did not pass memory=" from "caller
# explicitly passed memory=None" (which means bypass agent-level memory).
_MEMORY_UNSET = object()

# ---------------------------------------------------------------------------
# Message normalisation helpers
# ---------------------------------------------------------------------------


def _normalise_messages(messages: str | list) -> list[Message]:
    if isinstance(messages, str):
        return [Message(role=Role.USER, content=messages)]
    result = []
    for m in messages:
        if isinstance(m, Message):
            result.append(m)
        elif isinstance(m, dict):
            if "role" not in m:
                warnings.warn(
                    "Message dict is missing the 'role' key; defaulting to 'user'. "
                    "Pass role='user'|'assistant'|'system' explicitly.",
                    UserWarning,
                    stacklevel=3,
                )
            role = Role(m.get("role", "user"))
            result.append(Message(role=role, content=m.get("content", "")))
        else:
            raise TypeError(f"Expected Message or dict, got {type(m)}")
    return result


def _messages_to_str(messages: str | list) -> str:
    """Extract a plain-text question string from messages for verify prompts."""
    if isinstance(messages, str):
        return messages
    for m in messages:
        if isinstance(m, Message) and m.role == Role.USER:
            return m.content if isinstance(m.content, str) else str(m.content)
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m.get("content", ""))
    return str(messages)


def _serialise_tool_result(result: Any) -> str:
    """Serialise a tool return value to a string for the model.

    Pydantic models → model_dump_json(); dicts → json.dumps(); everything else → str().
    """
    if hasattr(result, "model_dump_json"):  # Pydantic model
        return result.model_dump_json()
    if isinstance(result, dict):
        import json

        try:
            return json.dumps(result)
        except (TypeError, ValueError):
            pass
    return str(result)


def _tool_result_message(call: ToolCall, result: Any, *, is_error: bool = False) -> Message:
    return Message(
        role=Role.TOOL,
        content=[
            ToolResultContent(
                tool_use_id=call.id,
                content=_serialise_tool_result(result),
                tool_name=call.name,
                is_error=is_error,
            )
        ],
    )


async def _call_event_async(callback: Callable, name: str, payload: Any) -> None:
    """Call an on_event callback, awaiting it if it is a coroutine."""
    result = callback(name, payload)
    if inspect.iscoroutine(result):
        await result


# ---------------------------------------------------------------------------
# LazyAgent
# ---------------------------------------------------------------------------


class LazyAgent:
    """Unified LLM agent with optional session, context injection, and tool loops.

    Parameters
    ----------
    provider:
        Provider name (``"anthropic"``, ``"openai"``, ``"google"``, ``"deepseek"``)
        or a ``BaseProvider`` instance.
    name:
        Human-readable name. Used in graph schema, logging, and as tool name when
        this agent is exposed as a tool.
    description:
        Description used when this agent is exposed as a tool.
    model:
        Override the provider's default model.
    system:
        System prompt. Combined with ``context`` if both are given.
    context:
        A ``LazyContext`` (or callable ``() → str``) injected into the system
        prompt at execution time.
    tools:
        Pre-bound tools used when this agent is invoked as a delegated tool.
    session:
        A ``LazySession`` for shared store, tracking, and graph registration.
    max_retries:
        Retry on transient API errors (429/5xx). Default: 0.
    api_key:
        API key override. Falls back to the provider's standard env var.
    **kwargs:
        Forwarded to the provider constructor.

    State after execution
    ---------------------
    ``_last_output : str | None``
        Plain text of the last response. Read by ``LazyContext.from_agent()``
        for agent-to-agent context injection — the stable, public access path.
    ``_last_response : CompletionResponse | None``
        Full response object of the last call. Not yet a stable public API;
        use ``agent._last_response.parsed`` to access a typed Pydantic object
        when the agent was called with ``output_schema``.
        ``agent._last_response.usage`` gives token counts.
    """

    def __init__(
        self,
        provider: str | Any,
        *,
        name: str | None = None,
        description: str | None = None,
        model: str | None = None,
        system: str | None = None,
        context: LazyContext | Callable[[], str] | None = None,
        tools: list[LazyTool | ToolDefinition | dict] | None = None,
        native_tools: list[NativeTool | str] | None = None,
        output_schema: type | dict | None = None,
        session: LazySession | None = None,
        memory: Memory | None = None,
        max_retries: int = 0,
        api_key: str | None = None,
        verbose: bool = False,
        verify: Any | None = None,      # Option C: judge active on every call
        max_verify: int = 3,
        **kwargs: Any,
    ) -> None:
        self._executor = Executor(
            provider,
            api_key=api_key,
            model=model,
            max_retries=max_retries,
            **kwargs,
        )
        self.id = str(uuid.uuid4())
        self.name = name or self.id[:8]
        self.description = description
        self.system = system
        self.context = context
        self.tools: list[LazyTool | ToolDefinition | dict] = tools or []
        self.output_schema: type | dict | None = output_schema
        self.native_tools: list[NativeTool] = [NativeTool(t) if isinstance(t, str) else t for t in (native_tools or [])]
        self.session = session
        self.memory: Memory | None = memory
        self.verify: Any | None = verify          # Option C: agent-level judge
        self.max_verify: int = max_verify

        # Stores the last text output; read by LazyContext.from_agent() when
        # another agent wants to use this agent's result as context.
        self._last_output: str | None = None
        # Full response of the last call — includes .parsed, .usage, .grounding_sources.
        # Not a stable public API yet; access via agent._last_response.parsed for typed output.
        self._last_response: CompletionResponse | None = None

        # Tracking — scoped to this agent.
        # If a session is provided, use its EventLog.  Otherwise, when verbose=True,
        # create a standalone in-memory EventLog that only prints to console.
        if session:
            self._log = session.events.agent_log(self.id, self.name)
            if verbose:
                session.events._console = True
        elif verbose:
            _solo = EventLog(self.id, level=TrackLevel.BASIC, console=True)
            self._log = _solo.agent_log(self.id, self.name)
        else:
            self._log = None  # type: ignore[assignment]

        # Register with session graph so the pipeline topology is recorded
        if session:
            session._register_agent(self)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _provider_name(self) -> str:
        return type(self._executor.provider).__name__.replace("Provider", "").lower()

    @property
    def _model_name(self) -> str:
        return self._executor.model

    def _build_system(self, extra_system: str | None = None) -> str | None:
        """Combine agent-level system prompt with an optional call-level addition.

        Order: agent.system → extra_system (call-level override).
        Context is merged separately in _build_effective_system().
        """
        parts = []
        if self.system:
            parts.append(self.system)
        if extra_system:
            parts.append(extra_system)
        return "\n\n".join(parts) if parts else None

    def _build_effective_system(
        self,
        extra_system: str | None,
        call_context: LazyContext | Callable[[], str] | None,
        tool_set: NormalizedToolSet,
    ) -> str | None:
        """Merge system prompt, call-level context, and tool guidance into one string.

        Order: system → call-level context (overrides agent-level) → tool guidance.
        Call-level context overrides agent-level so per-call injection is always visible.
        """
        # Call-level context takes precedence over the agent's default context
        effective_ctx = call_context or self.context
        ctx_text = (effective_ctx() if callable(effective_ctx) else effective_ctx.build()) if effective_ctx else None

        parts = []
        base = self._build_system(extra_system)
        if base:
            parts.append(base)
        if ctx_text:
            parts.append(ctx_text)
        guidance = self._render_guidance(tool_set)
        if guidance:
            parts.append(guidance)
        return "\n\n".join(parts) if parts else None

    def _build_tool_set(self, tools: list | None) -> NormalizedToolSet:
        """Merge call-level tools with agent-level tools and normalise.

        tools=None  → use agent-level tools only (default).
        tools=[t]   → merge call-level with agent-level tools.
        tools=[]    → explicitly no tools; overrides agent-level tools.
        """
        if tools is not None and len(tools) == 0:
            return NormalizedToolSet([], [], {})
        all_tools = list(tools or []) + list(self.tools)
        return NormalizedToolSet.from_list(all_tools) if all_tools else NormalizedToolSet([], [], {})

    def _merge_native_tools(self, call_level: list[NativeTool | str] | None) -> list[NativeTool]:
        """Return agent-level native tools plus any call-level extras (deduplicated).

        call_level=None  → use agent-level native tools only.
        call_level=[]    → explicitly no native tools; overrides agent-level.
        call_level=[t]   → merge call-level with agent-level (deduplicated).
        """
        if call_level is not None and len(call_level) == 0:
            return []
        call = [NativeTool(t) if isinstance(t, str) else t for t in (call_level or [])]
        seen: set[str] = set()
        merged: list[NativeTool] = []
        for t in self.native_tools + call:
            key = t.value if hasattr(t, "value") else str(t)
            if key not in seen:
                seen.add(key)
                merged.append(t)
        return merged

    def _build_request(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        native_tools: list[NativeTool] | None = None,
        output_schema: type | dict | None = None,
        thinking: bool | ThinkingConfig = False,
        skills: list[str] | None = None,
        stream: bool = False,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tool_choice: str | None = None,
        **extra: Any,
    ) -> CompletionRequest:
        thinking_cfg: ThinkingConfig | None = None
        if thinking is True:
            thinking_cfg = ThinkingConfig()
        elif isinstance(thinking, ThinkingConfig):
            thinking_cfg = thinking

        structured: StructuredOutputConfig | None = None
        if output_schema is not None:
            structured = StructuredOutputConfig(schema=output_schema)

        return CompletionRequest(
            messages=messages,
            system=system,
            tools=tools or [],
            native_tools=native_tools or [],
            structured_output=structured,
            thinking=thinking_cfg,
            skills=SkillsConfig(skills=skills) if skills else None,
            stream=stream,
            model=model,
            max_tokens=max_tokens or self._executor.provider.get_default_max_tokens(model),
            temperature=temperature,
            tool_choice=tool_choice,
            extra=extra,
        )

    def _track(self, event_type: str, **data: Any) -> None:
        if self._log:
            try:
                self._log.log(event_type, **data)
            except Exception as exc:
                _logger.debug("Event tracking failed: %s", exc)

    # ------------------------------------------------------------------
    # Streaming helpers — accumulate delta text so _last_output is set
    # even when the caller iterates the stream directly.
    # ------------------------------------------------------------------

    def _stream_and_track(self, gen: Iterator[StreamChunk]) -> Iterator[StreamChunk]:
        # Reset and accumulate inside the loop so `_last_output` always
        # reflects the bytes the caller actually received — even if the
        # caller breaks out of the iterator early.
        self._last_output = ""
        parts: list[str] = []
        for chunk in gen:
            if chunk.delta is not None:
                parts.append(chunk.delta)
                self._last_output = "".join(parts)
            yield chunk

    async def _astream_and_track(self, gen: AsyncIterator[StreamChunk]) -> AsyncIterator[StreamChunk]:
        self._last_output = ""
        parts: list[str] = []
        async for chunk in gen:
            if chunk.delta is not None:
                parts.append(chunk.delta)
                self._last_output = "".join(parts)
            yield chunk

    # ------------------------------------------------------------------
    # Shared helpers for chat/achat and loop/aloop (sync/async dedup)
    # ------------------------------------------------------------------

    def _validate_memory(self, messages: str | list, memory: Memory | None, stream: bool) -> None:
        """Shared validation for memory parameter. Raises on invalid combos."""
        if memory is None:
            return
        if not isinstance(messages, str):
            raise TypeError(
                "memory= requires messages to be a str. For list messages manage history manually via chat(list)."
            )
        if stream:
            raise TypeError(
                "stream=True is not compatible with memory=. "
                "Consume the stream manually and call memory._record() yourself."
            )

    def _prepare_chat_request(
        self,
        messages: str | list,
        *,
        system: str | None,
        tools: list | None,
        native_tools: list[NativeTool | str] | None,
        output_schema: type | dict | None,
        thinking: bool | ThinkingConfig,
        skills: list[str] | None,
        stream: bool,
        model: str | None,
        max_tokens: int | None,
        temperature: float | None,
        tool_choice: str | None,
        context: LazyContext | Callable[[], str] | None,
        **kwargs: Any,
    ) -> tuple[list[Message], CompletionRequest]:
        """Build messages and request — shared between chat() and achat()."""
        msgs = _normalise_messages(messages)
        tool_set = self._build_tool_set(tools)
        effective_system = self._build_effective_system(system, context, tool_set)
        effective_schema = output_schema if output_schema is not None else self.output_schema

        request = self._build_request(
            msgs,
            system=effective_system,
            tools=tool_set.definitions,
            native_tools=self._merge_native_tools(native_tools),
            output_schema=effective_schema,
            thinking=thinking,
            skills=skills,
            stream=stream,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            tool_choice=tool_choice,
            **kwargs,
        )
        return msgs, request

    def _record_response(self, resp: CompletionResponse) -> None:
        """Store response and track model_response event — shared between chat/achat."""
        self._last_output = resp.content
        self._last_response = resp
        self._track(
            Event.MODEL_RESPONSE,
            model=resp.model or self._model_name,
            stop_reason=resp.stop_reason,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            cost_usd=resp.usage.cost_usd,
            content=resp.content,
        )

    @staticmethod
    def _build_assistant_turn(resp: CompletionResponse) -> Any:
        """Build assistant message content with thinking + text + tool-use blocks.

        Shared between loop() and aloop() to avoid duplicating the
        thought_signature forwarding logic (required for Gemini).
        """
        _tc_blocks = [
            ToolUseContent(id=tc.id, name=tc.name, input=tc.arguments, thought_signature=tc.thought_signature)
            for tc in resp.tool_calls
        ]
        _thinking_blocks: list[Any] = [ThinkingContent(thinking=resp.thinking)] if resp.thinking else []
        if resp.tool_calls:
            _text_blocks: list[Any] = [TextContent(text=resp.content)] if resp.content else []
            return _thinking_blocks + _text_blocks + _tc_blocks
        return resp.content or ""

    def _finalize_loop(
        self,
        resp: CompletionResponse,
        verify_log: list[str],
        attempts: int,
        verify: Any,
        method: str,
        step: int,
    ) -> CompletionResponse:
        """Shared loop finalization — warn if verify exhausted, store result."""
        if verify is not None and len(verify_log) == attempts:
            import warnings

            warnings.warn(
                f"{method}() verify exhausted after {attempts} attempt(s) without approval. "
                "Returning last result unchanged. "
                "Increase max_verify= or review your verify function.",
                UserWarning,
                stacklevel=3,
            )

        resp.verify_log = verify_log  # type: ignore[union-attr]
        self._last_output = resp.content
        self._last_response = resp
        self._track(
            Event.AGENT_FINISH,
            method=method,
            stop_reason=resp.stop_reason,
            n_steps=step + 1,
        )
        return resp

    # ------------------------------------------------------------------
    # Loop action protocol — unified loop logic for sync/async
    # ------------------------------------------------------------------

    _CALL_MODEL = "call_model"
    _EXEC_TOOL = "exec_tool"
    _EXEC_TOOLS_BATCH = "exec_tools_batch"
    _EMIT_EVENT = "emit_event"
    _VERIFY = "verify"

    def _loop_logic(
        self,
        messages: str | list,
        *,
        tools: list | None,
        native_tools: list[NativeTool | str] | None,
        max_steps: int,
        tool_runner: Callable | None,
        on_event: Callable | None,
        verify: Any,
        max_verify: int,
        method: str,
        chat_kwargs: dict,
        force_final_after_tools: bool = False,
    ):
        """Generator that yields (action, *args) tuples at sync/async divergence points.

        The caller (loop or aloop) sends back the result of each action.
        This keeps all decision logic in one place.
        """
        _orig_q = _messages_to_str(messages)
        _current_messages: str | list = messages
        _attempts = max(1, max_verify) if verify is not None else 1
        resp: CompletionResponse | None = None
        _verify_log: list[str] = []
        step = 0

        # If tool_choice is a forced value ("required" or a specific tool name),
        # apply it only on the first model call.  After a tool executes, revert
        # to "auto" so the model can decide to produce the final answer.
        _first_tool_choice = chat_kwargs.get("tool_choice")
        _forced = _first_tool_choice not in (None, "auto", "none", "parallel")
        _auto_kwargs: dict = {**chat_kwargs, "tool_choice": "auto"} if _forced else chat_kwargs

        self._track(Event.AGENT_START, method=method, task=_orig_q[:200])

        for _attempt in range(_attempts):
            tool_set = self._build_tool_set(tools)
            convo = _normalise_messages(_current_messages)
            _tool_was_called = False

            for step in range(max_steps):
                kw = chat_kwargs if (step == 0 or not _tool_was_called) else _auto_kwargs
                # When force_final_after_tools is set (e.g. from json()), strip tools
                # from the model call after the first tool round so the model is
                # forced to produce a final answer rather than calling tools again.
                step_tools = [] if (force_final_after_tools and _tool_was_called) else tools
                step_native_tools = [] if (force_final_after_tools and _tool_was_called) else native_tools
                resp = yield (self._CALL_MODEL, convo, step_tools, step_native_tools, kw)

                if on_event:
                    yield (self._EMIT_EVENT, on_event, "step", {"step": step, "response": resp})

                if not resp.tool_calls:
                    break

                self._track(Event.LOOP_STEP, step=step, n_tool_calls=len(resp.tool_calls))
                convo.append(Message(role=Role.ASSISTANT, content=self._build_assistant_turn(resp)))

                _tool_was_called = True
                for tc in resp.tool_calls:
                    self._track(Event.TOOL_CALL, name=tc.name, arguments=tc.arguments)
                    if on_event:
                        yield (self._EMIT_EVENT, on_event, "tool_call", tc)

                _parallel = chat_kwargs.get("tool_choice") == "parallel"
                if _parallel and len(resp.tool_calls) > 1:
                    results = yield (
                        self._EXEC_TOOLS_BATCH,
                        resp.tool_calls,
                        tool_set.registry,
                        tool_runner,
                    )
                    for tc, (result, error) in zip(resp.tool_calls, results):
                        if error is not None:
                            self._track(Event.TOOL_ERROR, name=tc.name, error=str(error))
                            convo.append(_tool_result_message(tc, f"Error: {error}", is_error=True))
                        else:
                            self._track(Event.TOOL_RESULT, name=tc.name, result=str(result)[:500])
                            if on_event:
                                yield (self._EMIT_EVENT, on_event, "tool_result", {"call": tc, "result": result})
                            convo.append(_tool_result_message(tc, result))
                else:
                    for tc in resp.tool_calls:
                        try:
                            result = yield (self._EXEC_TOOL, tc, tool_set.registry, tool_runner)
                            self._track(Event.TOOL_RESULT, name=tc.name, result=str(result)[:500])
                            if on_event:
                                yield (self._EMIT_EVENT, on_event, "tool_result", {"call": tc, "result": result})
                            convo.append(_tool_result_message(tc, result))
                        except Exception as exc:
                            _logger.debug("Tool %r raised: %s", tc.name, exc, exc_info=True)
                            self._track(Event.TOOL_ERROR, name=tc.name, error=str(exc))
                            convo.append(_tool_result_message(tc, f"Error: {exc}", is_error=True))

            if resp is None:  # pragma: no cover
                raise AssertionError("loop produced no response — this is a bug, please report it")

            if on_event:
                yield (self._EMIT_EVENT, on_event, "done", resp)

            if verify is None:
                break

            _raw = yield (self._VERIFY, verify, _orig_q, resp.content)
            _verdict: str = str(_raw) if _raw is not None else ""
            if _verdict[:30].lower().startswith("approved"):
                break

            _verify_log.append(_verdict)
            if on_event:
                yield (self._EMIT_EVENT, on_event, "verify_rejected", {"attempt": _attempt + 1, "verdict": _verdict})
            _current_messages = f"{_orig_q}\n\nPrevious attempt rejected: {_verdict or '(no verdict)'}\nTry again."

        return self._finalize_loop(resp, _verify_log, _attempts, verify, method, step)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Guard helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run_input_guard(guard: Guard | None, messages: str | list) -> str | list:
        """Run input guard on the user message text. Returns (possibly modified) messages."""
        if guard is None:
            return messages
        text = messages if isinstance(messages, str) else _messages_to_str(messages)
        action = guard.check_input(text)
        if not action.allowed:
            raise GuardError(action)
        if action.modified_text is not None and isinstance(messages, str):
            return action.modified_text
        return messages

    @staticmethod
    def _run_output_guard(guard: Guard | None, resp: CompletionResponse) -> CompletionResponse:
        """Run output guard on the response content. Raises GuardError if blocked."""
        if guard is None or not resp.content:
            return resp
        action = guard.check_output(resp.content)
        if not action.allowed:
            raise GuardError(action)
        if action.modified_text is not None:
            resp.content = action.modified_text
        return resp

    @staticmethod
    async def _arun_input_guard(guard: Guard | None, messages: str | list) -> str | list:
        """Async input guard. Uses acheck_input if available, else check_input."""
        if guard is None:
            return messages
        text = messages if isinstance(messages, str) else _messages_to_str(messages)
        if hasattr(guard, "acheck_input"):
            action = await guard.acheck_input(text)
        else:
            action = guard.check_input(text)
        if not action.allowed:
            raise GuardError(action)
        if action.modified_text is not None and isinstance(messages, str):
            return action.modified_text
        return messages

    @staticmethod
    async def _arun_output_guard(guard: Guard | None, resp: CompletionResponse) -> CompletionResponse:
        """Async output guard. Uses acheck_output if available, else check_output."""
        if guard is None or not resp.content:
            return resp
        if hasattr(guard, "acheck_output"):
            action = await guard.acheck_output(resp.content)
        else:
            action = guard.check_output(resp.content)
        if not action.allowed:
            raise GuardError(action)
        if action.modified_text is not None:
            resp.content = action.modified_text
        return resp

    # ------------------------------------------------------------------
    # chat() — single turn
    # ------------------------------------------------------------------

    @overload
    def chat(self, messages: str | list, *, stream: Literal[False] = ..., **kwargs: Any) -> CompletionResponse: ...
    @overload
    def chat(self, messages: str | list, *, stream: Literal[True], **kwargs: Any) -> Iterator[StreamChunk]: ...

    def chat(
        self,
        messages: str | list,
        *,
        memory: Memory | None = _MEMORY_UNSET,  # type: ignore[assignment]
        system: str | None = None,
        tools: list | None = None,
        native_tools: list[NativeTool | str] | None = None,
        output_schema: type | dict | None = None,
        thinking: bool | ThinkingConfig = False,
        skills: list[str] | None = None,
        stream: bool = False,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tool_choice: str | None = None,
        context: LazyContext | Callable[[], str] | None = None,
        guard: Guard | None = None,
        verify: Verifier | Callable[..., Any] | None = None,
        max_verify: int = 3,
        **kwargs: Any,
    ) -> CompletionResponse | Iterator[StreamChunk]:
        """Send a single-turn chat request.

        Pass ``memory`` to accumulate conversation history automatically::

            mem = Memory()
            ai.chat("ciao", memory=mem)
            ai.chat("ricordi?", memory=mem)   # history included automatically

        ``tool_choice`` controls tool selection: ``"auto"`` (default), ``"none"``,
        ``"required"`` (force at least one tool call), or a specific tool name.

        ``context`` overrides the agent-level context for this call only.

        ``guard`` runs input validation before the LLM call and output validation
        after. Raises ``GuardError`` if the guard blocks content.
        """
        # Verify loop — if verify= is set, run the retry cycle here and call
        # self.chat(..., verify=None) each time to avoid infinite recursion.
        if verify is not None:
            if stream:
                raise TypeError("stream=True is incompatible with verify= in chat().")
            task = messages
            for _attempt in range(max_verify):
                resp = self.chat(
                    messages,
                    memory=memory,
                    system=system,
                    tools=tools,
                    native_tools=native_tools,
                    output_schema=output_schema,
                    thinking=thinking,
                    skills=skills,
                    stream=False,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    tool_choice=tool_choice,
                    context=context,
                    guard=guard,
                    verify=None,  # terminates recursion
                    **kwargs,
                )
                if not isinstance(resp, CompletionResponse):
                    return resp  # pragma: no cover
                verdict = (
                    verify.text(f"Question: {task}\nAnswer: {resp.content}")
                    if hasattr(verify, "text")
                    else verify(str(task), resp.content)
                )
                if verdict.strip().lower().startswith("approved"):
                    return resp
                messages = (
                    f"{task}\n\nPrevious attempt was rejected with this feedback: {verdict}\n"
                    "Please address the feedback and try again."
                )
            warnings.warn(
                f"chat() verify exhausted after {max_verify} attempt(s) without approval. "
                "Returning last result unchanged.",
                UserWarning,
                stacklevel=2,
            )
            return resp  # type: ignore[return-value]

        messages = self._run_input_guard(guard, messages)

        # Resolve effective memory:
        #   _MEMORY_UNSET (default) → fall back to agent-level self.memory
        #   None (explicit)         → no memory for this call (bypass agent-level)
        #   Memory instance         → use it directly
        if memory is _MEMORY_UNSET:
            effective_memory = self.memory
        else:
            effective_memory = memory  # type: ignore[assignment]
        if effective_memory is not None:
            self._validate_memory(messages, effective_memory, stream)
            full = effective_memory._build_input(messages)  # type: ignore[arg-type]
            resp = self.chat(
                full,
                memory=None,  # None = explicit bypass; _MEMORY_UNSET would re-apply agent memory
                system=system,
                tools=tools,
                native_tools=native_tools,
                output_schema=output_schema,
                thinking=thinking,
                skills=skills,
                stream=False,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                tool_choice=tool_choice,
                context=context,
                guard=guard,
                **kwargs,
            )
            if not isinstance(resp, CompletionResponse):  # pragma: no cover
                raise TypeError(f"Expected CompletionResponse, got {type(resp).__name__}")
            effective_memory._record(messages, resp.content)  # type: ignore[arg-type]
            return resp

        msgs, request = self._prepare_chat_request(
            messages,
            system=system,
            tools=tools,
            native_tools=native_tools,
            output_schema=output_schema,
            thinking=thinking,
            skills=skills,
            stream=stream,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            tool_choice=tool_choice,
            context=context,
            **kwargs,
        )

        self._track(Event.MODEL_REQUEST, model=request.model or self._model_name, n_messages=len(msgs))
        if stream:
            return self._stream_and_track(self._executor.stream(request))

        resp = self._executor.execute(request)
        # Run the output guard BEFORE recording so blocked content does
        # not leak into Event.MODEL_RESPONSE payloads or exporters.
        self._run_output_guard(guard, resp)
        self._record_response(resp)
        return resp

    @overload
    async def achat(
        self, messages: str | list, *, stream: Literal[False] = ..., **kwargs: Any
    ) -> CompletionResponse: ...
    @overload
    async def achat(
        self, messages: str | list, *, stream: Literal[True], **kwargs: Any
    ) -> AsyncIterator[StreamChunk]: ...

    async def achat(
        self,
        messages: str | list,
        *,
        memory: Memory | None = _MEMORY_UNSET,  # type: ignore[assignment]
        system: str | None = None,
        tools: list | None = None,
        native_tools: list[NativeTool | str] | None = None,
        output_schema: type | dict | None = None,
        thinking: bool | ThinkingConfig = False,
        skills: list[str] | None = None,
        stream: bool = False,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tool_choice: str | None = None,
        context: LazyContext | Callable[[], str] | None = None,
        guard: Guard | None = None,
        verify: Verifier | Callable[..., Any] | None = None,
        max_verify: int = 3,
        **kwargs: Any,
    ) -> CompletionResponse | AsyncIterator[StreamChunk]:
        """Async version of chat(). Accepts memory=, tool_choice=, guard=, verify=."""
        if verify is not None:
            if stream:
                raise TypeError("stream=True is incompatible with verify= in achat().")
            task = messages
            for _attempt in range(max_verify):
                resp = await self.achat(
                    messages,
                    memory=memory,
                    system=system,
                    tools=tools,
                    native_tools=native_tools,
                    output_schema=output_schema,
                    thinking=thinking,
                    skills=skills,
                    stream=False,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    tool_choice=tool_choice,
                    context=context,
                    guard=guard,
                    verify=None,
                    **kwargs,
                )
                if not isinstance(resp, CompletionResponse):
                    return resp  # pragma: no cover
                if hasattr(verify, "atext"):
                    verdict = await verify.atext(f"Question: {task}\nAnswer: {resp.content}")
                elif inspect.iscoroutinefunction(verify):
                    verdict = await verify(str(task), resp.content)
                else:
                    verdict = verify.text(f"Question: {task}\nAnswer: {resp.content}") if hasattr(verify, "text") else verify(str(task), resp.content)
                if verdict.strip().lower().startswith("approved"):
                    return resp
                messages = (
                    f"{task}\n\nPrevious attempt was rejected with this feedback: {verdict}\n"
                    "Please address the feedback and try again."
                )
            warnings.warn(
                f"achat() verify exhausted after {max_verify} attempt(s) without approval. "
                "Returning last result unchanged.",
                UserWarning,
                stacklevel=2,
            )
            return resp  # type: ignore[return-value]

        messages = await self._arun_input_guard(guard, messages)

        if memory is _MEMORY_UNSET:
            effective_memory = self.memory
        else:
            effective_memory = memory  # type: ignore[assignment]
        if effective_memory is not None:
            self._validate_memory(messages, effective_memory, stream)
            full = effective_memory._build_input(messages)  # type: ignore[arg-type]
            resp = await self.achat(
                full,
                memory=None,  # None = explicit bypass
                system=system,
                tools=tools,
                native_tools=native_tools,
                output_schema=output_schema,
                thinking=thinking,
                skills=skills,
                stream=False,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                tool_choice=tool_choice,
                context=context,
                guard=guard,
                **kwargs,
            )
            if not isinstance(resp, CompletionResponse):  # pragma: no cover
                raise TypeError(f"Expected CompletionResponse, got {type(resp).__name__}")
            effective_memory._record(messages, resp.content)  # type: ignore[arg-type]
            return resp

        msgs, request = self._prepare_chat_request(
            messages,
            system=system,
            tools=tools,
            native_tools=native_tools,
            output_schema=output_schema,
            thinking=thinking,
            skills=skills,
            stream=stream,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            tool_choice=tool_choice,
            context=context,
            **kwargs,
        )

        self._track(Event.MODEL_REQUEST, model=request.model or self._model_name, n_messages=len(msgs))
        if stream:
            return self._astream_and_track(self._executor.astream(request))

        resp = await self._executor.aexecute(request)
        # Output guard runs before _record_response — see sync path above.
        await self._arun_output_guard(guard, resp)
        self._record_response(resp)
        return resp

    # ------------------------------------------------------------------
    # loop() — agentic tool loop
    # ------------------------------------------------------------------

    def loop(
        self,
        messages: str | list,
        *,
        tools: list | None = None,
        native_tools: list[NativeTool | str] | None = None,
        max_steps: int = 8,
        tool_runner: Callable[[str, dict], Any] | None = None,
        on_event: Callable[[str, Any], None] | None = None,
        verify: Verifier | Callable[[str, str], str] | None = None,
        max_verify: int = 3,
        guard: Guard | None = None,
        tool_timeout: float | None = None,
        force_final_after_tools: bool = False,
        **chat_kwargs: Any,
    ) -> CompletionResponse:
        """Agentic loop: chat → execute tool calls → repeat until done.

        Terminates when the model produces no tool calls or ``max_steps``
        is reached.  Returns the final ``CompletionResponse``.

        Parameters
        ----------
        tools:
            LazyTool, ToolDefinition, or dict items.  Combined with
            agent-level ``self.tools``.
        tool_runner:
            Fallback callable ``(name, arguments) → Any`` for tools not in
            the registry (e.g. raw ToolDefinition without a LazyTool).
        on_event:
            Optional callback ``(event_name, payload)`` called on each step.
            Events: ``"step"``, ``"tool_call"``, ``"tool_result"``, ``"done"``,
            ``"verify_rejected"``.
        max_steps:
            Hard cap on loop iterations.
        verify:
            Optional judge.  A ``LazyAgent`` (calls ``.text()``) or a
            ``Callable[[question, answer], verdict]``.  Return a string
            starting with ``"approved"`` (case-insensitive) to accept the
            answer; anything else triggers a retry with the feedback.
        max_verify:
            Maximum verify attempts.  If exceeded the last worker result is
            returned unchanged — no exception is raised.

        ``tool_choice`` values (passed via **chat_kwargs):
            ``"auto"`` (default), ``"required"``, ``"none"``, ``"<tool_name>"``,
            ``"parallel"`` — execute multiple tool calls concurrently
            (async uses ``asyncio.gather()``).
        """
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        if chat_kwargs.get("stream"):
            raise TypeError("stream=True is not supported in loop(). Use chat() for streaming.")

        # Extract memory so it is applied once at the loop boundary (initial
        # message prepend + final result record) rather than on every internal
        # chat() step.  Per-call memory overrides agent-level memory.
        loop_memory: Memory | None = chat_kwargs.pop("memory", None)
        if loop_memory is None:
            loop_memory = self.memory
        # Explicitly bypass agent-level memory inside internal chat() calls so
        # chat() doesn't try to re-apply it to the already-built list messages.
        chat_kwargs["memory"] = None

        messages = self._run_input_guard(guard, messages)
        _mem_task = messages  # save original string for _record (before _build_input rebinds to list)
        if loop_memory is not None:
            self._validate_memory(messages, loop_memory, False)
            messages = loop_memory._build_input(messages)  # type: ignore[arg-type]

        gen = self._loop_logic(
            messages,
            tools=tools,
            native_tools=native_tools,
            max_steps=max_steps,
            tool_runner=tool_runner,
            on_event=on_event,
            verify=verify,
            max_verify=max_verify,
            method="loop",
            chat_kwargs=chat_kwargs,
            force_final_after_tools=force_final_after_tools,
        )
        action = next(gen)
        result: CompletionResponse | None = None
        while True:
            tag = action[0]
            if tag == self._CALL_MODEL:
                _, convo, t, nt, kw = action
                val = self.chat(convo, tools=t, native_tools=nt, **kw)
            elif tag == self._EXEC_TOOL:
                _, tc, registry, runner = action
                val = self._execute_tool(tc, registry, runner, tool_timeout=tool_timeout)
            elif tag == self._EXEC_TOOLS_BATCH:
                _, calls, registry, runner = action
                val = []
                for tc in calls:
                    try:
                        r = self._execute_tool(tc, registry, runner, tool_timeout=tool_timeout)
                        val.append((r, None))
                    except Exception as exc:
                        val.append((None, exc))
            elif tag == self._EMIT_EVENT:
                _, callback, name, payload = action
                callback(name, payload)
                val = None
            elif tag == self._VERIFY:
                _, v, q, ans = action
                val = v.text(f"Question: {q}\nAnswer: {ans}") if hasattr(v, "text") else v(q, ans)
            else:  # pragma: no cover
                val = None
            try:
                action = gen.send(val)
            except StopIteration as e:
                result = e.value
                break
        self._run_output_guard(guard, result)
        if loop_memory is not None:
            loop_memory._record(_mem_task, result.content)  # type: ignore[arg-type]
        return result

    async def aloop(
        self,
        messages: str | list,
        *,
        tools: list | None = None,
        native_tools: list[NativeTool | str] | None = None,
        max_steps: int = 8,
        tool_runner: Callable[[str, dict], Any] | None = None,
        on_event: Callable[[str, Any], None] | None = None,
        verify: Verifier | Callable[..., Any] | None = None,
        max_verify: int = 3,
        guard: Guard | None = None,
        tool_timeout: float | None = None,
        force_final_after_tools: bool = False,
        **chat_kwargs: Any,
    ) -> CompletionResponse:
        """Async version of loop().

        Parameters
        ----------
        verify:
            Optional judge.  A ``LazyAgent`` (calls ``.atext()``) or a
            callable ``(question, answer) → verdict`` (may be a coroutine).
            Return a string starting with ``"approved"`` to accept the answer.
        max_verify:
            Maximum verify attempts before returning the last result as-is.
        """
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        if chat_kwargs.get("stream"):
            raise TypeError("stream=True is not supported in aloop(). Use achat() for streaming.")

        # Same boundary-memory pattern as loop() — pop before passing to generator.
        loop_memory: Memory | None = chat_kwargs.pop("memory", None)
        if loop_memory is None:
            loop_memory = self.memory
        chat_kwargs["memory"] = None

        messages = await self._arun_input_guard(guard, messages)
        _mem_task = messages  # save original string for _record (before _build_input rebinds to list)
        if loop_memory is not None:
            self._validate_memory(messages, loop_memory, False)
            messages = loop_memory._build_input(messages)  # type: ignore[arg-type]

        gen = self._loop_logic(
            messages,
            tools=tools,
            native_tools=native_tools,
            max_steps=max_steps,
            tool_runner=tool_runner,
            on_event=on_event,
            verify=verify,
            max_verify=max_verify,
            method="aloop",
            chat_kwargs=chat_kwargs,
            force_final_after_tools=force_final_after_tools,
        )
        action = next(gen)
        result: CompletionResponse | None = None
        while True:
            tag = action[0]
            if tag == self._CALL_MODEL:
                _, convo, t, nt, kw = action
                val = await self.achat(convo, tools=t, native_tools=nt, **kw)
            elif tag == self._EXEC_TOOL:
                _, tc, registry, runner = action
                val = await self._aexecute_tool(tc, registry, runner, tool_timeout=tool_timeout)
            elif tag == self._EXEC_TOOLS_BATCH:
                _, calls, registry, runner = action

                async def _run_one(c, _reg=registry, _run=runner, _tt=tool_timeout):
                    try:
                        r = await self._aexecute_tool(c, _reg, _run, tool_timeout=_tt)
                        return (r, None)
                    except Exception as exc:
                        return (None, exc)

                val = await asyncio.gather(*[_run_one(c) for c in calls])
                val = list(val)
            elif tag == self._EMIT_EVENT:
                _, callback, name, payload = action
                await _call_event_async(callback, name, payload)
                val = None
            elif tag == self._VERIFY:
                _, v, q, ans = action
                if hasattr(v, "atext"):
                    val = await v.atext(f"Question: {q}\nAnswer: {ans}")
                elif inspect.iscoroutinefunction(v):
                    val = await v(q, ans)
                else:
                    val = v(q, ans)
            else:  # pragma: no cover
                val = None
            try:
                action = gen.send(val)
            except StopIteration as e:
                result = e.value
                break
        await self._arun_output_guard(guard, result)
        if loop_memory is not None:
            loop_memory._record(_mem_task, result.content)  # type: ignore[arg-type]
        return result

    # ------------------------------------------------------------------
    # Tool execution helpers
    # ------------------------------------------------------------------

    def _execute_tool(
        self,
        call: ToolCall,
        registry: dict[str, LazyTool],
        runner: Callable | None,
        *,
        tool_timeout: float | None = None,
    ) -> Any:
        """Run a tool call, optionally bounded by ``tool_timeout``.

        When ``tool_timeout`` is set, the call runs in a worker thread and
        times out cleanly with :class:`TimeoutError` rather than hanging
        the loop indefinitely (ChatGPT audit F5).
        """

        def _call() -> Any:
            # Registry lookup first (LazyTool with a known callable)
            if call.name in registry:
                return registry[call.name].run(call.arguments, parent=self)
            # Fallback to user-provided runner (e.g. native tool handling)
            if runner:
                return runner(call.name, call.arguments)
            raise RuntimeError(
                f"No handler for tool '{call.name}'. Add a LazyTool with that name or provide tool_runner."
            )

        if tool_timeout is None or tool_timeout <= 0:
            return _call()

        import concurrent.futures as _futures

        # Can't use `with ThreadPoolExecutor(...)` — __exit__ waits for
        # the worker to finish, negating the timeout. Create the pool
        # manually and call shutdown(wait=False) so the runaway thread
        # doesn't block the caller. The worker thread may leak (Python
        # has no thread-kill primitive); this is the documented trade-off.
        pool = _futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(_call)
        try:
            return future.result(timeout=tool_timeout)
        except _futures.TimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"tool {call.name!r} exceeded tool_timeout={tool_timeout}s") from exc
        finally:
            pool.shutdown(wait=False)

    async def _aexecute_tool(
        self,
        call: ToolCall,
        registry: dict[str, LazyTool],
        runner: Callable | None,
        *,
        tool_timeout: float | None = None,
    ) -> Any:
        """Async counterpart of :meth:`_execute_tool` with ``tool_timeout`` support."""

        async def _acall() -> Any:
            # Registry lookup first — LazyTool.arun() is always a coroutine
            if call.name in registry:
                return await registry[call.name].arun(call.arguments, parent=self)
            # Fallback: runner may be sync or async
            if runner:
                result = runner(call.name, call.arguments)
                if inspect.isawaitable(result):
                    return await result
                return result
            raise RuntimeError(
                f"No handler for tool '{call.name}'. Add a LazyTool with that name or provide tool_runner."
            )

        if tool_timeout is None or tool_timeout <= 0:
            return await _acall()
        try:
            return await asyncio.wait_for(_acall(), timeout=tool_timeout)
        except TimeoutError as exc:
            raise TimeoutError(f"tool {call.name!r} exceeded tool_timeout={tool_timeout}s") from exc

    # ------------------------------------------------------------------
    # as_tool() — expose this agent as a LazyTool
    # ------------------------------------------------------------------

    def as_tool(
        self,
        name: str | None = None,
        description: str | None = None,
        *,
        guidance: str | None = None,
        output_schema: type | dict | None = None,
        native_tools: list[NativeTool | str] | None = None,
        system_prompt: str | None = None,
        tool_choice: str | None = None,
        verify: Any | None = None,      # Option B: judge attached to this tool exposure
        max_verify: int = 3,
        strict: bool = False,
    ) -> LazyTool:
        """Wrap this agent as a LazyTool for use in another agent's loop.

        The tool's schema is always ``{"task": str}``.
        The task string is forwarded to this agent's loop() or chat().

        ``tool_choice`` controls tool selection in the inner loop:
        ``"required"`` forces tool use, ``"<name>"`` forces a specific tool.
        ``verify`` / ``max_verify`` (Option B): attach a judge to *this tool exposure*.
        The judge runs after every invocation, regardless of context.
        Takes precedence over the agent-level ``verify`` (Option C).
        """
        return LazyTool.from_agent(
            self,
            name=name or self.name,
            description=description or self.description,
            guidance=guidance,
            output_schema=output_schema,
            native_tools=native_tools,
            system_prompt=system_prompt,
            tool_choice=tool_choice,
            verify=verify,
            max_verify=max_verify,
            strict=strict,
        )

    # ------------------------------------------------------------------
    # result — canonical accessor for the last call's output
    # ------------------------------------------------------------------

    @property
    def result(self) -> Any:
        """Canonical result of the last call.

        Returns the typed Pydantic object when the last call had an active
        ``output_schema`` (agent-level or call-level) and parsing succeeded.
        Returns the text content string otherwise.  Returns ``None`` if the
        agent has never been called.

        This is the recommended accessor for pipeline code that needs the
        "final value" of an agent without caring about the internal
        representation::

            report_writer = LazyAgent("openai", output_schema=InvestmentReport)
            report_writer.loop("Analyse this data", tools=[...])

            report = report_writer.result   # InvestmentReport instance
            print(report.title)

        For text-only output::

            researcher = LazyAgent("anthropic")
            researcher.chat("Find AI news this week")
            print(researcher.result)        # plain string

        Implementation note
        -------------------
        Internally LazyBridge keeps two complementary fields:

        * ``_last_output : str | None``
          Always a plain string.  Read by ``LazyContext.from_agent()`` for
          agent-to-agent context injection.  Stays text-first deliberately —
          injecting structured objects into a system prompt would require
          explicit serialisation at the call site.

        * ``_last_response : CompletionResponse | None``
          The full provider response, including ``.parsed`` (Pydantic object),
          ``.usage``, ``.tool_calls``, and ``.grounding_sources``.
          Not yet a stable public API; prefer ``agent.result`` for the value
          and ``agent._last_response`` for advanced introspection.

        ``result`` unifies both: typed when available, text otherwise.
        """
        if self._last_response is None:
            return self._last_output
        if self._last_response.parsed is not None:
            return self._last_response.parsed
        return self._last_response.content

    # ------------------------------------------------------------------
    # Convenience text/json shortcuts
    # ------------------------------------------------------------------

    def text(
        self,
        messages: str | list,
        *,
        verify: Verifier | Callable[..., Any] | None = None,
        max_verify: int = 3,
        **kwargs,
    ) -> str:
        if kwargs.get("stream"):
            raise TypeError("stream=True is not supported in text(). Use chat(stream=True) instead.")

        task = messages
        for _attempt in range(max_verify if verify is not None else 1):
            resp = self.chat(messages, **kwargs)
            if not isinstance(resp, CompletionResponse):
                raise TypeError(f"Expected CompletionResponse, got {type(resp).__name__}")

            if verify is None:
                return resp.content

            verdict = (
                verify.text(f"Question: {task}\nAnswer: {resp.content}")
                if hasattr(verify, "text")
                else verify(str(task), resp.content)
            )
            if verdict.strip().lower().startswith("approved"):
                return resp.content

            messages = (
                f"{task}\n\nPrevious attempt was rejected with this feedback: {verdict}\n"
                "Please address the feedback and try again."
            )

        warnings.warn(
            f"text() verify exhausted after {max_verify} attempt(s) without approval. "
            "Returning last result unchanged.",
            UserWarning,
            stacklevel=2,
        )
        return resp.content  # type: ignore[return-value]

    # Appended to the system prompt on every json()/ajson() call so models
    # don't produce markdown or preamble even when native structured output is
    # available — belt-and-suspenders reinforcement.
    _JSON_SYSTEM_SUFFIX = (
        "Respond with a single valid JSON object matching the required schema. "
        "No preamble, no explanation, no markdown — JSON only."
    )

    def json(
        self,
        messages: str | list,
        schema: type | dict | None = None,
        *,
        verify: Verifier | Callable[..., Any] | None = None,
        max_verify: int = 3,
        **kwargs,
    ) -> Any:
        effective_schema = schema if schema is not None else self.output_schema
        if effective_schema is None:
            raise TypeError("json() requires a schema — pass schema= or set output_schema= on the agent.")
        if kwargs.get("stream"):
            raise TypeError("stream=True is not supported in json(). Use chat(stream=True) instead.")
        existing = kwargs.pop("system", None)
        kwargs["system"] = f"{existing}\n\n{self._JSON_SYSTEM_SUFFIX}" if existing else self._JSON_SYSTEM_SUFFIX

        # When the agent has tools, use loop() so tool rounds complete before
        # structured parsing.  Passing output_schema + response_format to a
        # provider in the same call as tools causes an empty-content response
        # (the model returns a tool call instead of JSON).
        has_tools = bool(
            self.tools or self.native_tools
            or kwargs.get("tools") or kwargs.get("native_tools")
        )

        task = messages  # keep original task for judge context
        _verify_log: list[str] = []
        for _attempt in range(max_verify if verify is not None else 1):
            if has_tools:
                resp = self.loop(messages, force_final_after_tools=True, **kwargs)
                from lazybridge.core.structured import apply_structured_validation, build_repair_messages

                apply_structured_validation(resp, resp.content, effective_schema)
                if resp.validation_error:
                    # Loop produced content but it failed schema validation.
                    # Ask the model to reformat without re-running tools.
                    repair_msgs = build_repair_messages(
                        [{"role": "user", "content": str(messages)}],
                        resp.content,
                        effective_schema,
                        resp.validation_error,
                    )
                    repair_resp = self.chat(
                        repair_msgs, output_schema=effective_schema,
                        memory=None, tools=[], native_tools=[],
                    )
                    if isinstance(repair_resp, CompletionResponse):
                        resp = repair_resp
                resp.raise_if_failed()
            else:
                resp = self.chat(messages, output_schema=effective_schema, **kwargs)
                if not isinstance(resp, CompletionResponse):
                    raise TypeError(f"Expected CompletionResponse, got {type(resp).__name__}")
                resp.raise_if_failed()

            if verify is None:
                return resp.parsed

            # Ask the judge
            answer = resp.content
            verdict = (
                verify.text(f"Question: {task}\nAnswer: {answer}")
                if hasattr(verify, "text")
                else verify(str(task), answer)
            )
            if verdict.strip().lower().startswith("approved"):
                return resp.parsed

            _verify_log.append(verdict)
            # Append feedback and retry
            messages = (
                f"{task}\n\nPrevious attempt was rejected with this feedback: {verdict}\n"
                "Please address the feedback and try again."
            )

        warnings.warn(
            f"json() verify exhausted after {max_verify} attempt(s) without approval. "
            "Returning last result unchanged.",
            UserWarning,
            stacklevel=2,
        )
        return resp.parsed  # type: ignore[return-value]

    async def atext(
        self,
        messages: str | list,
        *,
        verify: Verifier | Callable[..., Any] | None = None,
        max_verify: int = 3,
        **kwargs,
    ) -> str:
        if kwargs.get("stream"):
            raise TypeError("stream=True is not supported in atext(). Use achat(stream=True) instead.")

        task = messages
        for _attempt in range(max_verify if verify is not None else 1):
            resp = await self.achat(messages, **kwargs)
            if not isinstance(resp, CompletionResponse):
                raise TypeError(f"Expected CompletionResponse, got {type(resp).__name__}")

            if verify is None:
                return resp.content

            if hasattr(verify, "atext"):
                verdict = await verify.atext(f"Question: {task}\nAnswer: {resp.content}")
            elif inspect.iscoroutinefunction(verify):
                verdict = await verify(str(task), resp.content)
            else:
                verdict = verify.text(f"Question: {task}\nAnswer: {resp.content}") if hasattr(verify, "text") else verify(str(task), resp.content)

            if verdict.strip().lower().startswith("approved"):
                return resp.content

            messages = (
                f"{task}\n\nPrevious attempt was rejected with this feedback: {verdict}\n"
                "Please address the feedback and try again."
            )

        warnings.warn(
            f"atext() verify exhausted after {max_verify} attempt(s) without approval. "
            "Returning last result unchanged.",
            UserWarning,
            stacklevel=2,
        )
        return resp.content  # type: ignore[return-value]

    async def ajson(
        self,
        messages: str | list,
        schema: type | dict | None = None,
        *,
        verify: Verifier | Callable[..., Any] | None = None,
        max_verify: int = 3,
        **kwargs,
    ) -> Any:
        effective_schema = schema if schema is not None else self.output_schema
        if effective_schema is None:
            raise TypeError("ajson() requires a schema — pass schema= or set output_schema= on the agent.")
        if kwargs.get("stream"):
            raise TypeError("stream=True is not supported in ajson(). Use achat(stream=True) instead.")
        existing = kwargs.pop("system", None)
        kwargs["system"] = f"{existing}\n\n{self._JSON_SYSTEM_SUFFIX}" if existing else self._JSON_SYSTEM_SUFFIX

        has_tools = bool(
            self.tools or self.native_tools
            or kwargs.get("tools") or kwargs.get("native_tools")
        )

        task = messages
        for _attempt in range(max_verify if verify is not None else 1):
            if has_tools:
                resp = await self.aloop(messages, force_final_after_tools=True, **kwargs)
                from lazybridge.core.structured import apply_structured_validation, build_repair_messages

                apply_structured_validation(resp, resp.content, effective_schema)
                if resp.validation_error:
                    repair_msgs = build_repair_messages(
                        [{"role": "user", "content": str(messages)}],
                        resp.content,
                        effective_schema,
                        resp.validation_error,
                    )
                    repair_resp = await self.achat(
                        repair_msgs, output_schema=effective_schema,
                        memory=None, tools=[], native_tools=[],
                    )
                    if isinstance(repair_resp, CompletionResponse):
                        resp = repair_resp
                resp.raise_if_failed()
            else:
                resp = await self.achat(messages, output_schema=effective_schema, **kwargs)
                if not isinstance(resp, CompletionResponse):
                    raise TypeError(f"Expected CompletionResponse, got {type(resp).__name__}")
                resp.raise_if_failed()

            if verify is None:
                return resp.parsed

            answer = resp.content
            if hasattr(verify, "atext"):
                verdict = await verify.atext(f"Question: {task}\nAnswer: {answer}")
            elif inspect.iscoroutinefunction(verify):
                verdict = await verify(str(task), answer)
            else:
                verdict = verify(str(task), answer) if callable(verify) else verify.text(f"Question: {task}\nAnswer: {answer}")

            if verdict.strip().lower().startswith("approved"):
                return resp.parsed

            messages = (
                f"{task}\n\nPrevious attempt was rejected with this feedback: {verdict}\n"
                "Please address the feedback and try again."
            )

        warnings.warn(
            f"ajson() verify exhausted after {max_verify} attempt(s) without approval. "
            "Returning last result unchanged.",
            UserWarning,
            stacklevel=2,
        )
        return resp.parsed  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Dedicated streaming methods (unambiguous return types)
    # ------------------------------------------------------------------

    def chat_stream(
        self,
        messages: str | list,
        *,
        system: str | None = None,
        tools: list | None = None,
        native_tools: list[NativeTool | str] | None = None,
        output_schema: type | dict | None = None,
        thinking: bool | ThinkingConfig = False,
        skills: list[str] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tool_choice: str | None = None,
        context: LazyContext | Callable[[], str] | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """Stream a chat response. Always returns ``Iterator[StreamChunk]``.

        This is the recommended way to stream — the return type is unambiguous,
        unlike ``chat(stream=True)`` which returns a union type.

        Usage::

            for chunk in agent.chat_stream("tell me a story"):
                print(chunk.delta, end="", flush=True)
        """
        result = self.chat(
            messages,
            system=system,
            tools=tools,
            native_tools=native_tools,
            output_schema=output_schema,
            thinking=thinking,
            skills=skills,
            stream=True,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            tool_choice=tool_choice,
            context=context,
            **kwargs,
        )
        return result  # type: ignore[return-value]

    async def achat_stream(
        self,
        messages: str | list,
        *,
        system: str | None = None,
        tools: list | None = None,
        native_tools: list[NativeTool | str] | None = None,
        output_schema: type | dict | None = None,
        thinking: bool | ThinkingConfig = False,
        skills: list[str] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tool_choice: str | None = None,
        context: LazyContext | Callable[[], str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Async stream. Always returns ``AsyncIterator[StreamChunk]``.

        Usage::

            async for chunk in await agent.achat_stream("tell me a story"):
                print(chunk.delta, end="", flush=True)
        """
        result = await self.achat(
            messages,
            system=system,
            tools=tools,
            native_tools=native_tools,
            output_schema=output_schema,
            thinking=thinking,
            skills=skills,
            stream=True,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            tool_choice=tool_choice,
            context=context,
            **kwargs,
        )
        return result  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Guidance rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _render_guidance(tool_set: NormalizedToolSet) -> str | None:
        parts = []
        for tool in tool_set.bridges:
            if tool.guidance:
                parts.append(f"[{tool.name}]\n{tool.guidance}")
        return "\n\n".join(parts) if parts else None

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        session_info = f", session={self.session.id[:8]}..." if self.session else ""
        return (
            f"LazyAgent(name={self.name!r}, provider={self._provider_name!r}, model={self._model_name!r}{session_info})"
        )
