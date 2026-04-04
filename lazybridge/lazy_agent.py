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

import inspect
import logging
import uuid
import warnings
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

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
)
from lazybridge.lazy_context import LazyContext
from lazybridge.lazy_session import Event, EventLog, LazySession, TrackLevel
from lazybridge.lazy_tool import LazyTool, NormalizedToolSet
from lazybridge.memory import Memory

_logger = logging.getLogger(__name__)

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
                    UserWarning, stacklevel=3,
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
        if isinstance(m, Message) and str(m.role) in ("user", "Role.USER"):
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
        max_retries: int = 0,
        api_key: str | None = None,
        verbose: bool = False,
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
        self.native_tools: list[NativeTool] = [
            NativeTool(t) if isinstance(t, str) else t for t in (native_tools or [])
        ]
        self.session = session

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
            self._log = None

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
        """Merge call-level tools with agent-level tools and normalise."""
        all_tools = list(tools or []) + list(self.tools)
        return NormalizedToolSet.from_list(all_tools) if all_tools else NormalizedToolSet([], [], {})

    def _merge_native_tools(self, call_level: list[NativeTool | str] | None) -> list[NativeTool]:
        """Return agent-level native tools plus any call-level extras (deduplicated)."""
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
        parts: list[str] = []
        for chunk in gen:
            if chunk.delta is not None:
                parts.append(chunk.delta)
            yield chunk
        self._last_output = "".join(parts)

    async def _astream_and_track(self, gen: AsyncIterator[StreamChunk]) -> AsyncIterator[StreamChunk]:
        parts: list[str] = []
        async for chunk in gen:
            if chunk.delta is not None:
                parts.append(chunk.delta)
            yield chunk
        self._last_output = "".join(parts)

    # ------------------------------------------------------------------
    # chat() — single turn
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: str | list,
        *,
        memory: Memory | None = None,
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
        **kwargs: Any,
    ) -> CompletionResponse | Iterator[StreamChunk]:
        """Send a single-turn chat request.

        Pass ``memory`` to accumulate conversation history automatically::

            mem = Memory()
            ai.chat("ciao", memory=mem)
            ai.chat("ricordi?", memory=mem)   # history inclusa automaticamente

        ``tool_choice`` controls tool selection: ``"auto"`` (default), ``"none"``,
        ``"required"`` (force at least one tool call), or a specific tool name.

        ``context`` overrides the agent-level context for this call only.
        """
        if memory is not None:
            if not isinstance(messages, str):
                raise TypeError(
                    "memory= requires messages to be a str. "
                    "For list messages manage history manually via chat(list)."
                )
            if stream:
                raise TypeError(
                    "stream=True is not compatible with memory=. "
                    "Consume the stream manually and call memory._record() yourself."
                )
            full = memory._build_input(messages)
            resp = self.chat(full, system=system, tools=tools, native_tools=native_tools,
                             output_schema=output_schema, thinking=thinking, skills=skills,
                             stream=False, model=model, max_tokens=max_tokens,
                             temperature=temperature, tool_choice=tool_choice,
                             context=context, **kwargs)
            if not isinstance(resp, CompletionResponse):  # pragma: no cover
                raise TypeError(f"Expected CompletionResponse, got {type(resp).__name__}")
            memory._record(messages, resp.content)
            return resp

        msgs = _normalise_messages(messages)
        tool_set = self._build_tool_set(tools)
        effective_system = self._build_effective_system(system, context, tool_set)
        # Agent-level output_schema is the default; call-level takes precedence.
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

        self._track(Event.MODEL_REQUEST, model=request.model or self._model_name, n_messages=len(msgs))
        if stream:
            return self._stream_and_track(self._executor.stream(request))

        resp = self._executor.execute(request)
        self._last_output = resp.content
        self._last_response = resp
        self._track(Event.MODEL_RESPONSE,
                    model=resp.model or self._model_name,
                    stop_reason=resp.stop_reason,
                    input_tokens=resp.usage.input_tokens,
                    output_tokens=resp.usage.output_tokens,
                    content=resp.content)
        return resp

    async def achat(
        self,
        messages: str | list,
        *,
        memory: Memory | None = None,
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
        **kwargs: Any,
    ) -> CompletionResponse | AsyncIterator[StreamChunk]:
        """Async version of chat(). Accepts memory= and tool_choice=."""
        if memory is not None:
            if not isinstance(messages, str):
                raise TypeError(
                    "memory= requires messages to be a str. "
                    "For list messages manage history manually via achat(list)."
                )
            if stream:
                raise TypeError(
                    "stream=True is not compatible with memory=. "
                    "Consume the stream manually and call memory._record() yourself."
                )
            full = memory._build_input(messages)
            resp = await self.achat(full, system=system, tools=tools, native_tools=native_tools,
                                    output_schema=output_schema, thinking=thinking, skills=skills,
                                    stream=False, model=model, max_tokens=max_tokens,
                                    temperature=temperature, tool_choice=tool_choice,
                                    context=context, **kwargs)
            if not isinstance(resp, CompletionResponse):  # pragma: no cover
                raise TypeError(f"Expected CompletionResponse, got {type(resp).__name__}")
            memory._record(messages, resp.content)
            return resp

        msgs = _normalise_messages(messages)
        tool_set = self._build_tool_set(tools)
        effective_system = self._build_effective_system(system, context, tool_set)
        # Agent-level output_schema is the default; call-level takes precedence.
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

        self._track(Event.MODEL_REQUEST, model=request.model or self._model_name, n_messages=len(msgs))
        if stream:
            return self._astream_and_track(self._executor.astream(request))

        resp = await self._executor.aexecute(request)
        self._last_output = resp.content
        self._last_response = resp
        self._track(Event.MODEL_RESPONSE,
                    model=resp.model or self._model_name,
                    stop_reason=resp.stop_reason,
                    input_tokens=resp.usage.input_tokens,
                    output_tokens=resp.usage.output_tokens,
                    content=resp.content)
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
        verify: "LazyAgent | Callable[[str, str], str] | None" = None,
        max_verify: int = 3,
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
        """
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        if chat_kwargs.get("stream"):
            raise TypeError("stream=True is not supported in loop(). Use chat() for streaming.")

        _orig_q = _messages_to_str(messages)
        _current_messages: str | list = messages
        _attempts = max(1, max_verify) if verify is not None else 1
        resp: CompletionResponse | None = None
        _verify_log: list[str] = []
        step = 0

        self._track(Event.AGENT_START, method="loop", task=_orig_q[:200])

        for _attempt in range(_attempts):
            # Build tool_set once for the execution registry used in _execute_tool().
            # Do NOT pass tool_set.bridges to chat() — chat() calls _build_tool_set()
            # internally and would add self.tools a second time, causing duplicate names.
            tool_set = self._build_tool_set(tools)
            convo = _normalise_messages(_current_messages)

            for step in range(max_steps):
                resp = self.chat(
                    convo,
                    tools=tools,  # pass originals; chat() merges with self.tools once
                    native_tools=native_tools,
                    **chat_kwargs,
                )  # type: ignore[assignment]

                if on_event:
                    on_event("step", {"step": step, "response": resp})

                if not resp.tool_calls:
                    break

                self._track(Event.LOOP_STEP, step=step, n_tool_calls=len(resp.tool_calls))

                # Append the assistant's turn (with tool-use blocks).
                # Build content list explicitly so text + thinking + tool blocks are never lost.
                _tc_blocks = [ToolUseContent(id=tc.id, name=tc.name, input=tc.arguments)
                              for tc in resp.tool_calls]
                _thinking_blocks: list[Any] = (
                    [ThinkingContent(thinking=resp.thinking)] if resp.thinking else []
                )
                if resp.tool_calls:
                    _text_blocks: list[Any] = (
                        [TextContent(text=resp.content)] if resp.content else []
                    )
                    _asst_content: Any = _thinking_blocks + _text_blocks + _tc_blocks
                else:
                    _asst_content = resp.content or ""
                convo.append(Message(role=Role.ASSISTANT, content=_asst_content))

                # Execute each tool call and append the result
                for tc in resp.tool_calls:
                    self._track(Event.TOOL_CALL, name=tc.name, arguments=tc.arguments)
                    if on_event:
                        on_event("tool_call", tc)
                    try:
                        result = self._execute_tool(tc, tool_set.registry, tool_runner)
                        self._track(Event.TOOL_RESULT, name=tc.name, result=str(result)[:500])
                        if on_event:
                            on_event("tool_result", {"call": tc, "result": result})
                        convo.append(_tool_result_message(tc, result))
                    except Exception as exc:
                        _logger.debug("Tool %r raised: %s", tc.name, exc, exc_info=True)
                        self._track(Event.TOOL_ERROR, name=tc.name, error=str(exc))
                        convo.append(_tool_result_message(tc, f"Error: {exc}", is_error=True))

            if resp is None:  # pragma: no cover
                raise AssertionError("loop produced no response — this is a bug, please report it")

            if on_event:
                on_event("done", resp)

            if verify is None:
                break

            _raw = (
                verify.text(f"Question: {_orig_q}\nAnswer: {resp.content}")
                if hasattr(verify, "text")
                else verify(_orig_q, resp.content)
            )
            _verdict: str = str(_raw) if _raw is not None else ""
            if _verdict[:30].lower().startswith("approved"):
                break

            _verify_log.append(_verdict)
            if on_event:
                on_event("verify_rejected", {"attempt": _attempt + 1, "verdict": _verdict})
            _current_messages = (
                f"{_orig_q}\n\nPrevious attempt rejected: {_verdict or '(no verdict)'}\nTry again."
            )

        resp.verify_log = _verify_log  # type: ignore[union-attr]
        self._last_output = resp.content  # type: ignore[union-attr]
        self._last_response = resp  # type: ignore[union-attr]
        self._track(Event.AGENT_FINISH, method="loop",
                    stop_reason=resp.stop_reason,  # type: ignore[union-attr]
                    n_steps=step + 1)
        return resp  # type: ignore[return-value]

    async def aloop(
        self,
        messages: str | list,
        *,
        tools: list | None = None,
        native_tools: list[NativeTool | str] | None = None,
        max_steps: int = 8,
        tool_runner: Callable[[str, dict], Any] | None = None,
        on_event: Callable[[str, Any], None] | None = None,
        verify: "LazyAgent | Callable[..., Any] | None" = None,
        max_verify: int = 3,
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

        _orig_q = _messages_to_str(messages)
        _current_messages: str | list = messages
        _attempts = max(1, max_verify) if verify is not None else 1
        resp: CompletionResponse | None = None
        _verify_log: list[str] = []
        step = 0

        self._track(Event.AGENT_START, method="aloop", task=_orig_q[:200])

        for _attempt in range(_attempts):
            tool_set = self._build_tool_set(tools)
            convo = _normalise_messages(_current_messages)

            for step in range(max_steps):
                resp = await self.achat(
                    convo,
                    tools=tools,  # pass originals; achat() merges with self.tools once
                    native_tools=native_tools,
                    **chat_kwargs,
                )  # type: ignore[assignment]

                if on_event:
                    await _call_event_async(on_event, "step", {"step": step, "response": resp})

                if not resp.tool_calls:
                    break

                self._track(Event.LOOP_STEP, step=step, n_tool_calls=len(resp.tool_calls))

                _tc_blocks = [ToolUseContent(id=tc.id, name=tc.name, input=tc.arguments)
                              for tc in resp.tool_calls]
                _thinking_blocks: list[Any] = (
                    [ThinkingContent(thinking=resp.thinking)] if resp.thinking else []
                )
                if resp.tool_calls:
                    _text_blocks: list[Any] = (
                        [TextContent(text=resp.content)] if resp.content else []
                    )
                    _asst_content: Any = _thinking_blocks + _text_blocks + _tc_blocks
                else:
                    _asst_content = resp.content or ""
                convo.append(Message(role=Role.ASSISTANT, content=_asst_content))

                for tc in resp.tool_calls:
                    self._track(Event.TOOL_CALL, name=tc.name, arguments=tc.arguments)
                    if on_event:
                        await _call_event_async(on_event, "tool_call", tc)
                    try:
                        result = await self._aexecute_tool(tc, tool_set.registry, tool_runner)
                        self._track(Event.TOOL_RESULT, name=tc.name, result=str(result)[:500])
                        if on_event:
                            await _call_event_async(on_event, "tool_result", {"call": tc, "result": result})
                        convo.append(_tool_result_message(tc, result))
                    except Exception as exc:
                        _logger.debug("Tool %r raised: %s", tc.name, exc, exc_info=True)
                        self._track(Event.TOOL_ERROR, name=tc.name, error=str(exc))
                        convo.append(_tool_result_message(tc, f"Error: {exc}", is_error=True))

            if resp is None:  # pragma: no cover
                raise AssertionError("loop produced no response — this is a bug, please report it")

            if on_event:
                await _call_event_async(on_event, "done", resp)

            if verify is None:
                break

            if hasattr(verify, "atext"):
                _raw = await verify.atext(
                    f"Question: {_orig_q}\nAnswer: {resp.content}"
                )
            elif inspect.iscoroutinefunction(verify):
                _raw = await verify(_orig_q, resp.content)
            else:
                _raw = verify(_orig_q, resp.content)
            _verdict: str = str(_raw) if _raw is not None else ""

            if _verdict[:30].lower().startswith("approved"):
                break

            _verify_log.append(_verdict)
            if on_event:
                await _call_event_async(on_event, "verify_rejected", {"attempt": _attempt + 1, "verdict": _verdict})
            _current_messages = (
                f"{_orig_q}\n\nPrevious attempt rejected: {_verdict or '(no verdict)'}\nTry again."
            )

        resp.verify_log = _verify_log  # type: ignore[union-attr]
        self._last_output = resp.content  # type: ignore[union-attr]
        self._last_response = resp  # type: ignore[union-attr]
        self._track(Event.AGENT_FINISH, method="aloop",
                    stop_reason=resp.stop_reason,  # type: ignore[union-attr]
                    n_steps=step + 1)
        return resp  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Tool execution helpers
    # ------------------------------------------------------------------

    def _execute_tool(
        self,
        call: ToolCall,
        registry: dict[str, LazyTool],
        runner: Callable | None,
    ) -> Any:
        # Registry lookup first (LazyTool with a known callable)
        if call.name in registry:
            return registry[call.name].run(call.arguments, parent=self)
        # Fallback to user-provided runner (e.g. native tool handling)
        if runner:
            return runner(call.name, call.arguments)
        raise RuntimeError(
            f"No handler for tool '{call.name}'. "
            "Add a LazyTool with that name or provide tool_runner."
        )

    async def _aexecute_tool(
        self,
        call: ToolCall,
        registry: dict[str, LazyTool],
        runner: Callable | None,
    ) -> Any:
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
            f"No handler for tool '{call.name}'. "
            "Add a LazyTool with that name or provide tool_runner."
        )

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
        strict: bool = False,
    ) -> LazyTool:
        """Wrap this agent as a LazyTool for use in another agent's loop.

        The tool's schema is always ``{"task": str}``.
        The task string is forwarded to this agent's loop() or chat().
        """
        return LazyTool.from_agent(
            self,
            name=name or self.name,
            description=description or self.description,
            guidance=guidance,
            output_schema=output_schema,
            native_tools=native_tools,
            system_prompt=system_prompt,
            strict=strict,
        )

    # ------------------------------------------------------------------
    # Convenience text/json shortcuts
    # ------------------------------------------------------------------

    def text(self, messages: str | list, **kwargs) -> str:
        if kwargs.get("stream"):
            raise TypeError("stream=True is not supported in text(). Use chat(stream=True) instead.")
        resp = self.chat(messages, **kwargs)
        assert isinstance(resp, CompletionResponse)
        return resp.content

    # Appended to the system prompt on every json()/ajson() call so models
    # don't produce markdown or preamble even when native structured output is
    # available — belt-and-suspenders reinforcement.
    _JSON_SYSTEM_SUFFIX = (
        "Respond with a single valid JSON object matching the required schema. "
        "No preamble, no explanation, no markdown — JSON only."
    )

    def json(self, messages: str | list, schema: type | dict, **kwargs) -> Any:
        if kwargs.get("stream"):
            raise TypeError("stream=True is not supported in json(). Use chat(stream=True) instead.")
        existing = kwargs.pop("system", None)
        kwargs["system"] = (
            f"{existing}\n\n{self._JSON_SYSTEM_SUFFIX}" if existing else self._JSON_SYSTEM_SUFFIX
        )
        resp = self.chat(messages, output_schema=schema, **kwargs)
        assert isinstance(resp, CompletionResponse)
        return resp.parsed

    async def atext(self, messages: str | list, **kwargs) -> str:
        if kwargs.get("stream"):
            raise TypeError("stream=True is not supported in atext(). Use achat(stream=True) instead.")
        resp = await self.achat(messages, **kwargs)
        assert isinstance(resp, CompletionResponse)
        return resp.content

    async def ajson(self, messages: str | list, schema: type | dict, **kwargs) -> Any:
        if kwargs.get("stream"):
            raise TypeError("stream=True is not supported in ajson(). Use achat(stream=True) instead.")
        existing = kwargs.pop("system", None)
        kwargs["system"] = (
            f"{existing}\n\n{self._JSON_SYSTEM_SUFFIX}" if existing else self._JSON_SYSTEM_SUFFIX
        )
        resp = await self.achat(messages, output_schema=schema, **kwargs)
        assert isinstance(resp, CompletionResponse)
        return resp.parsed

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
            f"LazyAgent(name={self.name!r}, "
            f"provider={self._provider_name!r}, "
            f"model={self._model_name!r}"
            f"{session_info})"
        )
