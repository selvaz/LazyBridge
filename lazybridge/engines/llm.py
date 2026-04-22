"""LLMEngine — agentic tool-calling loop over any provider."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from typing import TYPE_CHECKING, Any, Literal

from lazybridge.core.executor import Executor
from lazybridge.core.types import (
    CompletionRequest,
    CompletionResponse,
    Message,
    NativeTool,
    Role,
    StructuredOutputConfig,
    ToolCall,
)
from lazybridge.envelope import Envelope, EnvelopeMetadata, ErrorInfo
from lazybridge.session import EventType

if TYPE_CHECKING:
    from lazybridge.memory import Memory
    from lazybridge.session import Session
    from lazybridge.tools import Tool


class LLMEngine:
    """Drives the LLM ↔ tool-call loop for a single agent invocation.

    Parameters
    ----------
    model:
        Model string, e.g. "claude-opus-4-7". Provider is inferred automatically.
    thinking:
        Enable extended thinking (Anthropic) or reasoning (OpenAI o-series).
    max_turns:
        Maximum tool-call rounds before giving up with MaxTurnsExceeded error.
    tool_choice:
        "auto" — provider decides when to call tools (default).
        "any"  — provider must call at least one tool.

        When the model emits multiple tool calls in a single turn,
        LazyBridge always executes them concurrently via ``asyncio.gather``.
        That is a capability of the engine, not a configuration knob;
        there is no "serial" execution path for LLM-emitted tool calls.
    temperature:
        Sampling temperature. None = provider default.
    system:
        Static system prompt. Agent.sources= / Envelope.context are added on top.
    native_tools:
        Provider-native server-side tools, e.g. NativeTool.WEB_SEARCH.
    max_retries:
        Retries on transient provider errors (429, 5xx, network/timeout).
        Default 3 — production-safe.  Pass 0 to disable.
    retry_delay:
        Base delay (seconds) for exponential backoff with ±10% jitter.
    request_timeout:
        Per-completion deadline in seconds.  Caps the time a hung
        provider can block an agent run.  ``None`` disables the
        framework-level timeout and defers to the provider SDK.
    """

    def __init__(
        self,
        model: str,
        *,
        provider: str | None = None,
        thinking: bool = False,
        max_turns: int = 10,
        tool_choice: Literal["auto", "any"] = "auto",
        temperature: float | None = None,
        system: str | None = None,
        native_tools: list[NativeTool | str] | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        request_timeout: float | None = 120.0,
    ) -> None:
        self.model = model
        self.thinking = thinking
        self.max_turns = max_turns
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_timeout = request_timeout
        # Backward-compat: accept ``"parallel"`` but collapse to ``"auto"``
        # with a deprecation warning. The framework no longer has a
        # separate "parallel mode" — tool calls are always executed via
        # asyncio.gather when the model emits more than one in a turn.
        if tool_choice == "parallel":
            import warnings

            warnings.warn(
                "LLMEngine(tool_choice='parallel') is deprecated. "
                "Concurrent tool execution is now the default and cannot be "
                "disabled; drop the argument (or use 'auto'/'any').",
                DeprecationWarning,
                stacklevel=2,
            )
            tool_choice = "auto"
        self.tool_choice = tool_choice
        self.temperature = temperature
        self.system = system
        self.native_tools: list[NativeTool] = [
            NativeTool(t) if isinstance(t, str) else t
            for t in (native_tools or [])
        ]
        # Provider may be passed explicitly (used by Agent.from_provider
        # when the model is a tier alias like "top" / "cheap" that
        # _infer_provider can't route on its own).  Falls back to the
        # inference heuristic on the model string.
        self.provider = provider or self._infer_provider(model)

    # Provider name aliases accepted as the model argument
    _PROVIDER_NAMES = {"anthropic", "claude", "openai", "gpt", "google", "gemini", "deepseek"}

    #: Exact-match provider aliases — first lookup.  ``Agent("anthropic")`` and
    #: ``Agent("claude")`` both resolve to the ``anthropic`` provider.  Users
    #: can extend this at runtime via :meth:`register_provider_alias`.
    _PROVIDER_ALIASES: dict[str, str] = {
        "anthropic": "anthropic",
        "claude": "anthropic",
        "openai": "openai",
        "gpt": "openai",
        "google": "google",
        "gemini": "google",
        "deepseek": "deepseek",
    }

    #: Ordered substring / prefix rules applied when no exact alias matches.
    #: Each entry is ``(kind, pattern, provider)`` where ``kind`` is
    #: ``"contains"`` or ``"startswith"``.  First match wins.  Users can
    #: prepend new rules via :meth:`register_provider_rule` — newly
    #: registered rules take priority over built-ins so shipping a new
    #: "claude-opus-5-*" alias is one call, not a code edit.
    _PROVIDER_RULES: list[tuple[str, str, str]] = [
        ("contains",  "claude",   "anthropic"),
        ("contains",  "gpt",      "openai"),
        ("startswith", "o1",      "openai"),
        ("startswith", "o3",      "openai"),
        ("contains",  "gemini",   "google"),
        ("contains",  "deepseek", "deepseek"),
    ]

    #: Fallback provider when nothing matches.  A warning is emitted on
    #: fallback so ``Agent("grok-2")`` (an unrecognised model) does NOT
    #: silently get routed to Anthropic and fail with a cryptic API-side
    #: error.  Set to ``None`` in a subclass to disable the fallback and
    #: raise ``ValueError`` instead.
    _PROVIDER_DEFAULT: "str | None" = "anthropic"

    @classmethod
    def register_provider_alias(cls, alias: str, provider: str) -> None:
        """Register an exact-match model-string → provider alias.

        Example::

            LLMEngine.register_provider_alias("mistral", "mistral")
            Agent("mistral")   # resolves to the mistral provider
        """
        cls._PROVIDER_ALIASES = {**cls._PROVIDER_ALIASES, alias.lower(): provider}

    @classmethod
    def register_provider_rule(
        cls,
        pattern: str,
        provider: str,
        *,
        kind: Literal["contains", "startswith"] = "contains",
    ) -> None:
        """Register a substring / prefix routing rule.

        New rules take priority over built-ins so you can override default
        routing without editing the framework source::

            LLMEngine.register_provider_rule("claude-opus-5", "anthropic")
            Agent("claude-opus-5-20260701")   # routed to anthropic
        """
        cls._PROVIDER_RULES = [(kind, pattern.lower(), provider), *cls._PROVIDER_RULES]

    @classmethod
    def _infer_provider(cls, model: str) -> str:
        m = model.lower()
        alias = cls._PROVIDER_ALIASES.get(m)
        if alias is not None:
            return alias
        for kind, pattern, provider in cls._PROVIDER_RULES:
            if kind == "contains" and pattern in m:
                return provider
            if kind == "startswith" and m.startswith(pattern):
                return provider
        # Nothing matched — warn loudly rather than silently route to
        # the default provider.  Raising would break the "no-config
        # Agent('some-model')" ergonomic, so we stick with a warning
        # and let the provider surface its own "unknown model" error.
        if cls._PROVIDER_DEFAULT is None:
            raise ValueError(
                f"No provider rule matches model {model!r} and no default is "
                f"configured. Register a rule via "
                f"LLMEngine.register_provider_rule(...) or set "
                f"_PROVIDER_DEFAULT on a subclass."
            )
        import warnings

        warnings.warn(
            f"No provider rule matches model {model!r}; defaulting to "
            f"{cls._PROVIDER_DEFAULT!r}. Register a rule via "
            f"LLMEngine.register_provider_rule({model!r}, <provider>) to silence.",
            stacklevel=3,
        )
        return cls._PROVIDER_DEFAULT

    def _make_executor(self) -> Executor:
        # Use ``self.provider`` (set at __init__, possibly explicitly by
        # Agent.from_provider) rather than re-running inference.  The
        # BaseProvider handles tier / provider-name aliases on the model
        # string via ``resolve_model_alias``; there is no reason to
        # special-case them here.
        return Executor(
            self.provider,
            model=self.model,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        )

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------

    async def run(
        self,
        env: Envelope,
        *,
        tools: list["Tool"],
        output_type: type,
        memory: "Memory | None",
        session: "Session | None",
    ) -> Envelope:
        run_id = str(uuid.uuid4())
        t_start = time.monotonic()
        agent_name = getattr(self, "_agent_name", "agent")

        if session:
            session.emit(EventType.AGENT_START, {"agent_name": agent_name, "task": env.task}, run_id=run_id)

        try:
            result = await self._loop(
                env, tools=tools, output_type=output_type,
                memory=memory, session=session, run_id=run_id,
            )
        except Exception as exc:
            if session:
                session.emit(EventType.AGENT_FINISH, {"agent_name": agent_name, "error": str(exc)}, run_id=run_id)
            return Envelope.error_envelope(exc)

        latency_ms = (time.monotonic() - t_start) * 1000
        # Rebuild metadata via ``model_copy`` so we don't rely on
        # Pydantic v2's default-mutable semantics.  This stays correct
        # even if a future version ships with ``frozen=True`` on
        # EnvelopeMetadata.
        result = result.model_copy(update={
            "metadata": result.metadata.model_copy(update={
                "latency_ms": latency_ms,
                "run_id": run_id,
            }),
        })

        if session:
            session.emit(
                EventType.AGENT_FINISH,
                {"agent_name": agent_name, "payload": result.text(), "latency_ms": latency_ms},
                run_id=run_id,
            )

        return result

    # ------------------------------------------------------------------
    # _loop() — shared by run() and stream()
    # ------------------------------------------------------------------

    async def _loop(
        self,
        env: Envelope,
        *,
        tools: list["Tool"],
        output_type: type,
        memory: "Memory | None",
        session: "Session | None",
        run_id: str,
        # When truthy, yield str tokens instead of building Envelope
        _stream_sink: "asyncio.Queue[str | None] | None" = None,
    ) -> Envelope:
        from lazybridge.core.types import TextContent, ThinkingConfig, ToolResultContent, ToolUseContent
        from pydantic import BaseModel

        executor = self._make_executor()

        messages: list[Message] = []
        if memory:
            messages.extend(memory.messages())

        system = self.system or ""
        if env.context:
            system = (f"{system}\n\nContext:\n{env.context}".strip()
                      if system else f"Context:\n{env.context}")

        task_text = env.task or env.text()
        messages.append(Message(role=Role.USER, content=task_text))

        tool_defs = [t.definition() for t in tools]
        tool_map = {t.name: t for t in tools}

        structured_cfg: StructuredOutputConfig | None = None
        # Activate structured output when the caller declared a non-str
        # output type — including generic forms like ``list[MyModel]``
        # or ``dict[str, MyModel]``.  Pre-fix the ``isinstance(type)``
        # filter rejected generics because ``list[X]`` isn't a ``type``,
        # so Agent(output=list[Summary]) silently got plain text back.
        if output_type is not str and output_type is not Any:
            from typing import get_origin
            if isinstance(output_type, type) or get_origin(output_type) is not None:
                structured_cfg = StructuredOutputConfig(schema=output_type)

        thinking_cfg = ThinkingConfig(enabled=True) if self.thinking else None
        # "parallel" is our asyncio strategy; provider always gets "auto"
        provider_tc = self.tool_choice

        total_in = total_out = 0
        cost = 0.0
        # Nested aggregation: when a tool returns an ``Envelope`` (an
        # Agent wrapped via ``as_tool``), its own metadata is added to
        # these buckets so the outer Envelope reflects total pipeline
        # cost without double-counting with ``total_in`` / ``total_out``.
        nested_in = nested_out = 0
        nested_cost = 0.0
        model_used: str | None = None

        for turn in range(self.max_turns):
            if session:
                session.emit(EventType.LOOP_STEP, {"turn": turn, "messages": len(messages)}, run_id=run_id)

            req = CompletionRequest(
                messages=messages,
                system=system or None,
                temperature=self.temperature,
                tools=tool_defs,
                native_tools=self.native_tools,
                tool_choice=provider_tc if tool_defs else None,
                structured_output=structured_cfg if not tool_defs else None,
                thinking=thinking_cfg,
                stream=_stream_sink is not None,
            )

            if session:
                session.emit(
                    EventType.MODEL_REQUEST,
                    {"provider": executor._provider.__class__.__name__, "model": self.model, "turn": turn},
                    run_id=run_id,
                )

            if _stream_sink is not None:
                # Streaming path — accumulate full response while yielding tokens.
                # request_timeout intentionally does NOT wrap streaming:
                # a slow stream is still making progress, and wrapping the
                # whole iterator in ``wait_for`` would cancel on total
                # duration rather than stall duration.
                resp = await self._stream_turn(executor, req, _stream_sink)
            elif self.request_timeout is not None:
                resp = await asyncio.wait_for(
                    executor.aexecute(req), timeout=self.request_timeout
                )
            else:
                resp = await executor.aexecute(req)

            model_used = resp.model or self.model
            total_in += resp.usage.input_tokens
            total_out += resp.usage.output_tokens
            if resp.usage.cost_usd:
                cost += resp.usage.cost_usd

            if session:
                session.emit(
                    EventType.MODEL_RESPONSE,
                    {
                        "content": resp.content[:500],
                        "input_tokens": resp.usage.input_tokens,
                        "output_tokens": resp.usage.output_tokens,
                        "cost_usd": resp.usage.cost_usd,
                        "stop_reason": resp.stop_reason,
                    },
                    run_id=run_id,
                )

            if not resp.tool_calls:
                payload: Any
                if structured_cfg and resp.parsed:
                    payload = resp.parsed
                else:
                    payload = resp.content

                if memory:
                    memory.add(task_text, resp.content, tokens=total_in + total_out)

                return Envelope(
                    task=env.task,
                    context=env.context,
                    payload=payload,
                    metadata=EnvelopeMetadata(
                        input_tokens=total_in,
                        output_tokens=total_out,
                        cost_usd=cost,
                        nested_input_tokens=nested_in,
                        nested_output_tokens=nested_out,
                        nested_cost_usd=nested_cost,
                        model=model_used,
                        provider=executor._provider.__class__.__name__,
                    ),
                )

            # Append assistant turn with tool calls
            assistant_blocks: list[Any] = []
            if resp.content:
                assistant_blocks.append(TextContent(text=resp.content))
            for tc in resp.tool_calls:
                assistant_blocks.append(ToolUseContent(id=tc.id, name=tc.name, input=tc.arguments))
            messages.append(Message(role=Role.ASSISTANT, content=assistant_blocks))

            # Execute tool calls — always concurrently.  A single tool call
            # in a turn is just a one-element gather; N calls run in parallel.
            # Tool-is-Tool uniformity: each ``tc`` may target a plain
            # function, an Agent wrapped via ``as_tool()``, or an Agent of
            # Agents — the engine does not special-case any of them.
            raw_results = await asyncio.gather(
                *[self._exec_tool(tc, tool_map, session=session, run_id=run_id)
                  for tc in resp.tool_calls],
                return_exceptions=True,
            )

            result_blocks: list[Any] = []
            for tc, tr in zip(resp.tool_calls, raw_results):
                # Detect Envelope-returning tools (agent-as-tool,
                # verified tools): preserve their metadata and error
                # state, stringify via .text() for the provider.
                if isinstance(tr, Envelope):
                    # Aggregate usage. Include the nested Envelope's
                    # own ``nested_*`` so aggregation is transitive
                    # across arbitrarily-deep agent trees.
                    nm = tr.metadata
                    nested_in += (nm.input_tokens + nm.nested_input_tokens)
                    nested_out += (nm.output_tokens + nm.nested_output_tokens)
                    nested_cost += (nm.cost_usd + nm.nested_cost_usd)
                    content = tr.text()
                    is_err = not tr.ok
                elif isinstance(tr, Exception):
                    content = f"Tool error: {tr}"
                    is_err = True
                else:
                    content = str(tr)
                    is_err = False
                result_blocks.append(
                    ToolResultContent(
                        tool_use_id=tc.id,
                        content=content,
                        tool_name=tc.name,
                        is_error=is_err,
                    )
                )
            messages.append(Message(role=Role.USER, content=result_blocks))

        return Envelope(
            task=env.task,
            error=ErrorInfo(
                type="MaxTurnsExceeded",
                message=f"Reached max_turns={self.max_turns}",
                retryable=False,
            ),
            metadata=EnvelopeMetadata(
                input_tokens=total_in, output_tokens=total_out, cost_usd=cost,
                nested_input_tokens=nested_in, nested_output_tokens=nested_out,
                nested_cost_usd=nested_cost, model=model_used,
            ),
        )

    async def _stream_turn(
        self,
        executor: Executor,
        req: CompletionRequest,
        sink: "asyncio.Queue[str | None]",
    ) -> CompletionResponse:
        """Stream one LLM turn, push tokens to sink, return reconstructed CompletionResponse."""
        from lazybridge.core.types import CompletionResponse, UsageStats

        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        stop_reason = "end_turn"
        usage = UsageStats()
        model_out: str | None = None

        async for chunk in executor.astream(req):
            if chunk.delta:
                content_parts.append(chunk.delta)
                await sink.put(chunk.delta)
            if chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)
            if chunk.stop_reason:
                stop_reason = chunk.stop_reason
            if chunk.usage:
                usage = chunk.usage
            if chunk.is_final:
                model_out = getattr(chunk, "model", None)

        return CompletionResponse(
            content="".join(content_parts),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
            model=model_out,
        )

    async def _exec_tool(
        self,
        tc: ToolCall,
        tool_map: dict[str, "Tool"],
        *,
        session: "Session | None",
        run_id: str,
    ) -> Any:
        if session:
            session.emit(EventType.TOOL_CALL, {"tool": tc.name, "arguments": tc.arguments}, run_id=run_id)

        tool = tool_map.get(tc.name)
        if tool is None:
            err = RuntimeError(f"Unknown tool: {tc.name!r}")
            if session:
                session.emit(EventType.TOOL_ERROR, {"tool": tc.name, "error": str(err)}, run_id=run_id)
            return err

        try:
            result = await tool.run(**tc.arguments)
            if session:
                session.emit(EventType.TOOL_RESULT, {"tool": tc.name, "result": str(result)[:500]}, run_id=run_id)
            return result
        except Exception as exc:
            if session:
                session.emit(
                    EventType.TOOL_ERROR,
                    {"tool": tc.name, "error": str(exc), "type": type(exc).__name__},
                    run_id=run_id,
                )
            return exc

    # ------------------------------------------------------------------
    # stream() — full multi-turn loop with token streaming
    # ------------------------------------------------------------------

    async def stream(
        self,
        env: Envelope,
        *,
        tools: list["Tool"],
        output_type: type,
        memory: "Memory | None",
        session: "Session | None",
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from the full tool-calling loop.

        Yields str tokens as the LLM generates them. Tool calls between turns
        are executed silently; the next-turn response is then streamed.
        This means token output is continuous across tool-call boundaries.
        """
        run_id = str(uuid.uuid4())
        agent_name = getattr(self, "_agent_name", "agent")

        if session:
            session.emit(EventType.AGENT_START, {"agent_name": agent_name, "task": env.task}, run_id=run_id)

        sink: asyncio.Queue[str | None] = asyncio.Queue()

        async def _run_loop() -> None:
            try:
                await self._loop(
                    env, tools=tools, output_type=output_type,
                    memory=memory, session=session, run_id=run_id,
                    _stream_sink=sink,
                )
            finally:
                await sink.put(None)  # sentinel — loop done

        task = asyncio.create_task(_run_loop())
        try:
            while True:
                token = await sink.get()
                if token is None:
                    break
                yield token
        finally:
            await task  # propagate exceptions
