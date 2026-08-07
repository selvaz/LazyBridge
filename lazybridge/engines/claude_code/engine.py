"""Prototype ``Engine`` implementation backed by Claude Agent SDK."""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
import warnings
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar, get_origin

from lazybridge.engines.base import resolve_agent_name
from lazybridge.envelope import Envelope, EnvelopeMetadata
from lazybridge.session import EventType

from .mcp_adapter import to_mcp_tools
from .protocol import ClaudeSdkClient, ClaudeSdkOptions
from .sdk_client import AgentSdkClient, ClaudeSdkRequestError

_T = TypeVar("_T")

#: Exception types treated as transient and worth retrying — connection /
#: process-level failures, not logical errors (bad model, tool failure,
#: malformed args). Mirrors ``LLMEngine``'s "429/5xx/network/timeout" retry
#: policy at the granularity available to a CLI/SDK-backed engine: there is
#: no per-HTTP-call visibility here, only pass/fail on the whole SDK call.
_TRANSIENT_ERROR_TYPES: tuple[type[BaseException], ...] = (TimeoutError, ConnectionError, OSError)
#: OSError subclasses that indicate a permanent configuration problem
#: (missing/unreadable ``claude`` executable) rather than a transient
#: connection blip — must not be retried.
_NON_TRANSIENT_OS_TYPES: tuple[type[OSError], ...] = (FileNotFoundError, PermissionError, NotADirectoryError, IsADirectoryError)
#: HTTP status codes on ``ClaudeSdkRequestError.status`` treated as
#: transient provider failures — mirrors LLMEngine's "429/5xx" policy.
_TRANSIENT_HTTP_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}

#: The Claude Agent SDK's own message reader (``_internal/query.py``) raises
#: a bare, untyped ``Exception`` when the CLI subprocess crashes right after
#: emitting a result frame while a background tool task was still in flight
#: (see the SDK's own comment referencing upstream issue #1088) — it
#: replaces the resulting ``ProcessError`` with the last result's text,
#: producing messages like "Claude Code returned an error result: success"
#: even though that turn actually completed. There is no typed exception to
#: match on, so match the SDK-authored prefix instead. A fresh retry (no
#: ``resume``) reruns the same turn and reliably succeeds in practice.
_SDK_RESULT_RACE_PREFIX = "Claude Code returned an error result:"


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, ClaudeSdkRequestError):
        return exc.status in _TRANSIENT_HTTP_STATUS
    if isinstance(exc, _NON_TRANSIENT_OS_TYPES):
        return False
    if isinstance(exc, _TRANSIENT_ERROR_TYPES):
        return True
    return type(exc) is Exception and str(exc).startswith(_SDK_RESULT_RACE_PREFIX)


def _structured_output_instructions(output_type: Any) -> str | None:
    """Build a prompt block asking for JSON matching ``output_type``.

    ``output_type`` reaches ``run``/``stream`` on every call (it can change
    turn to turn, e.g. during ``Agent._validate_and_retry``'s correction
    loop) but there is no SDK/CLI knob here to *enforce* structured output
    server-side the way ``LLMEngine`` does via ``StructuredOutputConfig``.
    Without this, the model would just chat in prose, fail
    ``Agent._validate_and_retry``'s post-hoc JSON parse
    (``lazybridge.core.structured``), and burn output-retry turns on tasks
    that would otherwise succeed on the first attempt. Returns ``None`` for
    the default ``str``/``Any`` output type (nothing to add) or when the
    schema can't be derived (falls back to the existing post-hoc retry).
    """
    if output_type is str or output_type is Any:
        return None
    if not (isinstance(output_type, type) or get_origin(output_type) is not None):
        return None
    try:
        from pydantic import BaseModel, TypeAdapter

        if isinstance(output_type, type) and issubclass(output_type, BaseModel):
            schema = output_type.model_json_schema()
        else:
            schema = TypeAdapter(output_type).json_schema()
    except Exception:
        return None
    return (
        "Respond with valid JSON only — no prose, no markdown code fences — "
        "matching exactly this JSON schema:\n"
        f"{json.dumps(schema, indent=2)}"
    )


class ClaudeCodeEngine:
    """A standard LazyBridge Engine whose model loop is Claude Code.

    It deliberately accepts the same run parameters as ``LLMEngine`` and is
    stateless at the Claude SDK layer: LazyBridge ``Memory``
    remains the one conversation memory and is updated after a successful run.
    """

    def __init__(
        self,
        model: str = "sonnet",
        *,
        cwd: str | None = None,
        system: str | None = None,
        max_turns: int = 20,
        file_roots: list[str | Path] | None = None,
        web: bool = True,
        reasoning_effort: str | None = None,
        thinking: str | int | None = None,
        fallback_model: str | None = None,
        session_mode: str = "memory",
        session_name: str | None = None,
        request_timeout: float | None = 120.0,
        stream_idle_timeout: float | None = 90.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        tool_timeout: float | None = None,
        client: ClaudeSdkClient | None = None,
    ) -> None:
        self.model = model
        self.cwd = cwd
        self.system = system
        self.max_turns = max_turns
        if request_timeout is not None and request_timeout <= 0:
            raise ValueError(f"request_timeout must be > 0 or None, got {request_timeout!r}")
        self.request_timeout = request_timeout
        if stream_idle_timeout is not None and stream_idle_timeout <= 0:
            raise ValueError(f"stream_idle_timeout must be > 0 or None, got {stream_idle_timeout!r}")
        self.stream_idle_timeout = stream_idle_timeout
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries!r}")
        self.max_retries = max_retries
        if retry_delay <= 0:
            raise ValueError(f"retry_delay must be > 0, got {retry_delay!r}")
        self.retry_delay = retry_delay
        if tool_timeout is not None and tool_timeout <= 0:
            raise ValueError(f"tool_timeout must be > 0 or None, got {tool_timeout!r}")
        self.tool_timeout = tool_timeout
        roots = file_roots if file_roots is not None else ([cwd] if cwd else [])
        self.file_roots = tuple(str(Path(root).resolve()) for root in roots if root is not None)
        self.web = web
        if reasoning_effort not in {None, "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("reasoning_effort must be low, medium, high, xhigh, max, or None")
        self.reasoning_effort = reasoning_effort
        self.thinking = self._thinking_config(thinking)
        self.fallback_model = fallback_model
        if session_mode not in {"memory", "runtime"}:
            raise ValueError("session_mode must be 'memory' or 'runtime'")
        self.session_mode = session_mode
        self.session_name = session_name
        self._client = client or AgentSdkClient()

    @staticmethod
    def _thinking_config(value: str | int | None) -> dict[str, Any] | None:
        if value is None:
            return None
        if value in {"adaptive", "disabled"}:
            return {"type": value}
        if isinstance(value, int) and value > 0:
            return {"type": "enabled", "budget_tokens": value}
        raise ValueError("thinking must be 'adaptive', 'disabled', a positive token budget, or None")

    def _runtime_slot(self, agent_name: str) -> str:
        return f"claude-code:{self.session_name or agent_name}"

    def _resume_id(self, session: Any | None, agent_name: str) -> str | None:
        if self.session_mode != "runtime" or session is None:
            return None
        state = getattr(session, "_lazybridge_runtime_sessions", {})
        return state.get(self._runtime_slot(agent_name))

    def _remember_session(self, session: Any | None, agent_name: str, runtime_id: str | None) -> None:
        if self.session_mode != "runtime" or session is None or not runtime_id:
            return
        state = getattr(session, "_lazybridge_runtime_sessions", None)
        if state is None:
            state = {}
            session._lazybridge_runtime_sessions = state
        state[self._runtime_slot(agent_name)] = runtime_id

    def _prompt(self, env: Envelope[Any], memory: Any | None, output_type: type = str) -> str:
        if env.images or env.audio is not None:
            # No multimodal wiring yet: the Agent SDK prompt built here is
            # plain text (see AgentSdkClient._prompt_stream). Unlike
            # LLMEngine._build_user_content, which either forwards
            # images/audio or raises under strict_multimodal, attachments
            # here would otherwise vanish with no signal to the caller.
            warnings.warn(
                f"{type(self).__name__} does not forward Envelope.images/.audio to "
                "Claude Code yet — attachment(s) dropped from this run.",
                UserWarning,
                stacklevel=3,
            )
        parts: list[str] = []
        if memory is not None:
            history = self._memory_text(memory)
            if history:
                parts.append(f"Conversation context from LazyBridge:\n{history}")
        if env.context:
            parts.append(f"Additional context:\n{env.context}")
        parts.append(env.task or env.text())
        instructions = _structured_output_instructions(output_type)
        if instructions:
            parts.append(instructions)
        return "\n\n".join(part for part in parts if part)

    @staticmethod
    def _memory_text(memory: Any) -> str:
        if not hasattr(memory, "messages"):
            return str(memory.text())
        lines: list[str] = []
        for message in memory.messages():
            role = str(getattr(message, "role", "user")).replace("Role.", "").title()
            lines.append(f"{role}: {getattr(message, 'content', '')}")
        return "\n".join(lines)

    async def _idle_guarded_stream(self, agen: AsyncIterator[Any]) -> AsyncGenerator[Any, None]:
        """Yield items from ``agen``, raising on inter-chunk idle timeout.

        Mirrors ``LLMEngine._idle_guarded_stream``: catches a stalled SDK
        stream (no chunk for ``stream_idle_timeout`` seconds) instead of
        pinning the consumer forever.  A transparent passthrough when
        ``stream_idle_timeout`` is ``None``.
        """
        if self.stream_idle_timeout is None:
            async for item in agen:
                yield item
            return
        aiter = agen.__aiter__()
        while True:
            try:
                item = await asyncio.wait_for(aiter.__anext__(), timeout=self.stream_idle_timeout)
            except StopAsyncIteration:
                return
            except TimeoutError as exc:
                raise TimeoutError(
                    f"Claude Code stream went idle for {self.stream_idle_timeout}s without "
                    "delivering a chunk (set ClaudeCodeEngine(stream_idle_timeout=...) to a "
                    "higher value if streams legitimately pause that long)."
                ) from exc
            yield item

    async def _call_with_retries(self, make_call: Callable[[], Awaitable[_T]]) -> _T:
        """Run ``make_call()`` under ``request_timeout``, retrying transient failures.

        ``make_call`` is a thunk (not a bare coroutine) because a coroutine
        object can only be awaited once — each retry needs a fresh one.
        Mirrors ``LLMEngine``/``Executor``'s ``max_retries``/``retry_delay``
        with exponential backoff and jitter, at the coarse whole-call
        granularity available here (see ``_is_transient``).
        """
        attempt = 0
        while True:
            call = make_call()
            try:
                return await asyncio.wait_for(call, timeout=self.request_timeout) if self.request_timeout is not None else await call
            except BaseException as exc:
                if attempt >= self.max_retries or not _is_transient(exc):
                    raise
                delay = self.retry_delay * (2**attempt) * (0.9 + 0.2 * random.random())
                attempt += 1
                await asyncio.sleep(delay)

    def _options(self, tools: list[Any], observe: Any, *, partial: bool = False, resume: str | None = None) -> ClaudeSdkOptions:
        builtin_tools = (
            (("Read", "Glob", "Grep") if self.file_roots else ())
            + (("WebSearch", "WebFetch") if self.web else ())
        )
        return ClaudeSdkOptions(
            cwd=self.cwd,
            model=self.model,
            fallback_model=self.fallback_model,
            reasoning_effort=self.reasoning_effort,
            thinking=self.thinking,
            system_prompt=self.system,
            max_turns=self.max_turns,
            resume=resume,
            builtin_tools=builtin_tools,
            file_roots=self.file_roots,
            mcp_tools=to_mcp_tools(tools, observer=observe, tool_timeout=self.tool_timeout),
            include_partial_messages=partial,
        )

    async def run(
        self,
        env: Envelope[Any],
        *,
        tools: list[Any],
        output_type: type,
        memory: Any | None,
        session: Any | None,
        store: Any | None = None,
        plan_state: Any | None = None,
    ) -> Envelope[Any]:
        run_id = str(uuid.uuid4())
        started = time.monotonic()
        agent_name = resolve_agent_name(self, "agent")
        if session:
            session.emit(EventType.AGENT_START, {"agent_name": agent_name, "task": env.task}, run_id=run_id)
        try:
            def observe(kind: str, payload: dict[str, Any]) -> None:
                if not session:
                    return
                event_type = {
                    "call": EventType.TOOL_CALL,
                    "result": EventType.TOOL_RESULT,
                    "error": EventType.TOOL_ERROR,
                    "timeout": EventType.TOOL_TIMEOUT,
                }[kind]
                session.emit(event_type, payload, run_id=run_id)

            prompt = self._prompt(env, None if self.session_mode == "runtime" and session else memory, output_type)
            options = self._options(tools, observe, resume=self._resume_id(session, agent_name))
            result = await self._call_with_retries(lambda: self._client.run(prompt, options=options))
            self._remember_session(session, agent_name, result.session_id)
            if session:
                # Mirrors LLMEngine's MODEL_RESPONSE payload shape so
                # generic consumers (Session.usage_summary(), or any
                # external cost-report script that reads event_type=
                # "model_response" the way LLMEngine populates it) see
                # this engine's usage too, not just LLMEngine's.
                session.emit(
                    EventType.MODEL_RESPONSE,
                    {
                        "agent_name": agent_name,
                        "provider": "claude-code",
                        "model": result.model or self.model,
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                        "cost_usd": result.cost_usd,
                    },
                    run_id=run_id,
                )
            if memory is not None:
                memory.add(env.task or "", result.text, tokens=result.input_tokens + result.output_tokens)
            out = Envelope(
                task=env.task,
                payload=result.text,
                metadata=EnvelopeMetadata(
                    model=result.model or self.model,
                    provider="claude-code",
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    cost_usd=result.cost_usd,
                    latency_ms=(time.monotonic() - started) * 1000,
                    run_id=run_id,
                ),
            )
        except TimeoutError as exc:
            # asyncio.wait_for cancels the in-flight SDK call; a timeout is
            # a transient condition, unlike e.g. a schema/tool error, so
            # mark it retryable for callers that inspect ErrorInfo.
            out = Envelope.error_envelope(exc, retryable=True)
        except Exception as exc:
            out = Envelope.error_envelope(exc)
        if session:
            payload = {"agent_name": agent_name, "payload": out.text(), "latency_ms": (time.monotonic() - started) * 1000}
            if not out.ok:
                payload["error"] = out.error.message if out.error else "unknown"
            session.emit(EventType.AGENT_FINISH, payload, run_id=run_id)
        return out

    async def stream(
        self,
        env: Envelope[Any],
        *,
        tools: list[Any],
        output_type: type,
        memory: Any | None,
        session: Any | None,
    ) -> AsyncIterator[str]:
        run_id = str(uuid.uuid4())
        started = time.monotonic()
        agent_name = resolve_agent_name(self, "agent")
        if session:
            session.emit(EventType.AGENT_START, {"agent_name": agent_name, "task": env.task}, run_id=run_id)
        chunks: list[str] = []
        try:
            def observe(kind: str, payload: dict[str, Any]) -> None:
                if session:
                    session.emit(
                        {"call": EventType.TOOL_CALL, "result": EventType.TOOL_RESULT, "error": EventType.TOOL_ERROR, "timeout": EventType.TOOL_TIMEOUT}[kind],
                        payload,
                        run_id=run_id,
                    )

            input_tokens = output_tokens = 0
            cost_usd = 0.0
            async for event in self._idle_guarded_stream(
                self._client.stream(
                    self._prompt(env, None if self.session_mode == "runtime" and session else memory, output_type),
                    options=self._options(tools, observe, partial=True, resume=self._resume_id(session, agent_name)),
                )
            ):
                if event.text:
                    chunks.append(event.text)
                    yield event.text
                if event.final:
                    self._remember_session(session, agent_name, event.session_id)
                    input_tokens, output_tokens = event.input_tokens, event.output_tokens
                    cost_usd = event.cost_usd
            text = "".join(chunks)
            if session:
                session.emit(
                    EventType.MODEL_RESPONSE,
                    {
                        "agent_name": agent_name,
                        "provider": "claude-code",
                        "model": self.model,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cost_usd": cost_usd,
                    },
                    run_id=run_id,
                )
            if memory is not None:
                memory.add(env.task or "", text, tokens=input_tokens + output_tokens)
            if session:
                session.emit(
                    EventType.AGENT_FINISH,
                    {"agent_name": agent_name, "payload": text, "latency_ms": (time.monotonic() - started) * 1000},
                    run_id=run_id,
                )
        except Exception as exc:
            if session:
                session.emit(EventType.AGENT_FINISH, {"agent_name": agent_name, "error": str(exc)}, run_id=run_id)
            raise
