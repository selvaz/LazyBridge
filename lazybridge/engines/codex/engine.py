"""``CodexEngine`` — a LazyBridge engine backed by a local Codex CLI."""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, TypeVar, cast, get_origin

from lazybridge.engines.base import resolve_agent_name
from lazybridge.engines.coding import (
    ApprovalGate,
    ApprovalRequest,
    CodingAgentConfig,
    ask_approval,
    remembering_gate,
    session_approvals,
)
from lazybridge.envelope import Envelope, EnvelopeMetadata
from lazybridge.session import EventType

from .app_server import CodexAppServerClient
from .dynamic_tools import definitions, dispatcher

_T = TypeVar("_T")

#: Exception types treated as transient and worth retrying — connection /
#: process-level failures, not logical errors. Mirrors ``LLMEngine``'s
#: "429/5xx/network/timeout" retry policy at the granularity available to a
#: subprocess-backed engine: pass/fail on the whole App Server call, not
#: individual JSON-RPC round-trips.
_TRANSIENT_ERROR_TYPES: tuple[type[BaseException], ...] = (TimeoutError, ConnectionError, OSError)
#: OSError subclasses that indicate a permanent configuration problem
#: (missing/unreadable ``codex`` executable) rather than a transient
#: connection blip — must not be retried.
_NON_TRANSIENT_OS_TYPES: tuple[type[OSError], ...] = (
    FileNotFoundError,
    PermissionError,
    NotADirectoryError,
    IsADirectoryError,
)


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, _NON_TRANSIENT_OS_TYPES):
        return False
    return isinstance(exc, _TRANSIENT_ERROR_TYPES)


def _structured_output_instructions(output_type: Any) -> str | None:
    """Build a prompt block asking for JSON matching ``output_type``.

    Unlike ``ClaudeCodeEngine``, which constrains the answer server-side via
    the Agent SDK's ``output_format``, this engine primes the prompt. Codex's
    ``turn/start`` *does* expose a native ``outputSchema``, but it accepts
    only OpenAI-strict schemas — ``additionalProperties: false`` on every
    object **and** ``required`` listing every property — which a plain
    Pydantic schema does not satisfy (verified live: the turn fails with
    ``invalid_json_schema``). Until a strict-mode rewrite exists, asking in
    the prompt keeps arbitrary ``output=`` types working and merely falls
    back to ``Agent._validate_and_retry``'s post-hoc repair when the model
    strays.

    Returns ``None`` for the default ``str``/``Any`` output type, or when the
    schema can't be derived.
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


class CodexEngine:
    """A standard LazyBridge ``Engine`` whose model loop is Codex.

    It talks to ``codex app-server`` over JSON-RPC, starting one ephemeral,
    read-only, approval-free thread per run, and reuses the local Codex
    login: no API key is read, copied, or stored. LazyBridge keeps owning
    memory and tools — the engine exposes the current tool list as App Server
    *dynamic tools* and dispatches every call back to ``Tool.run()``.

    See :doc:`/guides/full/codex-engine` for setup and the verified/unwired
    protocol surface.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        cwd: str | None = None,
        system: str | None = None,
        reasoning_effort: str | None = None,
        request_timeout: float | None = 120.0,
        stream_idle_timeout: float | None = 90.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        tool_timeout: float | None = None,
        config: CodingAgentConfig | None = None,
        client: CodexAppServerClient | None = None,
    ) -> None:
        self.model, self.cwd, self.system = model, cwd, system
        #: Free-form per-model effort string (``"low"``/``"medium"``/``"high"``
        #: on current models); the App Server advertises each model's accepted
        #: values through ``model/list``, so it is passed through unvalidated.
        self.reasoning_effort = reasoning_effort
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
        self.config = config or CodingAgentConfig()
        self._client = client or CodexAppServerClient()

    def _scoped_gate(self, session: Any, agent_name: str) -> ApprovalGate:
        """The configured gate, with ``allow_session`` scoped per agent+Session."""
        return remembering_gate(self.config.approval_gate, session_approvals(session, "codex", agent_name))

    def _tool_dispatcher(
        self, tools: list[Any], observe: Any, gate: ApprovalGate
    ) -> Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]:
        dispatch = dispatcher(tools, observe, tool_timeout=self.tool_timeout)
        if self.config.codex.preapprove_dynamic_tools:
            return dispatch

        async def gated(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
            # ``gate`` already answers from the agent's session cache when the
            # tool was approved with ``allow_session`` on an earlier run.
            decision = await ask_approval(
                gate,
                ApprovalRequest(
                    provider="codex",
                    kind="tool",
                    name=tool,
                    arguments=arguments,
                    cwd=self.cwd,
                ),
            )
            if decision.action not in {"allow", "allow_session"}:
                return {
                    "success": False,
                    "contentItems": [
                        {
                            "type": "inputText",
                            "text": decision.message or "Tool call denied by approval gate",
                        }
                    ],
                }
            return cast("dict[str, Any]", await dispatch(tool, arguments))

        return gated

    async def _call_with_retries(self, make_call: Callable[[], Awaitable[_T]]) -> _T:
        """Run ``make_call()`` under ``request_timeout``, retrying transient failures.

        ``make_call`` is a thunk, not a bare coroutine, because each retry
        needs a fresh coroutine (and, here, a fresh ``codex app-server``
        subprocess — ``CodexAppServerClient.run`` spawns/tears one down per
        call, so a retry is naturally a clean restart).

        ``request_timeout`` bounds the *entire* retry loop, not each
        individual attempt — resetting the deadline per attempt would let
        ``max_retries`` stalled attempts plus backoff block for a multiple of
        the advertised timeout (e.g. ~4x with the defaults) instead of the
        deadline actually enforced.
        """

        async def _attempt_loop() -> _T:
            attempt = 0
            while True:
                try:
                    return await make_call()
                except Exception as exc:
                    if attempt >= self.max_retries or not _is_transient(exc):
                        raise
                    delay = self.retry_delay * (2**attempt) * (0.9 + 0.2 * random.random())
                    attempt += 1
                    await asyncio.sleep(delay)

        if self.request_timeout is None:
            return await _attempt_loop()
        return await asyncio.wait_for(_attempt_loop(), timeout=self.request_timeout)

    def _attachments(self, env: Envelope[Any]) -> list[dict[str, Any]]:
        """Convert ``Envelope.images`` into App Server ``UserInput`` items.

        ``ImageContent`` carries either a ``url`` or ``base64_data``; the App
        Server's ``image`` variant takes a URL, and a ``data:`` URL is
        accepted for inline bytes (both verified live).

        ``Envelope.audio`` is *not* forwarded: the protocol has ``audio`` /
        ``localAudio`` variants and accepts them without error, but the model
        then reports it cannot access the attachment (verified live with
        both), so sending it would cost tokens for nothing.
        """
        items: list[dict[str, Any]] = []
        for image in env.images or []:
            url = getattr(image, "url", None)
            data = getattr(image, "base64_data", None)
            if url:
                items.append({"type": "image", "url": url})
            elif data:
                media_type = getattr(image, "media_type", None) or "image/png"
                items.append({"type": "image", "url": f"data:{media_type};base64,{data}"})
            else:
                warnings.warn(
                    f"{type(self).__name__}: image attachment carries neither url nor "
                    "base64_data — dropped from this run.",
                    UserWarning,
                    stacklevel=4,
                )
        if env.audio is not None:
            warnings.warn(
                f"{type(self).__name__} does not forward Envelope.audio to Codex: the App "
                "Server accepts audio input but the model cannot access it (verified live) "
                "— attachment dropped from this run.",
                UserWarning,
                stacklevel=4,
            )
        return items

    def _prompt(self, env: Envelope[Any], memory: Any | None, output_type: type = str) -> str:
        history = str(memory.text()) if memory is not None else ""
        parts = [
            f"LazyBridge conversation context:\n{history}" if history else "",
            env.context,
            env.task or env.text(),
            _structured_output_instructions(output_type) or "",
        ]
        return "\n\n".join(part for part in parts if part)

    def _emit_model_response(self, session: Any, agent_name: str, result: Any, run_id: str) -> None:
        # Mirrors LLMEngine's MODEL_RESPONSE payload shape so generic
        # consumers (Session.usage_summary(), or any external cost-report
        # script reading event_type="model_response") see this engine's usage
        # too. ``cost_usd`` is structurally 0.0 here — see CodexRunResult.
        session.emit(
            EventType.MODEL_RESPONSE,
            {
                "agent_name": agent_name,
                "provider": "codex",
                "model": self.model,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cost_usd": result.cost_usd,
            },
            run_id=run_id,
        )

    def _observer(self, session: Any, run_id: str) -> Callable[[str, dict[str, Any]], None]:
        def observe(kind: str, payload: dict[str, Any]) -> None:
            if session:
                session.emit(
                    {
                        "call": EventType.TOOL_CALL,
                        "result": EventType.TOOL_RESULT,
                        "error": EventType.TOOL_ERROR,
                        "timeout": EventType.TOOL_TIMEOUT,
                    }[kind],
                    payload,
                    run_id=run_id,
                )

        return observe

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
        run_id, started = str(uuid.uuid4()), time.monotonic()
        agent_name = resolve_agent_name(self, "agent")
        if session:
            session.emit(EventType.AGENT_START, {"agent_name": agent_name, "task": env.task}, run_id=run_id)
        out: Envelope[Any]
        try:
            observe = self._observer(session, run_id)
            gate = self._scoped_gate(session, agent_name)
            attachments = self._attachments(env)
            result = await self._call_with_retries(
                lambda: self._client.run(
                    prompt=self._prompt(env, memory, output_type),
                    model=self.model,
                    cwd=self.cwd,
                    dynamic_tools=definitions(tools),
                    on_tool_call=self._tool_dispatcher(tools, observe, gate),
                    attachments=attachments,
                    effort=self.reasoning_effort,
                    developer_instructions=self.system,
                    sandbox=self.config.codex.sandbox,
                    approval_policy=self.config.codex.approval_policy,
                    approval_gate=gate,
                )
            )
            if session:
                self._emit_model_response(session, agent_name, result, run_id)
            if memory is not None:
                memory.add(env.task or "", result.text, tokens=result.input_tokens + result.output_tokens)
            out = Envelope(
                task=env.task,
                payload=result.text,
                metadata=EnvelopeMetadata(
                    model=self.model,
                    provider="codex",
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    cost_usd=result.cost_usd,
                    latency_ms=(time.monotonic() - started) * 1000,
                    run_id=run_id,
                ),
            )
        except TimeoutError as exc:
            # asyncio.wait_for cancels the in-flight client.run() call, and
            # CodexAppServerClient's own finally block terminates the
            # subprocess. A timeout is transient, unlike a tool/schema error,
            # so mark it retryable.
            out = Envelope.error_envelope(exc, retryable=True)
        except Exception as exc:
            out = Envelope.error_envelope(exc)
        if session:
            payload: dict[str, Any] = {
                "agent_name": agent_name,
                "payload": out.text(),
                "latency_ms": (time.monotonic() - started) * 1000,
            }
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
        """Stream text deltas from the Codex App Server run.

        Bridges the App Server's ``item/agentMessage/delta`` push events
        (synchronously awaited in ``CodexAppServerClient.run``) into a bounded
        ``asyncio.Queue`` so a slow consumer applies backpressure all the way
        to the reader, matching ``LLMEngine.stream`` / ``ClaudeCodeEngine.stream``.
        """
        run_id, started = str(uuid.uuid4()), time.monotonic()
        agent_name = resolve_agent_name(self, "agent")
        if session:
            session.emit(EventType.AGENT_START, {"agent_name": agent_name, "task": env.task}, run_id=run_id)

        sink: asyncio.Queue[str | None] = asyncio.Queue(maxsize=64)

        async def on_text(chunk: str) -> None:
            await sink.put(chunk)

        async def _run_loop() -> None:
            try:
                observe = self._observer(session, run_id)
                gate = self._scoped_gate(session, agent_name)
                result = await self._client.run(
                    prompt=self._prompt(env, memory, output_type),
                    model=self.model,
                    cwd=self.cwd,
                    dynamic_tools=definitions(tools),
                    on_tool_call=self._tool_dispatcher(tools, observe, gate),
                    on_text=on_text,
                    attachments=self._attachments(env),
                    effort=self.reasoning_effort,
                    developer_instructions=self.system,
                    sandbox=self.config.codex.sandbox,
                    approval_policy=self.config.codex.approval_policy,
                    approval_gate=gate,
                )
                if session:
                    self._emit_model_response(session, agent_name, result, run_id)
                if memory is not None:
                    memory.add(env.task or "", result.text, tokens=result.input_tokens + result.output_tokens)
            except asyncio.CancelledError:
                raise
            except BaseException:
                await sink.put(None)  # wake the consumer so the error surfaces
                raise
            else:
                await sink.put(None)  # sentinel — loop done

        task = asyncio.create_task(_run_loop())
        cancelled_by_us = False
        error: BaseException | None = None
        try:
            while True:
                get_call = sink.get()
                token = (
                    await asyncio.wait_for(get_call, timeout=self.stream_idle_timeout)
                    if self.stream_idle_timeout is not None
                    else await get_call
                )
                if token is None:
                    break
                yield token
        except TimeoutError as exc:
            error = TimeoutError(
                f"Codex stream went idle for {self.stream_idle_timeout}s without delivering a "
                "chunk (set CodexEngine(stream_idle_timeout=...) to a higher value if streams "
                "legitimately pause that long)."
            )
            error.__cause__ = exc
        finally:
            if not task.done():
                task.cancel()
                cancelled_by_us = True
            try:
                await task
            except asyncio.CancelledError:
                if not cancelled_by_us:
                    raise
            except Exception as exc:
                if error is None:
                    error = exc
            if session:
                finish_payload: dict[str, Any] = {
                    "agent_name": agent_name,
                    "cancelled": cancelled_by_us,
                    "latency_ms": (time.monotonic() - started) * 1000,
                }
                if error is not None:
                    finish_payload["error"] = str(error)
                session.emit(EventType.AGENT_FINISH, finish_payload, run_id=run_id)
            if error is not None:
                raise error
