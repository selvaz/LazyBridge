"""Prototype ``Engine`` implementation backed by Claude Agent SDK."""

from __future__ import annotations

import asyncio
import contextlib
import random
import time
import uuid
import warnings
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any, ClassVar, TypeVar, get_origin

from lazybridge.engines.base import resolve_agent_name
from lazybridge.engines.coding import ApprovalGate, CodingAgentConfig, remembering_gate, session_approvals
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
_NON_TRANSIENT_OS_TYPES: tuple[type[OSError], ...] = (
    FileNotFoundError,
    PermissionError,
    NotADirectoryError,
    IsADirectoryError,
)
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


def _output_schema(output_type: Any) -> dict[str, Any] | None:
    """Derive the JSON schema for ``output_type``, or ``None``.

    ``output_type`` reaches ``run``/``stream`` on every call (it can change
    turn to turn, e.g. during ``Agent._validate_and_retry``'s correction
    loop). Returns ``None`` for the default ``str``/``Any`` output type
    (nothing to constrain) or when the schema can't be derived — in which
    case the run falls back to LazyBridge's post-hoc JSON parse and retry.
    """
    if output_type is str or output_type is Any:
        return None
    if not (isinstance(output_type, type) or get_origin(output_type) is not None):
        return None
    try:
        from pydantic import BaseModel, TypeAdapter

        if isinstance(output_type, type) and issubclass(output_type, BaseModel):
            return dict(output_type.model_json_schema())
        return dict(TypeAdapter(output_type).json_schema())
    except Exception:
        return None


def _output_format(output_type: Any) -> dict[str, Any] | None:
    """Native structured output for the Agent SDK, mirroring ``LLMEngine``.

    The SDK's ``output_format`` (``--json-schema`` on the CLI) constrains the
    final message server-side and returns the parsed object on
    ``ResultMessage.structured_output`` — the same guarantee ``LLMEngine``
    gets from ``StructuredOutputConfig``, and strictly better than asking for
    JSON in the prompt: no prose to strip, no wasted ``_validate_and_retry``
    turns. Verified live (claude_agent_sdk 0.2.128) to accept a plain Pydantic
    schema, optional fields and ``$defs`` included — no strict-mode rewrite
    needed, unlike Codex's ``turn/start`` ``outputSchema``.
    """
    schema = _output_schema(output_type)
    return {"type": "json_schema", "schema": schema} if schema is not None else None


class ClaudeCodeEngine:
    """A standard LazyBridge Engine whose model loop is Claude Code.

    It deliberately accepts the same run parameters as ``LLMEngine`` and is
    stateless at the Claude SDK layer: LazyBridge ``Memory``
    remains the one conversation memory and is updated after a successful run.

    **Durable sessions.** ``persist_session=True`` keeps the Claude Code
    session and exposes :attr:`session_id`; ``ClaudeCodeEngine(session_id=...)``
    resumes it — from a *later process*, since the SDK stores sessions on disk.
    That is the mirror of ``CodexEngine(persist_thread=...)``, and it is what
    makes a follow-up question cheap: Claude still has what it read.

    It differs from ``session_mode="runtime"``, which parks the id on a
    LazyBridge ``Session`` object and therefore never leaves the process. When
    an explicit ``session_id`` is set it wins over the parked one.

    Resuming moves where the conversation lives, so it also stops prepending
    ``Memory`` to the prompt (Claude has the history; sending it again states
    the same turns twice) and serialises runs per session id within the
    process — one session is one transcript.
    """

    #: One lock per durable session id, shared by every engine in the process:
    #: two engines resuming the same session are the same hazard as one engine
    #: doing it twice.
    _session_locks: ClassVar[dict[str, asyncio.Lock]] = {}

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
        session_id: str | None = None,
        persist_session: bool = False,
        request_timeout: float | None = 120.0,
        stream_idle_timeout: float | None = 90.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        tool_timeout: float | None = None,
        config: CodingAgentConfig | None = None,
        client: ClaudeSdkClient | None = None,
        tag: str | None = "lazybridge",
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
        # No explicit config preserves the pre-1.1 behaviour. Choose
        # ``CodingAgentConfig.reviewer()`` or ``.writer(gate)`` to opt into
        # the fail-closed profiles.
        self.config = config or CodingAgentConfig()
        # ``file_roots`` confinement is a hook over the FILE tools; names
        # outside that set (and the web pair) — ``Bash`` above all — have no
        # path sandbox, so their only boundary is the approval gate's policy.
        # Granting one without a gate would advertise a confined writer that
        # is not confined: refuse at construction, not at the first escape.
        _hook_confined = {"Read", "Glob", "Grep", "Edit", "Write", "NotebookEdit", "WebSearch", "WebFetch"}
        unconfined = tuple(t for t in self.config.claude.extra_tools if t not in _hook_confined)
        if unconfined and self.config.approval_gate is None:
            raise ValueError(
                f"extra_tools grants {unconfined} which file_roots cannot confine; "
                "configure CodingAgentConfig.approval_gate so a policy governs them "
                "(granting Bash without a gate would be an unconfined shell)"
            )
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
        #: Claude Code session to resume, and after each run the session the
        #: run used. Unlike ``session_mode="runtime"`` — which parks the id on
        #: a LazyBridge ``Session`` object and so only spans one process — this
        #: is a plain handle the caller keeps, so a *later process* can resume
        #: the same conversation (the SDK stores sessions on disk).
        self.session_id = session_id
        self.persist_session = persist_session or session_id is not None
        #: True once this engine is continuing an existing session, so the
        #: prompt stops carrying LazyBridge ``Memory``: Claude already has the
        #: history, and sending it again states past turns twice.
        self._resuming = session_id is not None
        #: Applied, once, to every NEW durable session this engine creates —
        #: never on resume, a tag already set stays set. Unlike Codex's
        #: ``thread_source`` (sent at creation, on the wire), the Agent SDK
        #: has no such field: this is instead appended post-hoc via the
        #: SDK's own ``tag_session()`` (a ``{"type":"tag",...}`` JSONL entry
        #: in the session file — the same mechanism ``list_sessions()``/the
        #: interactive CLI's session picker read). Defaults to
        #: ``"lazybridge"`` so every session this engine starts is
        #: identifiable later — ``list_sessions()`` has no server-side tag
        #: filter, so filter its returned list by ``.tag == "lazybridge"``,
        #: then ``delete_session(s.session_id)`` for a retention/cleanup
        #: pass. Pass ``None`` to skip tagging. Requires
        #: ``persist_session=True`` (or a ``session_id``) — an ephemeral
        #: session has nothing durable to tag.
        self.tag = tag
        #: Guards the run that *creates* the session, before there is an id to
        #: key the shared per-session lock on.
        self._own_lock = asyncio.Lock()
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

    def _session_lock(self) -> Any:
        """Serialise runs that continue the same Claude Code session.

        One session is one transcript: two turns appended at once interleave.
        Before the first durable run there is no id to key on, and that is not
        a free pass — two concurrent first runs would open two sessions and
        race to store their ids — so persistent engines fall back to a lock of
        their own until the id exists.
        """
        if not self.persist_session:
            return contextlib.nullcontext()
        if not self.session_id:
            return self._own_lock
        return type(self)._session_locks.setdefault(self.session_id, asyncio.Lock())

    def _resume_id(self, session: Any | None, agent_name: str) -> str | None:
        # An explicit handle wins over the Session-parked one: the caller
        # naming a session is a stronger statement than the ambient default.
        if self.session_id:
            return self.session_id
        if self.session_mode != "runtime" or session is None:
            return None
        state = getattr(session, "_lazybridge_runtime_sessions", {})
        return state.get(self._runtime_slot(agent_name))

    def _tag_new_session(self, session_id: str) -> None:
        """Best-effort ``tag_session()`` call for a session just created.

        Called only when this engine transitions from no-session to
        having one — never on every turn of an already-tagged session,
        since the SDK appends a JSONL line per call and "most recent tag
        wins" makes repeat calls pointless I/O, not idempotent no-ops.
        Failure is a warning, not a raised error: a tag is identification
        metadata, not something a run's success should depend on.
        """
        if self.tag is None:
            return
        try:
            from claude_agent_sdk import tag_session

            tag_session(session_id, self.tag, directory=self.cwd)
        except Exception as exc:  # defensive: tagging must never break a run
            warnings.warn(f"ClaudeCodeEngine: could not tag session {session_id!r}: {exc}", stacklevel=2)

    def _remember_session(self, session: Any | None, agent_name: str, runtime_id: str | None) -> None:
        if self.session_mode != "runtime" or session is None or not runtime_id:
            return
        state = getattr(session, "_lazybridge_runtime_sessions", None)
        if state is None:
            state = {}
            session._lazybridge_runtime_sessions = state
        state[self._runtime_slot(agent_name)] = runtime_id

    def _attachments(self, env: Envelope[Any]) -> tuple[dict[str, Any], ...]:
        """Convert ``Envelope.images`` into Anthropic image content blocks.

        Only inline bytes are forwarded: the CLI accepts a ``base64`` source
        (verified live) but rejects a ``url`` one, so URL-only images are
        dropped with a warning rather than silently costing a turn — pass a
        path or bytes and LazyBridge's ``_coerce_image`` inlines them.

        ``Envelope.audio`` is never forwarded: Claude accepts no audio input.
        """
        blocks: list[dict[str, Any]] = []
        for image in env.images or []:
            data = getattr(image, "base64_data", None)
            if data:
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": getattr(image, "media_type", None) or "image/png",
                            "data": data,
                        },
                    }
                )
            else:
                warnings.warn(
                    f"{type(self).__name__}: Claude Code accepts inline image bytes only — "
                    f"the URL image {getattr(image, 'url', None)!r} was dropped from this run. "
                    "Pass a local path or bytes instead (ImageContent.from_path).",
                    UserWarning,
                    stacklevel=4,
                )
        if env.audio is not None:
            warnings.warn(
                f"{type(self).__name__} does not forward Envelope.audio: Claude accepts no "
                "audio input — attachment dropped from this run.",
                UserWarning,
                stacklevel=4,
            )
        return tuple(blocks)

    def _prompt(self, env: Envelope[Any], memory: Any | None) -> str:
        parts: list[str] = []
        if memory is not None:
            history = self._memory_text(memory)
            if history:
                parts.append(f"Conversation context from LazyBridge:\n{history}")
        if env.context:
            parts.append(f"Additional context:\n{env.context}")
        parts.append(env.task or env.text())
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

        ``request_timeout`` bounds the *entire* retry loop, not each
        individual attempt — resetting the deadline per attempt would let
        ``max_retries`` stalled attempts plus backoff block for a multiple
        of the advertised timeout (e.g. ~4x with the defaults) instead of
        the deadline actually enforced.
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

    def _scoped_gate(self, session: Any, agent_name: str) -> ApprovalGate:
        """The configured gate, with ``allow_session`` scoped per agent+Session."""
        return remembering_gate(self.config.approval_gate, session_approvals(session, "claude-code", agent_name))

    async def usage(self, *, timeout: float | None = None) -> Any:
        """This engine's account usage, read from Claude Code's own ``/usage``.

        A convenience over :func:`~lazybridge.engines.claude_code.usage.fetch_claude_usage`
        that reuses this engine's ``model``/``cwd`` and — when this engine was
        built with an injected test client — that same client, so a
        ``ClaudeCodeEngine(client=FakeSdk())`` in a test never reaches the real
        SDK here either. Nothing else about the engine's configuration (tools,
        approval gate, policy) applies, since ``/usage`` is a CLI meta command,
        not a turn this agent's tools or approvals could touch. See that
        function for what can fail and why the SDK's own rate-limit event is
        not used instead.
        """
        from lazybridge.engines.claude_code.usage import fetch_claude_usage

        kwargs: dict[str, Any] = {"model": self.model, "cwd": self.cwd, "client": self._client}
        if timeout is not None:
            kwargs["timeout"] = timeout
        return await fetch_claude_usage(**kwargs)

    def _options(
        self,
        tools: list[Any],
        observe: Any,
        *,
        output_type: type = str,
        partial: bool = False,
        resume: str | None = None,
        gate: ApprovalGate | None = None,
    ) -> ClaudeSdkOptions:
        builtin_tools = (("Read", "Glob", "Grep") if self.file_roots else ()) + (
            ("WebSearch", "WebFetch") if self.web else ()
        )
        # Policy-granted additions (e.g. Write/Edit/Bash for a gated writer
        # agent), deduplicated while preserving order.
        for name in self.config.claude.extra_tools:
            if name not in builtin_tools:
                builtin_tools += (name,)
        return ClaudeSdkOptions(
            cwd=self.cwd,
            model=self.model,
            fallback_model=self.fallback_model,
            reasoning_effort=self.reasoning_effort,
            thinking=self.thinking,
            system_prompt=self.system,
            max_turns=self.max_turns,
            resume=resume,
            allowed_tools=self.config.claude.allowed_tools,
            preapprove_application_tools=self.config.claude.preapprove_application_tools,
            disallowed_tools=self.config.claude.disallowed_tools,
            setting_sources=self.config.claude.setting_sources,
            auto_compact_window=self.config.claude.auto_compact_window,
            permission_mode=self.config.claude.permission_mode,
            approval_gate=gate if gate is not None else self.config.approval_gate,
            builtin_tools=builtin_tools,
            file_roots=self.file_roots,
            mcp_tools=to_mcp_tools(tools, observer=observe, tool_timeout=self.tool_timeout),
            include_partial_messages=partial,
            output_format=_output_format(output_type),
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

            attachments = self._attachments(env)
            async with self._session_lock():
                # Prompt and options are built INSIDE the lock: both read
                # ``session_id``, and reading it before waiting would let two
                # concurrent first runs each see "no session yet" and open one.
                carries_history = (self.session_mode == "runtime" and session) or self._resuming
                prompt = self._prompt(env, None if carries_history else memory)
                options = self._options(
                    tools,
                    observe,
                    output_type=output_type,
                    resume=self._resume_id(session, agent_name),
                    gate=self._scoped_gate(session, agent_name) if self.config.approval_gate else None,
                )
                result = await self._call_with_retries(
                    lambda: self._client.run(prompt, options=options, attachments=attachments)
                )
                if self.persist_session and result.session_id:
                    # Inside the lock too: the next queued run must see it.
                    # The SDK can return a NEW id for a resumed session, so
                    # the engine follows the chain rather than pinning the
                    # original — pinning would replay an ever-older session.
                    if not self._resuming:
                        self._tag_new_session(result.session_id)
                    self.session_id = result.session_id
                    self._resuming = True
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
            payload = {
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
                        {
                            "call": EventType.TOOL_CALL,
                            "result": EventType.TOOL_RESULT,
                            "error": EventType.TOOL_ERROR,
                            "timeout": EventType.TOOL_TIMEOUT,
                        }[kind],
                        payload,
                        run_id=run_id,
                    )

            input_tokens = output_tokens = 0
            cost_usd = 0.0
            async for event in self._idle_guarded_stream(
                self._client.stream(
                    self._prompt(
                        env,
                        None if ((self.session_mode == "runtime" and session) or self._resuming) else memory,
                    ),
                    options=self._options(
                        tools,
                        observe,
                        output_type=output_type,
                        partial=True,
                        resume=self._resume_id(session, agent_name),
                        gate=self._scoped_gate(session, agent_name) if self.config.approval_gate else None,
                    ),
                    attachments=self._attachments(env),
                )
            ):
                if event.text:
                    chunks.append(event.text)
                    yield event.text
                if event.final:
                    if self.persist_session and event.session_id:
                        if not self._resuming:
                            self._tag_new_session(event.session_id)
                        self.session_id = event.session_id
                        self._resuming = True
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
