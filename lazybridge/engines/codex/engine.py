"""``CodexEngine`` — a LazyBridge engine backed by a local Codex CLI."""

from __future__ import annotations

import asyncio
import contextlib
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
    loop_scoped_lock,
    remembering_gate,
    session_approvals,
)
from lazybridge.envelope import Envelope, EnvelopeMetadata
from lazybridge.session import EventType

from .app_server import CodexAppServerClient, CodexTurnUncertain
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


def _ran(progress: dict[str, Any]) -> bool:
    """Did a turn actually reach the model on this call?

    ``turn_sent`` alone is not that fact: a request can be transmitted and
    then *rejected*, which leaves the thread empty. Anything else after
    transmission may have run, so it counts as history.
    """
    return bool(progress.get("turn_sent")) and not progress.get("rejected")


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

    **Durable threads.** Pass ``persist_thread=True`` and the thread survives
    the subprocess; :attr:`thread_id` then holds a handle that a *later
    process* can resume (``CodexEngine(thread_id=...)``), so Codex keeps its
    own transcript — the files it read, the reasoning it did — instead of
    being re-primed from scratch on every call. That is the mode that makes a
    follow-up question cheap.

    It also moves where the conversation lives, so the engine changes three
    things when resuming:

    * **one memory authority** — LazyBridge ``Memory`` is no longer prepended
      to the prompt (Codex has the history); Memory keeps recording for the
      application's own audit/recovery use;
    * **no blind retry** — a turn that fails *after* the server accepted it is
      raised as ``CodexTurnUncertain`` rather than replayed, because a durable
      turn may already be committed, tool side effects included;
    * **serialised per thread** — concurrent runs against one thread id are
      queued in-process. Nothing can stop a *different* process resuming the
      same thread concurrently; don't share a thread id across processes.

    **Native review.** ``review_target={"type": "baseBranch", "branch": "main"}``
    (or ``{"type": "uncommittedChanges"}`` / ``{"type": "commit", "sha": ...}``)
    runs Codex' own review harness instead of a prompted turn: it returns
    severity-tagged findings with file:line, and the agent's prompt is **not
    sent** — the protocol has no slot for one, so the review cannot be steered.
    Combine it with ``persist_thread=True`` and the review lands in the thread,
    so an ordinary follow-up turn can then ask about it.

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
        thread_id: str | None = None,
        persist_thread: bool = False,
        review_target: dict[str, Any] | None = None,
        thread_source: str | None = "lazybridge",
    ) -> None:
        self.model, self.cwd, self.system = model, cwd, system
        #: Thread to resume, and after each run the thread the run used —
        #: durable only when ``persist_thread`` (resuming implies it).
        self.thread_id = thread_id
        self.persist_thread = persist_thread or thread_id is not None
        #: True while a run is resuming an existing thread, as opposed to
        #: creating the first one: only then does Codex already hold history.
        self._resuming = thread_id is not None
        #: When set, every run of this engine is a native ``review/start``
        #: against this target instead of a prompted turn — see ``run()``.
        self.review_target = review_target
        #: Sent as ``ThreadStartParams.threadSource`` on every NEW thread this
        #: engine creates (never on resume — see ``app_server.py``). Lands on
        #: disk as ``session_meta.payload.thread_source`` (verified live) —
        #: NOT the same field as ``session_meta.payload.source``, which is
        #: something else and unaffected by this. Every LazyBridge thread is
        #: already identifiable without this, via ``session_meta.payload
        #: .originator == "lazybridge"`` (unconditional, from the
        #: ``initialize`` call — see ``app_server.py``); this field adds a
        #: second, caller-chosen label on top, e.g. to tell two different
        #: LazyBridge-based applications' threads apart. Pass ``None`` to
        #: omit it.
        self.thread_source = thread_source
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

    def _client_kwargs(
        self,
        env: Envelope[Any],
        tools: list[Any],
        output_type: type,
        memory: Any | None,
        observe: Any,
        gate: ApprovalGate,
        attachments: list[dict[str, Any]],
        progress: dict[str, Any],
        on_text: Callable[[str], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """The one description of a client call, shared by ``run``/``stream``.

        They had a verbatim copy each, and that duplicate is exactly where the
        durable-thread handling drifted apart — the streaming path silently
        missed the handle write-back and the timeout classification.
        """
        kwargs: dict[str, Any] = {
            "prompt": self._prompt(env, memory, output_type),
            "model": self.model,
            "cwd": self.cwd,
            "dynamic_tools": definitions(tools),
            "on_tool_call": self._tool_dispatcher(tools, observe, gate),
            "attachments": attachments,
            "effort": self.reasoning_effort,
            "developer_instructions": self.system,
            "sandbox": self.config.codex.sandbox,
            "approval_policy": self.config.codex.approval_policy,
            "approval_gate": gate,
            "thread_id": self.thread_id,
            "ephemeral": not self.persist_thread,
            "review_target": self.review_target,
            "progress": progress,
            "thread_source": self.thread_source,
        }
        if on_text is not None:
            kwargs["on_text"] = on_text
        return kwargs

    def _absorb(self, thread_id: str | None, *, has_history: bool = True) -> None:
        """Keep a durable thread's handle, whatever the run's outcome.

        Called on success *and* on every failure path, cancellation included:
        an interrupted turn is precisely the case where the id matters, since
        resuming and inspecting the thread is the documented recovery.

        ``has_history`` is separate because the two facts are separate. A
        thread whose first turn was *rejected* exists but is empty: keeping its
        id is right, while marking it as carrying history is not — the next
        call would then withhold Memory from a model that has never seen any.
        """
        if self.persist_thread and thread_id:
            self.thread_id = thread_id
            if has_history:
                self._resuming = True

    def _durable_timeout(self, progress: dict[str, Any], exc: BaseException) -> BaseException:
        """Classify a timeout: retryable, or an accepted turn of unknown fate.

        ``asyncio.wait_for`` cancels the client with ``CancelledError``, a
        BaseException that unwinds past the client's own uncertainty handling —
        so this is the last place that can tell the two apart, and it does it
        from what the client managed to report before being cut off.
        """
        if not (self.persist_thread and progress.get("turn_sent")):
            return exc  # nothing was sent, or nothing durable: a clean retry
        thread = progress.get("thread_id") or self.thread_id or ""
        return CodexTurnUncertain(
            f"Turn timed out on durable thread {thread or '(unknown)'}; whether it was "
            f"committed is unknown. Resume and inspect it before retrying.",
            thread_id=thread,
            turn_id=None,
        )

    @contextlib.asynccontextmanager
    async def _thread_lock(self) -> Any:
        """Serialise runs that target the same durable thread.

        A thread is a single transcript: two turns appended to it at once
        interleave, and the App Server may reject the second outright. Fresh
        ephemeral threads share nothing, so they stay fully parallel.

        A persistent engine takes **both** locks, always in this order: its own
        (so its runs queue even before a thread id exists — two concurrent
        first runs would otherwise open two threads and race to store the id)
        and then the id's (so a *different* engine resuming the same thread
        queues too). Taking only the second would leave a hole: a run that
        starts after the id is stored keys straight onto an uncontended shared
        lock while the run that created it is still going.

        Both are scoped to the running event loop — see ``loop_scoped_lock``.
        """
        if not self.persist_thread:
            yield
            return
        async with loop_scoped_lock(f"codex-engine:{id(self)}"):
            if self.thread_id:
                async with loop_scoped_lock(f"codex-thread:{self.thread_id}"):
                    yield
            else:
                yield

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
        # On a resumed thread Codex' own transcript already holds the history:
        # prepending LazyBridge's Memory would re-state past user and assistant
        # turns as if they were new, giving the model two chronologies of the
        # same conversation. One authority per thread — see the class docstring.
        history = str(memory.text()) if memory is not None and not self._resuming else ""
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
        progress: dict[str, Any] = {}
        try:
            observe = self._observer(session, run_id)
            gate = self._scoped_gate(session, agent_name)
            attachments = self._attachments(env)
            async with self._thread_lock():
                call = lambda: self._client.run(  # noqa: E731 - a thunk, one per retry
                    **self._client_kwargs(env, tools, output_type, memory, observe, gate, attachments, progress)
                )
                result = await self._call_with_retries(call)
                self._absorb(result.thread_id)
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
            # subprocess. Whether that is a clean retry or an accepted turn of
            # unknown fate depends on how far the call got — which is what
            # ``progress`` records, since the cancellation unwinds past the
            # client's own handling.
            failure = self._durable_timeout(progress, exc)
            out = Envelope.error_envelope(failure, retryable=failure is exc)
        except CodexTurnUncertain as exc:
            # Keep the handle even though the run failed: inspecting the thread
            # IS the documented recovery path, and on a first durable run this
            # is the only place the id is ever seen.
            self._absorb(exc.thread_id or progress.get("thread_id"))
            out = Envelope.error_envelope(exc)
        except Exception as exc:
            out = Envelope.error_envelope(exc)
        finally:
            # Every exit, including an *external* cancellation — which is a
            # BaseException and reaches none of the handlers above, yet can
            # still leave a durable thread behind.
            self._absorb(progress.get("thread_id"), has_history=_ran(progress))
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
        streamed = False

        async def on_text(chunk: str) -> None:
            nonlocal streamed
            streamed = True
            await sink.put(chunk)

        progress: dict[str, Any] = {}

        async def _run_loop() -> None:
            try:
                observe = self._observer(session, run_id)
                gate = self._scoped_gate(session, agent_name)
                # Same serialisation as run(): a streamed turn appends to the
                # same transcript, so it cannot be exempt from the per-thread
                # lock without letting two turns interleave.
                async with self._thread_lock():
                    result = await self._client.run(
                        **self._client_kwargs(
                            env,
                            tools,
                            output_type,
                            memory,
                            observe,
                            gate,
                            self._attachments(env),
                            progress,
                            on_text=on_text,
                        )
                    )
                    self._absorb(result.thread_id)
                if not streamed and result.text:
                    # A native review streams no deltas at all (measured), so a
                    # streaming caller would otherwise get an empty result while
                    # the findings sat in ``result.text``. Deliver them as one
                    # chunk rather than nothing.
                    await sink.put(result.text)
                if session:
                    self._emit_model_response(session, agent_name, result, run_id)
                if memory is not None:
                    memory.add(env.task or "", result.text, tokens=result.input_tokens + result.output_tokens)
            except asyncio.CancelledError:
                raise
            except BaseException:
                # Deliberately broader than ``Exception``: whatever ends this
                # task — including KeyboardInterrupt/SystemExit — the consumer
                # is parked on ``sink.get()`` and would wait out the whole
                # stream_idle_timeout before noticing. The sentinel wakes it so
                # the error surfaces immediately, and the exception is
                # re-raised untouched on the next line.
                await sink.put(None)
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
            idle = TimeoutError(
                f"Codex stream went idle for {self.stream_idle_timeout}s without delivering a "
                "chunk (set CodexEngine(stream_idle_timeout=...) to a higher value if streams "
                "legitimately pause that long)."
            )
            idle.__cause__ = exc
            # Same classification as run(): cancelling the loop below cancels a
            # turn that may already have been accepted on a durable thread.
            error = self._durable_timeout(progress, idle)
        finally:
            if not task.done():
                task.cancel()
                cancelled_by_us = True
            try:
                await task
            except asyncio.CancelledError:
                if not cancelled_by_us:
                    raise
            except CodexTurnUncertain as exc:
                # The handle is the recovery path; a streamed run must not lose
                # it just because it failed on the other side of a queue.
                self._absorb(exc.thread_id or progress.get("thread_id"))
                if error is None:
                    error = exc
            except Exception as exc:
                if error is None:
                    error = exc
            # Also covers the consumer walking away mid-stream: the generator
            # is closed, the worker cancelled, and the turn it started may
            # already be committed to a durable thread.
            self._absorb(progress.get("thread_id"), has_history=_ran(progress))
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
