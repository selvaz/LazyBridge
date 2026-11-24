"""Small JSON-RPC client for one ephemeral Codex App Server run.

The wire format here was verified against a live ``codex app-server``
(codex-cli 0.148.0); the authoritative schema comes from
``codex app-server generate-json-schema --out <dir>``.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import logging
import os
import random
import re
import shutil
import subprocess
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from lazybridge._asyncbridge import abandoned_tasks
from lazybridge.engines.coding import ApprovalGate, ApprovalRequest, ask_approval
from lazybridge.envelope import UncertainOutcomeError
from lazybridge.policy import PolicyDenied, current_policy_context

from .policy_launch import PolicyLaunch

ToolCallback = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]

#: Max size of ONE JSON-RPC line from the App Server. ``StreamReader``'s
#: default is 64 KiB, and a line over that makes ``readline()`` raise
#: ``ValueError: Separator is found, but chunk is longer than limit`` — which
#: surfaced live the first time a turn read a large file or a real ``git
#: diff``: the App Server puts whole command outputs and file contents in a
#: single ``item/...`` notification, so 64 KiB is routine, not exotic. 64 MiB
#: is the ceiling on one *message*, not on the turn, and is only ever
#: allocated for a line that actually arrives.
_STDOUT_LINE_LIMIT = 64 * 1024 * 1024

#: Newest stderr bytes kept for diagnostics. The pipe is always drained, so a
#: chatty child can neither block nor grow this buffer past the cap.
_STDERR_TAIL_BYTES = 16 * 1024
#: How much of that tail is quoted in the exception message.
_STDERR_MESSAGE_CHARS = 2000
#: Bound on each wait during teardown and on the return-code lookup after EOF.
_EXIT_GRACE = 2.0
#: How long teardown waits for cancelled tool/approval callbacks to unwind.
#: A callback that ignores cancellation or cleans up slowly is abandoned (with
#: a warning) rather than allowed to hold a finished turn hostage.
_CLIENT_TASK_TEARDOWN_GRACE = 3.0
#: Short wait for a clean exit after stdin closes (the real child needs ~1s, so
#: this usually falls through to terminate, as before) — kept small so teardown
#: stays fast.
_STDIN_EOF_GRACE = 0.25
#: Phases before any stateful request (thread/start, thread/resume) was written:
#: only a failure here is safe to retry, since a durable thread may already exist
#: after that point.
_PRE_THREAD_PHASES = frozenset({"spawn", "initialize", "initialized"})
#: Phases the client itself retries once when the child dies before the turn
#: was written. thread/resume is idempotent; a thread/start that dies before
#: answering never handed us the id, so its (empty) thread is orphaned whether
#: or not we retry. Live 10/10/2026: the weekly died with 0xC0000409 in
#: thread/start. The engine-level retry keeps the narrower _PRE_THREAD_PHASES.
_RETRYABLE_BEFORE_TURN = _PRE_THREAD_PHASES | {"thread/start", "thread/resume"}
#: Jittered pause (seconds) before the single startup retry.
_STARTUP_RETRY_DELAY = (0.2, 0.6)

_logger = logging.getLogger(__name__)
#: Strong references to abandoned callback tasks until they finish.
_abandoned_callbacks: set[asyncio.Task[None]] = set()
_PRERELEASE_WARNED: set[str] = set()
_PRERELEASE_WARNING_LOCK = threading.Lock()

_WINDOWS_EXIT_MEANINGS = {
    0xC0000409: "fast-fail/abort",
    0xC0000005: "access violation",
    0xC00000FD: "stack overflow",
}


#: Item types Codex emits at the start of EVERY turn (observed live, 0.171):
#: the echo of our own prompt, plus reasoning, which has no side effect.
_NON_OUTPUT_ITEM_TYPES = frozenset({"userMessage", "reasoning"})


def _is_agent_output(message: dict[str, Any]) -> bool:
    """True when a notification shows the agent acted or produced output.

    Observed turn start (codex-cli 0.171): ``turn/started`` then
    ``item/started`` + ``item/completed`` for the ``userMessage`` (our own
    prompt) BEFORE the first ``agentMessage``. Counting any ``item/*`` would
    therefore block every retry, so the prompt echo and reasoning are ignored.
    """
    method = message.get("method")
    if not isinstance(method, str) or not method.startswith("item/"):
        return False
    item = (message.get("params") or {}).get("item")
    if isinstance(item, dict):
        return item.get("type") not in _NON_OUTPUT_ITEM_TYPES
    # Deltas and approval/tool requests carry no ``item`` object.
    return "/reasoning/" not in method


# Exit codes that mean the child process crashed (not a clean error exit).
_CRASH_EXIT_CODES = frozenset(_WINDOWS_EXIT_MEANINGS)


def is_crash_returncode(returncode: int | None) -> bool:
    """Crash-like exit: a known Windows fatal code or death by signal (< 0)."""
    if returncode is None:
        return False
    return returncode < 0 or (returncode & 0xFFFFFFFF) in _CRASH_EXIT_CODES


@lru_cache(maxsize=32)
def _probe_codex_version(executable: str) -> str | None:
    """Read a CLI version without letting a broken install block discovery."""
    try:
        return subprocess.run(
            [executable, "--version"],
            capture_output=True,
            check=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def _display_version(executable: str) -> str:
    return (_probe_codex_version(executable) or "unknown").strip()


def _returncode_diagnostic(returncode: int | None) -> str:
    if returncode is None:
        return "unknown"
    unsigned = returncode & 0xFFFFFFFF
    rendered = f"{returncode} (0x{unsigned:08X})"
    if meaning := _WINDOWS_EXIT_MEANINGS.get(unsigned):
        rendered += f" {meaning}"
    return rendered


def _parse_codex_version(
    output: str | None,
) -> tuple[bool, tuple[int, int, int], tuple[tuple[int, int | str], ...]] | None:
    """Parse SemVer precedence, ignoring build metadata."""
    if output is None:
        return None
    number = r"(0|[1-9][0-9]*)"
    identifier = r"[0-9A-Za-z-]+"
    match = re.fullmatch(
        rf"codex-cli {number}\.{number}\.{number}"
        rf"(?:-({identifier}(?:\.{identifier})*))?"
        rf"(?:\+{identifier}(?:\.{identifier})*)?",
        output.strip(),
    )
    if match is None:
        return None
    major, minor, patch, prerelease = match.groups()
    identifiers: list[tuple[int, int | str]] = []
    if prerelease:
        for part in prerelease.split("."):
            if part.isascii() and part.isdigit():
                if len(part) > 1 and part.startswith("0"):
                    return None
                identifiers.append((0, int(part)))
            else:
                identifiers.append((1, part))
    return prerelease is None, (int(major), int(minor), int(patch)), tuple(identifiers)


@lru_cache(maxsize=1)
def _select_app_codex(candidates: tuple[tuple[str, int], ...]) -> str:
    """Cache discovery until the candidate paths or modification times change."""

    def rank(
        candidate: tuple[str, int],
    ) -> tuple[bool, bool, tuple[int, int, int], tuple[tuple[int, int | str], ...], int]:
        executable, mtime = candidate
        version = _parse_codex_version(_probe_codex_version(executable))
        if version is None:
            return False, False, (0, 0, 0), (), mtime
        stable, numeric, prerelease = version
        return True, stable, numeric, prerelease, mtime

    return max(candidates, key=rank)[0]


def codex_executable() -> str:
    """Locate the ``codex`` CLI.

    ``CODEX_BIN`` wins, then ``PATH`` (npm/global installs), then the Codex
    desktop app's versioned install directory — the latter is *not* added to
    ``PATH``, so its ``bin/<hash>/codex.exe`` layout would otherwise be
    unreachable from ``create_subprocess_exec``.

    App directories accept only exact ``codex.exe`` / ``codex`` filenames
    directly under each hash directory. Parsed versions rank stable releases
    before prereleases, then by numeric version and SemVer prerelease order,
    with mtime breaking ties. Failed, timed-out or unparseable version probes
    rank last, ordered by mtime. App selection is cached by paths and mtimes;
    environment and PATH overrides are checked on every call without probing.
    """
    if override := os.environ.get("CODEX_BIN"):
        return override
    if found := shutil.which("codex"):
        return found
    roots = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "OpenAI" / "Codex" / "bin",
        Path.home() / ".local" / "share" / "OpenAI" / "Codex" / "bin",
    ]
    candidates = [
        (str(exe), exe.stat().st_mtime_ns)
        for root in roots
        if root.is_dir()
        for name in ("codex.exe", "codex")
        for exe in root.glob(f"*/{name}")
        if exe.name == name and exe.is_file()
    ]
    if candidates:
        return _select_app_codex(tuple(sorted(set(candidates))))
    raise FileNotFoundError(
        "codex CLI not found on PATH or in the Codex app install directory — "
        "install it (`npm install -g @openai/codex`) or set CODEX_BIN to its full path."
    )


class CodexRequestRejected(RuntimeError):
    """The App Server answered one of our requests with a JSON-RPC error.

    A rejection is a *definitive* answer — an invalid review target, an unknown
    thread id — so it must not be dressed up as :class:`CodexTurnUncertain`:
    nothing was accepted, and the caller needs the server's actual message.
    """


class CodexTransportError(ConnectionError):
    """The App Server child died or its pipes broke.

    Still a ``ConnectionError`` so callers that treat transport failures as
    transient keep working. ``turn_sent`` records whether the turn request had
    been handed to the child (set before the write): once it is true the turn
    may have run, so neither this client nor the engine may replay it.
    """

    def __init__(
        self,
        message: str,
        *,
        pid: int | None = None,
        returncode: int | None = None,
        phase: str = "",
        stderr_tail: str = "",
        turn_sent: bool = False,
        executable: str | None = None,
        version: str | None = None,
    ) -> None:
        super().__init__(message)
        self.pid = pid
        self.returncode = returncode
        self.phase = phase
        self.stderr_tail = stderr_tail
        self.turn_sent = turn_sent
        self.executable = executable
        self.version = version


@dataclass
class _Attempt:
    """What the retry wrapper needs to know about one spawn, even on failure."""

    turn_sent: bool = False
    phase: str = "spawn"
    stderr: bytearray = field(default_factory=bytearray)


class CodexTurnUncertain(UncertainOutcomeError):
    """A turn on a **durable** thread failed after the server accepted it.

    ``turn/start`` is not idempotent: once the App Server has acknowledged it,
    a dropped connection leaves the turn possibly committed to the thread's
    rollout — possibly with tool side effects already performed. Replaying the
    prompt would duplicate it. This error is deliberately *not* in
    :data:`~lazybridge.engines.codex.engine._TRANSIENT_ERROR_TYPES`, so the
    engine surfaces "outcome unknown" instead of retrying blind. Resume the
    thread and inspect it before deciding.

    Ephemeral threads never raise this: nothing survives the subprocess, so a
    retry is a clean restart.
    """

    def __init__(self, message: str, *, thread_id: str, turn_id: str | None) -> None:
        super().__init__(message)
        self.thread_id = thread_id
        self.turn_id = turn_id


@dataclass(frozen=True)
class CodexRunResult:
    """Final result of one Codex App Server turn.

    ``cost_usd`` is always ``0.0``: under ChatGPT-plan auth the App Server
    reports plan rate-limit percentages, never a per-turn price. The field
    exists so ``Envelope.metadata`` stays uniform across engines.

    ``thread_id`` is the thread the turn ran in — worth keeping only when the
    thread is durable (``ephemeral=False``), since that is the handle
    ``run(thread_id=...)`` resumes.
    """

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    thread_id: str = ""
    model: str | None = None
    #: OpenAI convention: a SUBSET of ``input_tokens`` (``cachedInputTokens``).
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@contextlib.contextmanager
def _client_wait(progress: dict[str, Any] | None) -> Any:
    """Mark a period where the App Server waits on this client (tool call, approval).

    The engine's idle watcher does not trip while such a wait is in flight, and
    the end of the wait counts as activity.
    """
    if progress is None:
        yield
        return
    progress["client_waits"] = progress.get("client_waits", 0) + 1
    try:
        yield
    finally:
        progress["client_waits"] -= 1
        if on_activity := progress.get("on_activity"):
            on_activity()


#: Process-wide cap on concurrent Codex App Server children. Measured on
#: Windows (codex-cli 0.167-0.171, 2026-11-21): concurrent App Servers crash
#: with 0xC0000409 in 3-10% of turns while one-at-a-time runs did not crash.
#: Set ``LAZYBRIDGE_CODEX_MAX_CONCURRENCY`` (or call
#: :func:`set_codex_max_concurrency`) to serialise Codex turns; other engines
#: are unaffected. Unset or ``0`` means unlimited (the previous behaviour).
#: The cap is per process: separate worker processes each have their own.
_concurrency_lock = threading.Lock()
_concurrency_active = 0
#: ``None`` = not configured explicitly: fall back to the environment variable.
_concurrency_explicit: int | None = None
_concurrency_explicit_set = False


class _SlotHolder:
    """A held slot; inherited (by reference) by tasks spawned under it."""

    __slots__ = ("active",)

    def __init__(self) -> None:
        self.active = True


#: The slot held by this call chain, if any. A context *object*: tasks spawned
#: while it was held keep a reference, and ``active`` turns False when the
#: holder releases, so a late background task cannot ride on a slot that is
#: gone. While it is active, Codex work started inside the chain (the client
#: of the same call, or a nested Codex agent run from a tool callback while the
#: parent turn is suspended waiting for that tool) runs without taking another
#: slot: waiting would deadlock with a cap of 1. Inline nested agents may
#: therefore exceed the cap while the parent is idle. Detached delegation
#: jobs start without this binding and acquire their own slot.
#: With a cap, do not start another Codex turn from inside the loop that
#: consumes a Codex stream: a full stream queue keeps the producer (and its
#: slot) waiting for the consumer, which can deadlock at cap 1.
_holder: contextvars.ContextVar[_SlotHolder | None] = contextvars.ContextVar("lazybridge_codex_slot", default=None)


def _detached_codex_context() -> contextvars.Context:
    """Copy task context while dropping the slot loan from an inline parent."""
    context = contextvars.copy_context()
    context.run(_holder.set, None)
    return context


def set_codex_max_concurrency(limit: int | None) -> None:
    """Set the per-process cap on concurrent Codex App Servers.

    ``None`` or ``0`` means unlimited and overrides the environment variable.
    A new cap applies to new acquisitions; turns already running are not
    interrupted, and no new turn starts while the active count is at or above
    the new cap.
    """
    global _concurrency_explicit, _concurrency_explicit_set
    value = int(limit or 0)
    if value < 0:
        raise ValueError(f"codex max concurrency must be >= 0, got {limit!r}")
    with _concurrency_lock:
        _concurrency_explicit = value or None
        _concurrency_explicit_set = True


def reset_codex_max_concurrency() -> None:
    """Forget an explicit cap and read ``LAZYBRIDGE_CODEX_MAX_CONCURRENCY`` again (tests)."""
    global _concurrency_explicit, _concurrency_explicit_set
    with _concurrency_lock:
        _concurrency_explicit = None
        _concurrency_explicit_set = False


def _current_cap() -> int | None:
    if _concurrency_explicit_set:
        return _concurrency_explicit
    raw = os.environ.get("LAZYBRIDGE_CODEX_MAX_CONCURRENCY", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        _logger.warning("ignoring invalid LAZYBRIDGE_CODEX_MAX_CONCURRENCY=%r", raw)
        return None
    return value if value > 0 else None


async def _acquire_counted() -> None:
    global _concurrency_active
    # Poll a counter guarded by a threading lock: works across event loops and
    # threads, keeps the waiter cancellable, and honours cap changes at once.
    while True:
        with _concurrency_lock:
            cap = _current_cap()
            if cap is None or _concurrency_active < cap:
                _concurrency_active += 1
                return
        await asyncio.sleep(0.05)


def _release_counted() -> None:
    global _concurrency_active
    with _concurrency_lock:
        _concurrency_active -= 1


async def acquire_codex_slot() -> _SlotHolder | None:
    """Take a slot unless this chain already holds an active one (then ``None``).

    The caller must pass the result to :func:`release_codex_slot`. Installing
    the holder in a context is separate (:func:`enter_codex_slot`) so a stream
    can bind it to its producer task rather than to the consumer.
    """
    parent = _holder.get()
    if parent is not None and parent.active:
        return None
    await _acquire_counted()
    return _SlotHolder()


def release_codex_slot(holder: _SlotHolder | None) -> None:
    if holder is None or not holder.active:
        return
    holder.active = False
    _release_counted()


def enter_codex_slot(holder: _SlotHolder | None) -> None:
    """Bind ``holder`` to the current context (task-local; never reset)."""
    if holder is not None:
        _holder.set(holder)


@contextlib.asynccontextmanager
async def codex_concurrency_slot() -> AsyncIterator[None]:
    """Hold one Codex slot for a whole engine call (cancellable wait).

    The engine takes it *before* its request/idle timeouts start, so time
    queued never consumes a turn's timeout budget. Re-entry inside a chain that
    already holds an active slot does not wait (see ``_holder``).
    """
    holder = await acquire_codex_slot()
    token = _holder.set(holder) if holder is not None else None
    try:
        yield
    finally:
        release_codex_slot(holder)
        if token is not None:
            # A generator finalised in another task cannot reset the token;
            # the holder is already inactive, which is what matters.
            with contextlib.suppress(ValueError):
                _holder.reset(token)


_client_slot = codex_concurrency_slot


ProbeCallback = Callable[[Callable[[str, dict[str, Any]], Awaitable[Any]], dict[str, Any]], Awaitable[bool]]


class CodexAppServerClient:
    """One ``codex app-server`` subprocess per run, torn down in ``finally``."""

    def __init__(
        self,
        command: tuple[str, ...] | None = None,
        *,
        policy_release_report: Path | str | None = None,
        policy_state_dir: Path | str | None = None,
        policy_api_key: str | None = None,
    ) -> None:
        #: Resolved lazily so importing the package never touches the
        #: filesystem and a missing CLI surfaces at run() time, as a normal
        #: engine error, not at construction.
        self.command = command
        self.policy_release_report = Path(policy_release_report) if policy_release_report is not None else None
        self.policy_state_dir = Path(policy_state_dir) if policy_state_dir is not None else None
        self._policy_api_key = policy_api_key

    def _spawn_command(self, config_overrides: tuple[str, ...] = ()) -> tuple[str, ...]:
        """The argv for this run's App Server, per-agent overrides included.

        ``-c key=value`` is an App Server option and not merely an
        interactive one (verified against ``codex app-server --help``), so a
        setting given here lands on this subprocess and nowhere else — the
        shared ``~/.codex/config.toml`` is never touched.

        Overrides are appended to a caller-supplied ``command`` as well.
        Dropping them there would silently lose exactly the setting the
        caller asked for, which is worse than a custom command having to
        tolerate two extra argv entries.

        ``--strict-config`` is added, but ONLY when there is at least one
        override to protect. Verified live: without it, ``-c`` with an
        invented key exits cleanly and nothing is reported, so a key a running Codex build does not recognise is a
        silent no-op — the agent keeps Codex's default despite an explicit
        policy, and nothing says so. With it, the same invented key is a
        startup error instead (reported with the stderr tail).

        The flag is not applied unconditionally because it validates the
        *entire* resolved configuration, file included, not just this call's
        ``-c`` values — an agent that never sets a per-agent override should
        see zero change in whether its existing ``config.toml`` is accepted.
        Scoping it to "an override is present" means the stricter check only
        ever applies to the one case it exists to protect.
        """
        base = self.command or (codex_executable(), "app-server")
        if not config_overrides:
            return base
        return (
            *base,
            "--strict-config",
            *(part for value in config_overrides for part in ("-c", value)),
        )

    def prerelease_warning(self) -> dict[str, str] | None:
        """Warn once when this client will launch a prerelease Codex binary."""
        if self.command is not None:
            return None  # caller-supplied command: not ours to probe or judge
        try:
            executable = self._spawn_command()[0]
        except OSError:
            # Discovery/spawn reports the real configuration error in run().
            return None
        version = _display_version(executable)
        parsed = _parse_codex_version(version)
        if parsed is None:
            _logger.debug("Codex version output %r did not match the expected pattern", version)
            return None
        if parsed[0]:
            return None
        with _PRERELEASE_WARNING_LOCK:
            if executable in _PRERELEASE_WARNED:
                return None
            _PRERELEASE_WARNED.add(executable)
        message = (
            f"Codex prerelease CLI selected: {executable} ({version}). "
            "Use CODEX_BIN or PATH to select a stable Codex CLI."
        )
        _logger.warning(message)
        return {"executable": executable, "version": version, "message": message}

    async def run(
        self,
        *,
        prompt: str,
        model: str | None,
        cwd: str | None,
        dynamic_tools: list[dict[str, Any]],
        on_tool_call: ToolCallback,
        developer_instructions: str | None = None,
        on_text: Callable[[str], Awaitable[None]] | None = None,
        attachments: list[dict[str, Any]] | None = None,
        effort: str | None = None,
        output_schema: dict[str, Any] | None = None,
        sandbox: str = "read-only",
        writable_roots: list[str] | None = None,
        approval_policy: str = "never",
        approval_gate: ApprovalGate | None = None,
        thread_id: str | None = None,
        ephemeral: bool = True,
        review_target: dict[str, Any] | None = None,
        progress: dict[str, Any] | None = None,
        config_overrides: tuple[str, ...] = (),
        thread_source: str | None = None,
    ) -> CodexRunResult:
        """Run one turn, in a fresh thread or in ``thread_id``.

        A transport failure during startup, before the turn request was
        written, is retried once after a short jittered pause. After that
        point nothing is retried: the child may already have executed the turn
        and its side effects. Failures raise :class:`CodexTransportError`
        carrying the child's pid, return code, RPC phase and stderr tail.

        ``thread_id`` resumes an existing durable thread (``thread/resume``)
        instead of starting one, so Codex' own transcript carries the history
        — its file reads and prior reasoning included — and the caller does not
        have to re-send context. Resuming implies durability, so it also
        forces ``ephemeral=False``.

        Everything configurable is **re-supplied on resume** rather than left
        to whatever the stored thread recorded: ``cwd``, ``sandbox``,
        ``model``, ``developer_instructions`` and, above all, ``dynamic_tools``
        — the tool *callbacks* live in this subprocess, so a resumed thread
        cannot inherit them from the process that started it.

        ``ephemeral=True`` (the default) keeps the old behaviour: the thread
        exists only inside this subprocess and is unresumable afterwards —
        verified live, ``thread/resume`` on it answers ``no rollout found for
        thread id``.

        ``progress``, if given, is filled in as the call proceeds:
        ``thread_id`` as soon as the thread exists and ``turn_sent`` when the
        turn request goes out. It exists because the caller's own timeout
        cancels this coroutine outright — ``CancelledError`` unwinds past every
        ``except Exception`` here — and without it the engine cannot tell a
        hang during startup (nothing sent, safe to retry) from a hang after a
        durable turn was accepted (outcome unknown), nor report the id of a
        thread it never got a result from.

        ``review_target`` switches the turn to Codex' **native review mode**
        (``review/start``) instead of ``turn/start``: a typed target —
        ``{"type": "uncommittedChanges"}``, ``{"type": "baseBranch",
        "branch": "main"}`` or ``{"type": "commit", "sha": ...}`` — reviewed by
        Codex' own review harness, which returns severity-tagged findings
        (``[P1]``/``[P2]`` with file:line). The protocol has **no prompt slot**
        there, so ``prompt`` is not sent and the review cannot be steered;
        that is the trade for the harness. Delivery is always ``inline`` (the
        review runs in this thread, so a follow-up turn can refer to it):
        ``detached`` was measured to complete on a *different* thread and to
        raise an approval request the parent thread never sees, which is not
        usable non-interactively.

        ``thread_source`` is the protocol's own free-text
        ``ThreadStartParams.threadSource`` — "an optional client-supplied
        analytics source classification for this thread" (verified against
        the App Server's generated schema). Live-verified landing on disk as
        ``session_meta.payload.thread_source`` (snake_case) in the rollout
        file — a DIFFERENT field from ``session_meta.payload.source``, which
        is something else entirely (observed as ``"vscode"`` regardless of
        this value, on every thread checked, LazyBridge or interactive).
        Every LazyBridge-created thread is *already* identifiable without
        this field at all, via ``session_meta.payload.originator`` — always
        ``"lazybridge"``, set unconditionally by the ``initialize`` call's
        ``clientInfo.name`` (see below), not configurable per instance.
        ``thread_source`` adds a second, caller-chosen label for
        distinguishing *which* LazyBridge-based application created a given
        thread, on top of that. It is creation-time metadata: sent on
        ``thread/start`` only, never on ``thread/resume`` — a resumed thread
        already carries the value its creating call set, and Codex has no
        endpoint to change it after the fact.
        """
        async with _client_slot():
            kwargs = dict(
                prompt=prompt,
                model=model,
                cwd=cwd,
                dynamic_tools=dynamic_tools,
                on_tool_call=on_tool_call,
                developer_instructions=developer_instructions,
                on_text=on_text,
                attachments=attachments,
                effort=effort,
                output_schema=output_schema,
                sandbox=sandbox,
                writable_roots=writable_roots,
                approval_policy=approval_policy,
                approval_gate=approval_gate,
                thread_id=thread_id,
                ephemeral=ephemeral,
                review_target=review_target,
                progress=progress,
                config_overrides=config_overrides,
                thread_source=thread_source,
            )
            context = current_policy_context()
            if context is None:
                return await self._run_with_startup_retry(**kwargs)
            if self.command is not None or review_target is not None:
                raise PolicyDenied("codex", "Custom commands and native review lack governed launch evidence.")
            executable = await asyncio.to_thread(codex_executable)
            launch = await asyncio.to_thread(
                PolicyLaunch.admit, context.effective_policy, executable, self.policy_release_report
            )
            if cwd is not None and Path(cwd).resolve() != launch.policy.folder:
                raise PolicyDenied("codex", "Codex cwd differs from the assigned folder.")
            return await self._run_policy_launch(launch, kwargs)

    async def _run_policy_launch(
        self, launch: PolicyLaunch, kwargs: dict[str, Any], *, probe: ProbeCallback | None = None
    ) -> CodexRunResult:
        """Shared production and offline-model release-check launch path."""
        thread_id = kwargs.get("thread_id")
        with launch.home(
            thread_id, bool(thread_id) or not kwargs.get("ephemeral", True), self.policy_state_dir, self._policy_api_key
        ) as (home, env):
            return await self._run_with_startup_retry(
                **{
                    **kwargs,
                    "cwd": str(launch.policy.folder),
                    "sandbox": launch.config["sandbox_mode"],
                    "approval_policy": launch.config["approval_policy"],
                    "approval_gate": None,
                    "writable_roots": None,
                    "config_overrides": (),
                    "policy_launch": launch,
                    "process_options": {"cwd": home, "env": env},
                    "probe": probe,
                }
            )

    async def _run_with_startup_retry(self, **kwargs: Any) -> CodexRunResult:
        for attempt in range(2):
            state = _Attempt()
            try:
                return await self._run_once(state, **kwargs)
            except CodexTransportError as exc:
                if exc.turn_sent or attempt or exc.phase not in _RETRYABLE_BEFORE_TURN:
                    raise
                await asyncio.sleep(random.uniform(*_STARTUP_RETRY_DELAY))
        raise AssertionError("unreachable")  # pragma: no cover

    async def _run_once(
        self,
        state: _Attempt,
        *,
        prompt: str,
        model: str | None,
        cwd: str | None,
        dynamic_tools: list[dict[str, Any]],
        on_tool_call: ToolCallback,
        developer_instructions: str | None,
        on_text: Callable[[str], Awaitable[None]] | None,
        attachments: list[dict[str, Any]] | None,
        effort: str | None,
        output_schema: dict[str, Any] | None,
        sandbox: str,
        writable_roots: list[str] | None,
        approval_policy: str,
        approval_gate: ApprovalGate | None,
        thread_id: str | None,
        ephemeral: bool,
        review_target: dict[str, Any] | None,
        progress: dict[str, Any] | None,
        config_overrides: tuple[str, ...],
        thread_source: str | None,
        policy_launch: PolicyLaunch | None = None,
        process_options: dict[str, Any] | None = None,
        probe: ProbeCallback | None = None,
    ) -> CodexRunResult:
        if thread_id:
            ephemeral = False
        # Off the loop: the first lookup may probe app installs with `--version`.
        if policy_launch is None:
            command = await asyncio.to_thread(self._spawn_command, config_overrides)
        else:
            command = (policy_launch.executable, "app-server", "--strict-config")
        executable = command[0]
        version = (
            "custom command" if self.command is not None else await asyncio.to_thread(_display_version, executable)
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                limit=_STDOUT_LINE_LIMIT,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **(process_options or {}),
            )
        except (FileNotFoundError, PermissionError, NotADirectoryError, IsADirectoryError):
            raise  # configuration problems, not transport blips
        except OSError as exc:
            raise CodexTransportError(
                f"Codex App Server failed to spawn: {type(exc).__name__}: {exc}", phase="spawn"
            ) from exc
        assert process.stdin and process.stdout and process.stderr
        stderr_stream = process.stderr

        async def drain_stderr() -> None:
            # Always drained: an unread PIPE blocks the child once it fills.
            try:
                while chunk := await stderr_stream.read(4096):
                    state.stderr += chunk
                    if len(state.stderr) > _STDERR_TAIL_BYTES:
                        del state.stderr[: len(state.stderr) - _STDERR_TAIL_BYTES]
            except Exception:  # diagnostics must never break the run
                pass

        stderr_task = asyncio.create_task(drain_stderr())

        async def transport_failure(reason: str) -> CodexTransportError:
            returncode = process.returncode
            if returncode is None:
                with contextlib.suppress(TimeoutError):
                    returncode = await asyncio.wait_for(process.wait(), _EXIT_GRACE)
            await asyncio.wait({stderr_task}, timeout=_EXIT_GRACE / 4)
            tail = bytes(state.stderr).decode("utf-8", errors="replace").strip()
            message = (
                f"{reason} (executable={executable}, version={version}, pid={process.pid}, "
                f"returncode={_returncode_diagnostic(returncode)}, phase={state.phase}, "
                f"turn_sent={state.turn_sent})"
            )
            if tail:
                message += f"; stderr tail: {tail[-_STDERR_MESSAGE_CHARS:]}"
            else:
                message += "; stderr empty"
            return CodexTransportError(
                message,
                pid=process.pid,
                returncode=returncode,
                phase=state.phase,
                stderr_tail=tail,
                turn_sent=state.turn_sent,
                executable=executable,
                version=version,
            )

        pending: dict[int, asyncio.Future[Any]] = {}
        completed: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        if progress is not None:
            # Once the turn has an outcome, no further turn events can arrive:
            # the idle clock must not run during subprocess teardown.
            progress["turn_done"] = False
            completed.add_done_callback(lambda _: progress.__setitem__("turn_done", True))
        #: Last cumulative ``total`` seen per turn id. Keyed rather than
        #: folded into one value because a notification can arrive *before*
        #: the ``turn/start`` response tells us which id is ours: deciding
        #: "history or mine" on arrival would then attribute our own first
        #: usage report to the baseline and undercount the turn. Totals are
        #: cumulative and monotonic, so at the end ours is ``totals[turn_id]``
        #: and the baseline is the largest of the rest.
        totals: dict[str, dict[str, Any]] = {}
        #: Set from the ``turn/start`` response, which can land after the
        #: server's first notifications about that same turn.
        turn_id: str | None = None
        turn_sent = False
        turn_request_id: int | None = None
        #: A resumed thread has a past: the server can replay a
        #: ``turn/completed`` for a turn that finished long ago. On a fresh
        #: thread there is nothing to replay, so an unnamed completion there
        #: can only be ours.
        replay_possible = bool(thread_id)
        #: Completions seen on a resumed thread before the acknowledgement told
        #: us which turn is ours. Either a replay of an older turn or our own
        #: arriving early — only the id can say which, so they wait for it.
        held_completions: list[dict[str, Any]] = []
        counter = 0

        async def send(message: dict[str, Any]) -> None:
            assert process.stdin
            process.stdin.write((json.dumps(message) + "\n").encode())
            try:
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                raise await transport_failure(f"Codex App Server pipe broke: {type(exc).__name__}") from exc

        async def request(method: str, params: dict[str, Any], *, is_turn: bool = False) -> Any:
            nonlocal counter, turn_request_id
            counter += 1
            state.phase = method
            if is_turn:
                # Recorded before the round-trip so the reader can pick our
                # turn id out of the response itself: assigning it here, after
                # ``await``, would lose a race with notifications the server
                # already sent about that same turn.
                turn_request_id = counter
            future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
            pending[counter] = future
            await send({"method": method, "id": counter, "params": params})
            if is_turn and progress is not None:
                # After the write, not before: serialising an unserialisable
                # ``review_target`` raises here, and a request that never left
                # the process must not count as a turn that may have run.
                progress["turn_sent"] = True
                if on_activity := progress.get("on_activity"):
                    on_activity()
            result = await future
            if is_turn:
                state.phase = "turn running"
            return result

        def fail_waiters(exc: BaseException) -> None:
            """Wake the active RPC waiter, or the turn waiter, after reader failure."""
            unresolved = [future for future in pending.values() if not future.done()]
            pending.clear()
            for future in unresolved:
                future.set_exception(exc)
            if not completed.done():
                # Before turn/start is acknowledged, the request future is the
                # one the caller awaits; cancelling the unused completion future
                # avoids an unobserved-exception warning. Once no RPC request is
                # pending, completion is the active waiter and receives the error.
                if unresolved:
                    completed.cancel()
                else:
                    completed.set_exception(exc)

        #: Server-initiated requests (tool calls, approvals) being served. They
        #: run off the reader so it keeps draining stdout — and so notices EOF,
        #: a child exit or ``turn/completed`` — while a tool is still running.
        client_tasks: set[asyncio.Task[None]] = set()
        #: Approval prompts are serialised per run: concurrent requests must not
        #: reach a shared (terminal) gate at once, or answers could be crossed.
        approval_lock = asyncio.Lock()

        def turn_done_flag() -> bool:
            return bool(progress is not None and progress.get("turn_done"))

        async def respond(request_id: Any, result: dict[str, Any]) -> None:
            """Answer a server request, unless nobody can use the answer any more."""
            if process.returncode is not None or process.stdin is None or process.stdin.is_closing():
                return  # nobody left to read it; the reader reports the exit
            if completed.done() or turn_done_flag():
                return  # the turn has ended: a late answer would hit a dead request
            if (me := asyncio.current_task()) is not None and me.cancelling():
                return  # cancelled, even if the callback swallowed the cancellation
            try:
                await send({"id": request_id, "result": result})
            except CodexTransportError as exc:
                fail_waiters(exc)

        def spawn_client_task(coro: Coroutine[Any, Any, None]) -> None:
            task = asyncio.create_task(coro)
            client_tasks.add(task)

            def done(finished: asyncio.Task[None]) -> None:
                client_tasks.discard(finished)
                if not finished.cancelled() and (exc := finished.exception()) is not None:
                    fail_waiters(exc)

            task.add_done_callback(done)

        def cancel_client_tasks() -> None:
            for task in list(client_tasks):
                task.cancel()

        async def read_loop() -> None:
            try:
                await read_messages()
            finally:
                # The turn ended or the child went away: no response can be
                # delivered any more, so stop serving requests.
                cancel_client_tasks()

        async def read_messages() -> None:
            nonlocal turn_id
            assert process.stdout
            try:
                while line := await process.stdout.readline():
                    message = json.loads(line)
                    method = message.get("method")
                    if progress is not None and _is_agent_output(message):
                        progress["output_item_observed"] = True
                    if turn_sent and method is not None and progress is not None:
                        params = message.get("params") or {}
                        incoming_turn = params.get("turnId") or (params.get("turn") or {}).get("id")
                        incoming_thread = params.get("threadId")
                        on_activity = progress.get("on_activity")
                        if (
                            on_activity
                            and (turn_id is None or incoming_turn in (None, turn_id))
                            and incoming_thread in (None, active_thread)
                        ):
                            on_activity()
                    if method is None:
                        # A response to one of our own requests. Dispatching on
                        # the absence of "method" (rather than on "id" alone)
                        # matters: the App Server numbers its requests to us from
                        # 0 with a separate counter, so an ``item/tool/call`` id
                        # can collide with a still-pending client request id.
                        if message.get("id") == turn_request_id and "error" not in message:
                            # Learn our turn's id here, in message order, so
                            # every later notification can be attributed...
                            turn_id = ((message.get("result") or {}).get("turn") or {}).get("id")
                            # ...and settle anything that arrived before it.
                            # A completion can outrun its own acknowledgement;
                            # dropping it would hang the call until timeout and
                            # then report a turn that demonstrably finished as
                            # "outcome unknown".
                            if not completed.done():
                                # An exact id match wins over an unnamed one:
                                # taking the first unnamed completion would
                                # hand back a stale turn that happened to
                                # arrive before ours.
                                match = next(
                                    (h for h in held_completions if h.get("id") == turn_id),
                                    None,
                                ) or next(
                                    (h for h in held_completions if h.get("id") is None),
                                    None,
                                )
                                if match is not None:
                                    completed.set_result(match)
                            held_completions.clear()
                        future = pending.pop(message.get("id"), None)
                        if future is not None and not future.done():
                            if "error" in message:
                                # Recorded HERE, as the rejection is read —
                                # not where it is caught. Between the two the
                                # awaiting task can be cancelled, and the fact
                                # "this turn never ran" would be lost with it.
                                if progress is not None and message.get("id") == turn_request_id:
                                    progress["rejected"] = True
                                future.set_exception(
                                    CodexRequestRejected(message["error"].get("message", "Codex App Server error"))
                                )
                            else:
                                future.set_result(message.get("result", {}))
                    elif method == "item/tool/call":
                        if progress is not None:
                            progress["tool_call_observed"] = True
                        params = message["params"]
                        request_id = message["id"]

                        async def run_tool(params: dict[str, Any] = params, request_id: Any = request_id) -> None:
                            try:
                                with _client_wait(progress):
                                    result = await on_tool_call(params["tool"], params.get("arguments", {}))
                            except Exception as exc:  # defensive protocol response
                                result = {
                                    "success": False,
                                    "contentItems": [{"type": "inputText", "text": str(exc)}],
                                }
                            await respond(request_id, result)

                        spawn_client_task(run_tool())
                    elif method in {
                        "item/commandExecution/requestApproval",
                        "item/fileChange/requestApproval",
                        "item/permissions/requestApproval",
                    }:
                        params = message.get("params", {})
                        request_id = message["id"]
                        if method == "item/commandExecution/requestApproval":
                            kind, name = "command", "codex-shell"
                        elif method == "item/fileChange/requestApproval":
                            kind, name = "file_change", str(params.get("grantRoot") or "file changes")
                        else:
                            kind, name = "permissions", "sandbox permissions"

                        async def run_approval(
                            params: dict[str, Any] = params,
                            request_id: Any = request_id,
                            kind: str = kind,
                            name: str = name,
                        ) -> None:
                            with _client_wait(progress):
                                async with approval_lock:
                                    decision = await ask_approval(
                                        approval_gate,
                                        ApprovalRequest(
                                            provider="codex",
                                            kind=kind,  # type: ignore[arg-type]
                                            name=name,
                                            arguments=params,
                                            reason=params.get("reason"),
                                            cwd=params.get("cwd") or cwd,
                                            thread_id=params.get("threadId"),
                                            turn_id=params.get("turnId"),
                                            raw=params,
                                        ),
                                    )
                            if kind == "permissions":
                                requested = params.get("permissions", {})
                                granted = decision.permissions if decision.permissions is not None else requested
                                result = {
                                    "permissions": granted if decision.action in {"allow", "allow_session"} else {},
                                    "scope": "session" if decision.action == "allow_session" else "turn",
                                }
                            else:
                                native = {
                                    "allow": "accept",
                                    "allow_session": "acceptForSession",
                                    "deny": "decline",
                                    "cancel": "cancel",
                                }[decision.action]
                                result = {"decision": native}
                            await respond(request_id, result)

                        spawn_client_task(run_approval())
                    elif method == "item/agentMessage/delta":
                        delta = message.get("params", {}).get("delta", "")
                        if delta and on_text:
                            # Awaited so a slow consumer (bounded queue in
                            # CodexEngine.stream()) applies backpressure all the
                            # way back to this reader, same as the sink-based
                            # streaming path in LLMEngine / ClaudeCodeEngine.
                            await on_text(delta)
                    elif method == "thread/tokenUsage/updated":
                        # The only place the App Server reports usage — the
                        # ``turn/completed`` payload carries none. ``total`` (not
                        # ``last``) is used because ``last`` is only the final
                        # model call and would drop the ones made before each
                        # tool round-trip. But ``total`` is cumulative over the
                        # *thread*, so on a resumed thread it includes every
                        # earlier turn (verified live: 15137 after turn 1,
                        # 30292 after turn 2). Hence the baseline below, and
                        # the ``turnId`` filter: notifications tagged with an
                        # older turn are history, not this turn's cost.
                        params = message.get("params", {})
                        total = params.get("tokenUsage", {}).get("total")
                        if isinstance(total, dict):
                            # Recorded under its own turn id and attributed
                            # later: classifying it now would have to guess
                            # whose it is whenever it outruns the turn/start
                            # response.
                            totals[str(params.get("turnId"))] = total
                    elif method == "error" and not completed.done():
                        params = message.get("params", {})
                        if not params.get("willRetry"):
                            # A terminal error notification is the server
                            # *telling* us the turn failed — a usage limit, a
                            # refused request. Definitive, like a JSON-RPC
                            # rejection, so it must not be dressed up as "the
                            # turn may have run": seen live as "You've hit your
                            # usage limit" reported as an uncertain turn.
                            if progress is not None:
                                progress["rejected"] = True
                            completed.set_exception(
                                CodexRequestRejected(params.get("error", {}).get("message", "Codex App Server error"))
                            )
                    elif method == "turn/completed" and not completed.done():
                        # Only *our* turn completes the run. On a resumed
                        # thread the server can replay notifications for turns
                        # that finished long ago; taking the first one would
                        # return a previous answer as this call's result.
                        turn = message["params"]["turn"]
                        incoming = turn.get("id")
                        if not turn_sent:
                            continue  # nothing before we asked can be ours
                        if turn_id is not None:
                            # Our id is known: require a match, unless the
                            # server sent no id at all (older fixtures do).
                            if incoming is not None and incoming != turn_id:
                                continue
                        elif replay_possible:
                            # Resumed thread, our id not yet known: this is
                            # exactly the window where a replayed completion
                            # from an older turn would be taken as the answer.
                            # Held rather than dropped — it may equally be OUR
                            # completion outrunning its acknowledgement — and
                            # settled above once the id arrives.
                            held_completions.append(turn)
                            continue
                        completed.set_result(turn)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if isinstance(exc, CodexTransportError):
                    fail_waiters(exc)
                else:
                    failure = await transport_failure(f"Codex App Server reader failed: {type(exc).__name__}: {exc}")
                    failure.__cause__ = exc
                    fail_waiters(failure)
            else:
                if pending or not completed.done():
                    fail_waiters(await transport_failure("Codex App Server exited before completing the request"))

        reader = asyncio.create_task(read_loop())
        try:
            await request(
                "initialize",
                {
                    "clientInfo": {"name": "lazybridge", "title": "LazyBridge", "version": "0.1.0"},
                    "capabilities": {"experimentalApi": True},
                },
            )
            state.phase = "initialized"
            await send({"method": "initialized", "params": {}})
            if policy_launch is not None and probe is None:
                config = await request("config/read", {"cwd": cwd, "includeLayers": True})
                mismatches = policy_launch.config_mismatches(config)
                if mismatches:
                    raise PolicyDenied("codex", f"Effective Codex configuration differs: {', '.join(mismatches)}.")
            # ``sandbox`` is the CLI's kebab-case ``SandboxMode`` enum
            # (read-only / workspace-write / danger-full-access); "readOnly"
            # is rejected outright with "unknown variant".
            # Everything the thread runs with is sent on BOTH paths: a resumed
            # thread must not silently inherit the cwd/sandbox/model recorded
            # when it was created, and its dynamic-tool *callbacks* live in
            # this subprocess, so they have to be registered again here.
            thread_params: dict[str, Any] = {
                "model": model,
                "cwd": cwd,
                "approvalPolicy": approval_policy,
                "sandbox": sandbox,
                "dynamicTools": dynamic_tools,
            }
            if policy_launch is not None:
                thread_params["config"] = policy_launch.config
            elif writable_roots is not None:
                # An explicit [] is sent too: it REMOVES roots inherited from the user's config.toml.
                # There is no dedicated ``ThreadStartParams``/``ThreadResume
                # Params`` field for this (verified against the generated App
                # Server schema — ``codex app-server generate-json-schema``):
                # both only carry the coarse ``sandbox`` enum. The escape
                # hatch is ``config``, a free-form per-thread override with
                # the same shape as ``config.toml`` — so this is the
                # snake_case ``sandbox_workspace_write.writable_roots`` key,
                # not the camelCase RPC style used elsewhere in this params
                # dict. Additive to ``cwd``, which Codex always allows.
                thread_params["config"] = {"sandbox_workspace_write": {"writable_roots": list(writable_roots)}}
            if developer_instructions is not None:
                # Preserve Codex's own base instructions while giving the
                # application prompt the same priority as Engine.system.
                thread_params["developerInstructions"] = developer_instructions
            if thread_id:
                thread_params["threadId"] = thread_id
                thread = await request("thread/resume", thread_params)
            else:
                thread_params["ephemeral"] = ephemeral
                if thread_source is not None:
                    # Creation-time only: a resume must NOT resend this, or a
                    # thread created under one source would appear to have
                    # been reclassified under whatever resumed it.
                    thread_params["threadSource"] = thread_source
                thread = await request("thread/start", thread_params)
            active_thread = thread["thread"]["id"]
            if policy_launch is not None:
                if probe is None and (mismatches := policy_launch.thread_mismatches(thread)):
                    raise PolicyDenied("codex", f"Effective Codex thread differs: {', '.join(mismatches)}.")
                if thread_id is not None and active_thread != thread_id:
                    raise PolicyDenied("codex", "Codex resumed a different thread.")
                policy_launch.bind_thread(active_thread)
            reported_model = thread.get("model")
            resolved_model = reported_model if isinstance(reported_model, str) and reported_model.strip() else None
            if progress is not None:
                # Published before the turn: a durable thread that exists is
                # worth reporting even if this call never returns a result.
                progress["thread_id"] = active_thread
            if probe is not None and not await probe(request, thread):
                return CodexRunResult(text="", input_tokens=0, output_tokens=0, thread_id=active_thread)
            # ``input`` is the App Server's UserInput union: the text turn plus
            # any image attachments the engine converted. (Unused in native
            # review mode — ``review/start`` has no prompt slot.)
            turn_params: dict[str, Any] = {
                "threadId": active_thread,
                "input": [{"type": "text", "text": prompt}, *(attachments or [])],
            }
            if effort is not None:
                turn_params["effort"] = effort
            if output_schema is not None:
                # Native structured output: an OpenAI-strict schema, already
                # rewritten by the engine (see CodexEngine._native_output_schema).
                # Absent when the engine fell back to prompt priming instead.
                turn_params["outputSchema"] = output_schema
            try:
                # The uncertainty window opens when the request goes out, not
                # when its response comes back: the server can accept the turn
                # and then drop the connection before answering, and a retry
                # would replay a turn already committed to a durable thread —
                # tool side effects included.
                turn_sent = state.turn_sent = True
                if review_target is not None:
                    await request(
                        "review/start",
                        {"threadId": active_thread, "target": review_target, "delivery": "inline"},
                        is_turn=True,
                    )
                else:
                    await request("turn/start", turn_params, is_turn=True)
                turn = await completed
            except CodexRequestRejected:
                # The server said no. Nothing was accepted, so this is an
                # ordinary error with an actionable message — not uncertainty.
                # ``progress["rejected"]`` was already recorded by the reader,
                # where a cancellation cannot lose it.
                raise
            except Exception as exc:
                if ephemeral and policy_launch is None:
                    raise  # nothing survives the subprocess: a retry is clean
                recovery = (
                    "Inspect the workspace before retrying."
                    if ephemeral
                    else f"Resume thread {active_thread} and inspect it before retrying."
                )
                raise CodexTurnUncertain(
                    f"Turn was accepted but its outcome is unknown: {exc}. {recovery}",
                    thread_id=active_thread,
                    turn_id=turn_id,
                ) from exc
            if turn.get("status") != "completed":
                raise RuntimeError(turn.get("error", {}).get("message", f"Codex turn {turn.get('status')}"))
            # Ours by id; the baseline is the largest total reported for any
            # other turn (they are cumulative, so the largest is the state
            # immediately before this turn began).
            usage = totals.get(str(turn_id), {})
            others = [t for tid, t in totals.items() if tid != str(turn_id)]
            baseline = max(others, key=lambda t: int(t.get("totalTokens") or 0), default={})
            input_tokens = max(int(usage.get("inputTokens") or 0) - int(baseline.get("inputTokens") or 0), 0)
            output_tokens = max(int(usage.get("outputTokens") or 0) - int(baseline.get("outputTokens") or 0), 0)
            cache_read_input_tokens = max(
                int(usage.get("cachedInputTokens") or 0) - int(baseline.get("cachedInputTokens") or 0), 0
            )
            text = ""
            for item in reversed(turn.get("items", [])):
                if item.get("type") == "agentMessage":
                    text = item.get("text", "")
                    break
            return CodexRunResult(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                thread_id=active_thread,
                model=resolved_model,
                cache_read_input_tokens=cache_read_input_tokens,
            )
        finally:
            reader.cancel()
            cancel_client_tasks()
            if client_tasks:
                _, abandoned = await asyncio.wait(set(client_tasks), timeout=_CLIENT_TASK_TEARDOWN_GRACE)
                if abandoned:
                    for task in abandoned:
                        # Same as other abandoned tasks: the synchronous bridge
                        # must not wait for them while shutting its loop down.
                        abandoned_tasks.add(task)
                        _abandoned_callbacks.add(task)  # abandoned_tasks is weak
                        task.add_done_callback(_abandoned_callbacks.discard)
                    _logger.warning(
                        "Codex: abandoning %d tool/approval callback(s) that did not stop within %.1fs of cancellation",
                        len(abandoned),
                        _CLIENT_TASK_TEARDOWN_GRACE,
                    )
            try:
                await reader
            except asyncio.CancelledError:
                # Expected: we cancelled the reader on the line above, and
                # awaiting it is only how that cancellation is collected. Any
                # real reader failure has already been routed to the caller
                # through fail_waiters(), so there is nothing to re-raise here
                # — and re-raising would mask the original error.
                pass

            async def collect_child() -> None:
                await _reap(process)
                stderr_task.cancel()
                await asyncio.gather(stderr_task, return_exceptions=True)
                if policy_launch is not None:
                    # A native MCP child can keep pipe handles open on Windows.
                    # Close the asyncio transport before removing launch state.
                    transport = getattr(process, "_transport", None)
                    if transport is not None:
                        transport.close()

            await _finish_despite_cancel(asyncio.ensure_future(collect_child()))


async def _finish_despite_cancel(task: asyncio.Future[None]) -> None:
    """Await ``task`` to completion even if the caller is cancelled meanwhile.

    Cancellation is re-raised only after the task has finished, so the child
    is always killed and reaped.
    """
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    await asyncio.gather(task, return_exceptions=True)
    if cancelled:
        raise asyncio.CancelledError


async def _reap(process: asyncio.subprocess.Process) -> None:
    """Close stdin, give the child a moment to exit, then terminate and kill it."""
    if process.stdin is not None:
        try:
            process.stdin.close()
        except Exception:  # already closed or broken: nothing left to flush
            pass
    for stop, grace in ((None, _STDIN_EOF_GRACE), (process.terminate, _EXIT_GRACE), (process.kill, _EXIT_GRACE)):
        if stop is not None and process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                stop()
        try:
            await asyncio.wait_for(process.wait(), grace)
            return
        except TimeoutError:
            continue
    # Still alive after kill(): wedged beyond our reach. Stop waiting rather
    # than hold the caller (and its timeout) hostage forever.
