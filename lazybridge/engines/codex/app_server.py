"""Small JSON-RPC client for one ephemeral Codex App Server run.

The wire format here was verified against a live ``codex app-server``
(codex-cli 0.148.0); the authoritative schema comes from
``codex app-server generate-json-schema --out <dir>``.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lazybridge.engines.coding import ApprovalGate, ApprovalRequest, ask_approval

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


def codex_executable() -> str:
    """Locate the ``codex`` CLI.

    ``CODEX_BIN`` wins, then ``PATH`` (npm/global installs), then the Codex
    desktop app's versioned install directory — the latter is *not* added to
    ``PATH``, so its ``bin/<hash>/codex.exe`` layout would otherwise be
    unreachable from ``create_subprocess_exec``.
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
        exe
        for root in roots
        if root.is_dir()
        for exe in root.glob("*/codex*")
        if exe.is_file() and exe.suffix in ("", ".exe")
    ]
    if candidates:
        return str(max(candidates, key=lambda p: p.stat().st_mtime))
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


class CodexTurnUncertain(RuntimeError):
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


class CodexAppServerClient:
    """One ``codex app-server`` subprocess per run, torn down in ``finally``."""

    def __init__(self, command: tuple[str, ...] | None = None) -> None:
        #: Resolved lazily so importing the package never touches the
        #: filesystem and a missing CLI surfaces at run() time, as a normal
        #: engine error, not at construction.
        self.command = command

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
        invented key exits cleanly and stderr is discarded (``DEVNULL``
        below), so a key a running Codex build does not recognise is a
        silent no-op — the agent keeps Codex's default despite an explicit
        policy, and nothing says so. With it, the same invented key is a
        startup error instead.

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
        sandbox: str = "read-only",
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
        if thread_id:
            ephemeral = False
        command = self._spawn_command(config_overrides)
        # stderr is DEVNULL, not PIPE: nothing ever reads it here, and an
        # unread PIPE deadlocks the App Server once its stderr buffer fills.
        process = await asyncio.create_subprocess_exec(
            *command,
            limit=_STDOUT_LINE_LIMIT,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        assert process.stdin and process.stdout
        pending: dict[int, asyncio.Future[Any]] = {}
        completed: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
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
            await process.stdin.drain()

        async def request(method: str, params: dict[str, Any], *, is_turn: bool = False) -> Any:
            nonlocal counter, turn_request_id
            counter += 1
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
            return await future

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

        async def read_loop() -> None:
            nonlocal turn_id
            assert process.stdout
            try:
                while line := await process.stdout.readline():
                    message = json.loads(line)
                    method = message.get("method")
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
                        params = message["params"]
                        try:
                            result = await on_tool_call(params["tool"], params.get("arguments", {}))
                        except Exception as exc:  # defensive protocol response
                            result = {
                                "success": False,
                                "contentItems": [{"type": "inputText", "text": str(exc)}],
                            }
                        await send({"id": message["id"], "result": result})
                    elif method in {
                        "item/commandExecution/requestApproval",
                        "item/fileChange/requestApproval",
                        "item/permissions/requestApproval",
                    }:
                        params = message.get("params", {})
                        if method == "item/commandExecution/requestApproval":
                            kind, name = "command", str(params.get("command") or "command execution")
                        elif method == "item/fileChange/requestApproval":
                            kind, name = "file_change", str(params.get("grantRoot") or "file changes")
                        else:
                            kind, name = "permissions", "sandbox permissions"
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
                        await send({"id": message["id"], "result": result})
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
                failure = ConnectionError(f"Codex App Server reader failed: {type(exc).__name__}: {exc}")
                failure.__cause__ = exc
                fail_waiters(failure)
            else:
                if pending or not completed.done():
                    fail_waiters(ConnectionError("Codex App Server exited before completing the request"))

        reader = asyncio.create_task(read_loop())
        try:
            await request(
                "initialize",
                {
                    "clientInfo": {"name": "lazybridge", "title": "LazyBridge", "version": "0.1.0"},
                    "capabilities": {"experimentalApi": True},
                },
            )
            await send({"method": "initialized", "params": {}})
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
            if progress is not None:
                # Published before the turn: a durable thread that exists is
                # worth reporting even if this call never returns a result.
                progress["thread_id"] = active_thread
            # ``input`` is the App Server's UserInput union: the text turn plus
            # any image attachments the engine converted. (Unused in native
            # review mode — ``review/start`` has no prompt slot.)
            turn_params: dict[str, Any] = {
                "threadId": active_thread,
                "input": [{"type": "text", "text": prompt}, *(attachments or [])],
            }
            if effort is not None:
                turn_params["effort"] = effort
            try:
                # The uncertainty window opens when the request goes out, not
                # when its response comes back: the server can accept the turn
                # and then drop the connection before answering, and a retry
                # would replay a turn already committed to a durable thread —
                # tool side effects included.
                turn_sent = True
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
                if ephemeral:
                    raise  # nothing survives the subprocess: a retry is clean
                raise CodexTurnUncertain(
                    f"Turn was accepted but its outcome is unknown: {exc}. "
                    f"Resume thread {active_thread} and inspect it before retrying.",
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
            )
        finally:
            reader.cancel()
            try:
                await reader
            except asyncio.CancelledError:
                # Expected: we cancelled the reader on the line above, and
                # awaiting it is only how that cancellation is collected. Any
                # real reader failure has already been routed to the caller
                # through fail_waiters(), so there is nothing to re-raise here
                # — and re-raising would mask the original error.
                pass
            if process.returncode is None:
                process.terminate()
            await process.wait()
