"""Engine-agnostic background delegation primitives.

This extension follows ``docs/guides/core-vs-ext.md``: durable delegation is
an opinionated orchestration pattern layered on LazyBridge's core Agent,
Tool, and Store primitives, not a primitive every agent must carry.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lazybridge import Tool
from lazybridge._display import elide
from lazybridge.ext.delegation.jobs import JobRegistry

# Matches the consultant handles emitted in practice: hex/dashes plus the
# ``repo#id`` shape used by Codex. Deliberately narrower than ``\S+`` so it
# cannot swallow punctuation or unrelated answer text from the first line.
_HANDLE = re.compile(r"[0-9a-zA-Z_#-]+")


def _track(background_tasks: set[asyncio.Task[Any]], coro: Any) -> asyncio.Task[Any]:
    """Create a task and retain it strongly until it finishes.

    The event loop holds only a weak reference to an orphan task. Without a
    strong reference in ``background_tasks``, a fire-and-forget coroutine can
    be garbage-collected mid-run and die before writing a terminal status.
    The callback bounds the set by removing completed tasks.

    Known accepted constraint: every Tool built in this module (via
    :func:`make_background_delegate`, :func:`make_persistent_consultant`,
    :func:`make_parallel_delegate`, :func:`make_plan_delegate`) requires a
    genuinely PERSISTENT host event loop -- the task this creates must
    outlive the coroutine that scheduled it, which returns almost
    immediately by design. Driving one of these tools through
    :meth:`~lazybridge.Tool.run_sync` breaks this: LazyBridge's
    sync/async bridge (``lazybridge._asyncbridge.run_coroutine_blocking``)
    runs the coroutine on a fresh event loop that CANCELS every task
    still pending in its own ``finally`` cleanup as soon as the outer
    coroutine returns -- so the just-scheduled background job would be
    cancelled before it does any real work, while its durable record
    stays stuck at ``"running"``/``"awaiting_approval"``. The same
    fresh-loop-per-call behavior also means an :class:`asyncio.Lock`
    created once at Tool-build time (see
    :func:`make_persistent_consultant`) can become bound to one call's
    loop and then raise on a later call through a different one. This
    mirrors the promoted source's own assumption of one persistent host
    loop for the delegating agent's entire process lifetime -- these
    tools are designed to be awaited via :meth:`~lazybridge.Tool.run`
    from within that loop, never invoked through ``run_sync``. A real
    fix would need fire-and-forget work to live on a dedicated,
    always-on loop independent of whatever loop happens to invoke the
    tool -- a bigger architectural change than this extraction takes on;
    revisit if a caller genuinely needs to invoke these tools from
    ``run_sync``. Found by Codex review before this ever shipped.
    """
    task = asyncio.create_task(coro)
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    return task


def _safe_notify(notify: Callable[[str], None] | None, text: str) -> None:
    """Call ``notify`` without ever letting it fail the delegation caller.

    The starting notification runs synchronously after registration and
    scheduling but before the tool returns. If it escaped, the tool would
    look failed despite a live job and invite a needless duplicate retry.
    """
    if notify is None:
        return
    try:
        notify(text)
    except Exception:
        logging.getLogger(__name__).exception("notify failed for %r", text[:100])


def _engine_identity(engine: Any) -> tuple[str, str | None, str | None]:
    """Best-effort ``(engine class name, model, effort)`` for a job record,
    read directly off whatever ``engine_factory()`` built -- never a
    tool-call argument, since none of the tools in this module thread
    model/effort through their own public signatures today.

    Both :class:`~lazybridge.engines.claude_code.ClaudeCodeEngine` and
    :class:`~lazybridge.engines.codex.CodexEngine` expose ``model`` and
    ``reasoning_effort`` as plain attributes; a caller's own test double
    (or a future engine) need not -- ``getattr`` with a ``None`` default
    makes this purely cosmetic (a richer job record), never a hard
    requirement an engine must satisfy.
    """
    model = getattr(engine, "model", None)
    effort = getattr(engine, "reasoning_effort", None)
    return (
        type(engine).__name__,
        model if isinstance(model, str) else None,
        effort if isinstance(effort, str) else None,
    )


def _result_cost_usd(result: Any) -> float:
    """Direct plus nested cost off a finished :class:`~lazybridge.Agent.run`
    result -- matches LazyPulse's own envelope accounting. ``0.0`` for a
    result shaped without a ``metadata.cost_usd``/``nested_cost_usd`` pair
    (a caller's own test double, most commonly) rather than raising: a
    job's durable record must still get written even when its cost can't
    be read off the result it just produced."""
    metadata = getattr(result, "metadata", None)
    if metadata is None:
        return 0.0
    try:
        return float(getattr(metadata, "cost_usd", 0.0)) + float(getattr(metadata, "nested_cost_usd", 0.0))
    except (TypeError, ValueError):
        return 0.0


async def _run_delegate_job(
    job_id: str,
    objective: str,
    *,
    tool_name: str,
    label: str,
    engine_factory: Callable[[], Any],
    registry: JobRegistry,
    notify: Callable[[str], None] | None,
    plan_id: str | None = None,
    task_index: int | None = None,
    plan_task_text: str | None = None,
    created_at: str | None = None,
) -> None:
    """Run one objective with a fresh engine and persist its outcome.

    A FRESH ``engine_factory()`` call per job is load-bearing for real
    concurrency. ClaudeCodeEngine serializes calls through its session lock,
    so N jobs sharing one engine would still execute one at a time despite N
    surrounding asyncio Tasks; separate instances create real concurrency.

    ``created_at`` is the caller's own record of when the JOB was created
    (not when this coroutine happened to start running it, which can be
    much later on the ``pre_confirm`` path) -- threaded through every write
    this function makes rather than recomputed, so a job's full lifetime is
    visible on its terminal record. Computed fresh (``now``) only if the
    caller genuinely has no earlier timestamp to hand in.
    """
    from lazybridge import Agent

    created_at = created_at or datetime.now(UTC).isoformat()

    try:
        engine = engine_factory()
    except Exception as exc:
        registry.write(
            job_id,
            objective,
            tool_name=tool_name,
            status="failed",
            plan_id=plan_id,
            task_index=task_index,
            plan_task_text=plan_task_text,
            error=f"engine_factory failed: {exc}",
            execution_started=False,
            created_at=created_at,
            finished_at=datetime.now(UTC).isoformat(),
        )
        _safe_notify(notify, f"{label} job {job_id[:8]} FAILED before it started: {objective[:100]}\n\n{exc}")
        return

    engine_name, model, effort = _engine_identity(engine)
    # Crossing this write is the durable boundary between "queued/approved"
    # and "a worker is actually running, may already be spending money" --
    # execution_started=True from here on, cost_unknown=True until a result
    # envelope exists to read a real cost from.
    registry.write(
        job_id,
        objective,
        tool_name=tool_name,
        status="running",
        plan_id=plan_id,
        task_index=task_index,
        plan_task_text=plan_task_text,
        execution_started=True,
        engine=engine_name,
        model=model,
        effort=effort,
        created_at=created_at,
        cost_unknown=True,
    )

    try:
        worker = Agent(engine=engine, name=f"delegate-{tool_name}-{job_id[:8]}")
        result = await worker.run(objective)
    except Exception as exc:
        registry.write(
            job_id,
            objective,
            tool_name=tool_name,
            status="failed",
            plan_id=plan_id,
            task_index=task_index,
            plan_task_text=plan_task_text,
            error=str(exc),
            execution_started=True,
            engine=engine_name,
            model=model,
            effort=effort,
            created_at=created_at,
            finished_at=datetime.now(UTC).isoformat(),
            # No result envelope exists to read a real cost from -- stays
            # genuinely unknown rather than assumed zero.
            cost_unknown=True,
        )
        _safe_notify(notify, f"{label} job {job_id[:8]} FAILED: {objective[:100]}\n\n{exc}")
        return

    finished_at = datetime.now(UTC).isoformat()
    cost_usd = _result_cost_usd(result)
    if result.ok:
        text = result.text()
        registry.write(
            job_id,
            objective,
            tool_name=tool_name,
            status="done",
            plan_id=plan_id,
            task_index=task_index,
            plan_task_text=plan_task_text,
            result=text,
            execution_started=True,
            engine=engine_name,
            model=model,
            effort=effort,
            created_at=created_at,
            finished_at=finished_at,
            cost_usd=cost_usd,
            cost_unknown=False,
        )
        preview = elide(text)
        _safe_notify(notify, f"{label} job {job_id[:8]} done: {objective[:100]}\n\n{preview}")
    else:
        message = result.error.message if result.error else "unknown error"
        # Most engine-level failures route through Envelope.error_envelope(),
        # which never attaches real usage -- metadata stays at its all-zero
        # default regardless of how much billable work ran before the
        # failure. A computed cost of exactly 0.0 on a FAILED result is
        # therefore ambiguous (a genuine free failure -- a guard blocking
        # before any call, e.g. -- looks identical to "never measured") and
        # is recorded as unknown rather than a confident zero. A NONZERO
        # cost can only be genuine measured usage (surfaced by a failure
        # path that preserves its own envelope's metadata, e.g. an
        # output-validation failure built via `envelope.model_copy(update=
        # {"error": ...})`), so that is still recorded as known. Found by
        # Codex review before this ever shipped.
        cost_known = cost_usd != 0.0
        registry.write(
            job_id,
            objective,
            tool_name=tool_name,
            status="failed",
            plan_id=plan_id,
            task_index=task_index,
            plan_task_text=plan_task_text,
            error=message,
            execution_started=True,
            engine=engine_name,
            model=model,
            effort=effort,
            created_at=created_at,
            finished_at=finished_at,
            cost_usd=cost_usd if cost_known else None,
            cost_unknown=not cost_known,
        )
        _safe_notify(notify, f"{label} job {job_id[:8]} FAILED: {objective[:100]}\n\n{message}")


def make_background_delegate(
    *,
    tool_name: str,
    label: str,
    engine_factory: Callable[[], Any],
    registry: JobRegistry,
    background_tasks: set[asyncio.Task[Any]],
    notify: Callable[[str], None] | None,
    doc: str,
    pre_confirm: Callable[[str], Awaitable[bool]] | None = None,
    validate_model: Callable[[str | None], str | None] | None = None,
    model: str | None = None,
    admission_gate: Callable[[], Awaitable[Any]] | None = None,
) -> Tool:
    """Build a fire-and-forget delegate with durable status reporting.

    Fire-and-forget is intentional: with ``max_concurrent_inbound=1``, an
    in-turn await on a real delegated task (often the longest call an agent
    makes) blocks every other message. ``pre_confirm`` exists because once a
    sub-agent's own actions are not individually gated, the reliable human
    decision point is before the whole delegated objective starts.

    ``validate_model`` and ``model`` together let a caller reject a model
    its OWN ``engine_factory`` was built to use (e.g. a non-Anthropic model
    handed to a Claude Code engine) before anything is recorded or spawned
    -- ``model`` is not a per-call argument on the ``delegate`` tool this
    builds (the engine a caller wants is already fixed by whatever
    ``engine_factory`` closes over); it exists purely so this validation can
    run against the SAME value, once, up front. This package carries no
    opinion about which models are valid for which engine -- that policy
    belongs to the caller, passed in as ``validate_model``.

    ``admission_gate``, when given, is awaited exactly once, on the
    ``pre_confirm`` path only, right after a human approves and right
    before the engine actually starts -- never before ``pre_confirm`` is
    asked (a quota/rate check run there would be re-validated against a
    picture of the world that can be hours stale by the time an unbounded
    human wait finally resolves) and never on the no-``pre_confirm`` path
    (there is no wait to re-check anything across there). A rejection is
    recorded as ``status="failed"`` with ``execution_started=False`` --
    approved by the human, refused before any real work started, no tokens
    spent. This package carries no quota/admission POLICY of its own
    (``admission_gate`` is any zero-argument async callable returning
    either ``None`` -- treated as "allowed" -- or an object exposing
    ``.allowed: bool`` and, when denying, a human-readable ``.reason``);
    that policy is entirely the caller's.
    """

    async def _run_job(job_id: str, objective: str, created_at: str) -> None:
        if pre_confirm is not None:
            try:
                approved = await pre_confirm(objective)
            except Exception as exc:
                # A raised pre_confirm (approval channel disconnected,
                # timed out, ...) must still leave the job at a TERMINAL
                # status -- the initial write above already left it at
                # "awaiting_approval", and nothing else ever revisits a
                # background job outside this function, so an uncaught
                # exception here would show it stuck "awaiting_approval"
                # forever even though no approval is actually pending
                # anymore. Found by Codex review before this ever shipped.
                registry.write(
                    job_id,
                    objective,
                    tool_name=tool_name,
                    status="failed",
                    error=f"pre_confirm failed: {exc}",
                    execution_started=False,
                    created_at=created_at,
                    finished_at=datetime.now(UTC).isoformat(),
                )
                _safe_notify(notify, f"{label} job {job_id[:8]} FAILED before it started: {objective[:100]}\n\n{exc}")
                return
            if not approved:
                registry.write(
                    job_id,
                    objective,
                    tool_name=tool_name,
                    status="denied",
                    error="denied by approver",
                    execution_started=False,
                    created_at=created_at,
                    finished_at=datetime.now(UTC).isoformat(),
                )
                _safe_notify(notify, f"{label} job {job_id[:8]} DENIED before it started: {objective[:100]}")
                return
            if admission_gate is not None:
                # THE admission, not a preflight -- a human-approved delegation
                # can sit waiting, unbounded, for far longer than any reservation
                # lives, so the only point at which checking admission means
                # anything is right here: approved, not yet executing.
                try:
                    admission = await admission_gate()
                except Exception as exc:
                    registry.write(
                        job_id,
                        objective,
                        tool_name=tool_name,
                        status="failed",
                        error=f"admission_gate failed: {exc}",
                        execution_started=False,
                        created_at=created_at,
                        finished_at=datetime.now(UTC).isoformat(),
                    )
                    _safe_notify(
                        notify,
                        f"{label} job {job_id[:8]} was approved, but the admission check itself failed before "
                        f"it started: {exc}",
                    )
                    return
                if admission is not None and not getattr(admission, "allowed", True):
                    reason = getattr(admission, "reason", None) or "rejected by admission_gate"
                    registry.write(
                        job_id,
                        objective,
                        tool_name=tool_name,
                        status="failed",
                        error=str(reason),
                        execution_started=False,
                        created_at=created_at,
                        finished_at=datetime.now(UTC).isoformat(),
                    )
                    _safe_notify(
                        notify,
                        f"{label} job {job_id[:8]} was approved, but admission refused it by the time approval "
                        f"arrived: {reason}. Nothing was started.",
                    )
                    return
            registry.write(
                job_id, objective, tool_name=tool_name, status="running", execution_started=False, created_at=created_at
            )

        await _run_delegate_job(
            job_id,
            objective,
            tool_name=tool_name,
            label=label,
            engine_factory=engine_factory,
            registry=registry,
            notify=notify,
            created_at=created_at,
        )

    async def delegate(objective: str) -> str:
        # Before anything else: no job record, no approval asked. A model
        # this engine can never run is refused for free, same discipline
        # applied to every other cheap-to-check precondition in this
        # package.
        if validate_model is not None:
            rejection = validate_model(model)
            if rejection is not None:
                return rejection
        job_id = str(uuid.uuid4())
        created_at = datetime.now(UTC).isoformat()
        initial_status = "awaiting_approval" if pre_confirm is not None else "running"
        registry.write(job_id, objective, tool_name=tool_name, status=initial_status, created_at=created_at)
        _track(background_tasks, _run_job(job_id, objective, created_at))
        preview = elide(objective)
        if pre_confirm is not None:
            _safe_notify(
                notify,
                f"Requesting approval to delegate to {label} (job {job_id[:8]}, real write access): {preview}",
            )
            return (
                f"Requested approval for job {job_id[:8]} -- {label} will start once you approve. "
                "check_jobs() for status."
            )
        _safe_notify(notify, f"Delegating to {label} (job {job_id[:8]}, real write access): {preview}")
        return (
            f"Started job {job_id[:8]} -- {label} is working on this in the background, not blocking you. "
            "You'll get a notification when it finishes; check_jobs() shows status/results anytime."
        )

    delegate.__doc__ = doc
    return Tool.wrap(delegate, name=tool_name)


def make_persistent_consultant(
    build: Any,
    *,
    tool_name: str,
    registry: JobRegistry,
    background_tasks: set[asyncio.Task[Any]],
    notify: Callable[[str], None] | None = None,
    doc_suffix: str = "",
) -> Tool:
    """Build one serialized, handle-remembering consultant conversation.

    The lock guards the background job rather than the tool call, preserving
    low turn latency. Without it, overlapping calls can both read the same
    stale handle, open separate conversations, and leave only the final
    reply's handle remembered.
    """
    inner = build()
    params = inspect.signature(inner.func).parameters
    if "thread_id" in params:
        id_field = "thread_id"
    elif "session_id" in params:
        id_field = "session_id"
    else:
        raise TypeError(
            f"{tool_name}: wrapped consultant function has neither a thread_id nor a session_id "
            f"parameter (found: {list(params)}) -- consultant API may have changed"
        )
    state: dict[str, str | None] = {"handle": None}
    lock = asyncio.Lock()

    async def _run_job(job_id: str, question: str) -> None:
        async with lock:
            try:
                result = await inner.func(question=question, **{id_field: state["handle"]})
            except Exception as exc:
                registry.write(job_id, question, tool_name=tool_name, status="failed", error=str(exc))
                _safe_notify(notify, f"{tool_name} job {job_id[:8]} FAILED: {question[:100]}\n\n{exc}")
                return
            # Search only the header: the answer body may itself mention a
            # thread_id/session_id that is unrelated to the emitted handle.
            header = result.split("\n", 1)[0]
            match = re.search(rf"{id_field}=({_HANDLE.pattern})", header)
            if match:
                state["handle"] = match.group(1)
        registry.write(job_id, question, tool_name=tool_name, status="done", result=result)
        preview = elide(result)
        _safe_notify(notify, f"{tool_name} job {job_id[:8]} done: {question[:100]}\n\n{preview}")

    async def ask(question: str) -> str:
        job_id = str(uuid.uuid4())
        registry.write(job_id, question, tool_name=tool_name, status="running")
        _track(background_tasks, _run_job(job_id, question))
        preview = elide(question)
        _safe_notify(notify, f"Consulting {tool_name} (job {job_id[:8]}): {preview}")
        return (
            f"Started job {job_id[:8]} -- {tool_name} is answering in the background, not blocking you. "
            "check_jobs() for the answer when it's ready."
        )

    ask.__doc__ = (inner.func.__doc__ or "") + " " + doc_suffix
    return Tool.wrap(ask, name=tool_name)


def make_claude_delegate_engine_factory(*, workspace_root: Path, gate: Any, model: str) -> Callable[[], Any]:
    """Build a factory that creates one gated Claude Code engine per job."""
    from lazybridge.engines.coding import ClaudeCodePolicy, CodingAgentConfig

    config = CodingAgentConfig(
        claude=ClaudeCodePolicy(
            permission_mode="default",
            preapprove_application_tools=False,
            extra_tools=("Write", "Edit", "Bash"),
        ),
        approval_gate=gate,
    )

    def _engine_factory() -> Any:
        from lazybridge.engines.claude_code import ClaudeCodeEngine

        return ClaudeCodeEngine(model=model, cwd=str(workspace_root), config=config, request_timeout=None)

    return _engine_factory


def _resolve_engine_factory(
    engine_factory: Callable[[], Any] | None,
    *,
    workspace_root: Path | None,
    gate: Any,
    model: str,
) -> Callable[[], Any]:
    if engine_factory is not None:
        return engine_factory
    if workspace_root is None or gate is None:
        raise ValueError("workspace_root and gate are required when engine_factory is not provided")
    return make_claude_delegate_engine_factory(workspace_root=workspace_root, gate=gate, model=model)


async def _admission_rejection_message(
    admission_gate: Callable[[], Awaitable[Any]], *, index: int, total: int
) -> str | None:
    """One admission check for one objective in a batch -- ``None`` means
    allowed. See :func:`make_parallel_delegate`'s own docstring for why a
    batch stops at the FIRST refusal rather than admitting some objectives
    and refusing others."""
    admission = await admission_gate()
    if admission is None or getattr(admission, "allowed", True):
        return None
    reason = getattr(admission, "reason", None) or "rejected by admission_gate"
    return (
        f"REJECTED: {reason} (refused at objective {index} of {total}; "
        "the whole batch was held back rather than started in part)"
    )


def make_parallel_delegate(
    *,
    engine_factory: Callable[[], Any] | None = None,
    registry: JobRegistry,
    background_tasks: set[asyncio.Task[Any]],
    notify: Callable[[str], None] | None = None,
    doc: str,
    max_parallel_objectives: int = 8,
    max_in_flight_delegate_tasks: int = 12,
    workspace_root: Path | None = None,
    gate: Any = None,
    model: str = "sonnet",
    admission_gate: Callable[[], Awaitable[Any]] | None = None,
) -> Tool:
    """Build a capped, fire-and-forget parallel delegation tool.

    The per-call cap bounds the resource commitment one LLM tool call can
    make. The in-flight cap is deliberately process-local and cooperative,
    not a fleet-wide guarantee; it coordinates only callers sharing this
    ``background_tasks`` set.

    ``admission_gate``, when given, is checked ONCE PER OBJECTIVE, not once
    for the whole batch -- this tool is exactly the shape an admission
    policy exists for: N engines started from one decision. Checking once
    for the batch would let a single admission buy every objective's worth
    of whatever that policy limits; skipping the per-objective check
    entirely would let a fan-out start above a ceiling a single call would
    have respected. The checks run BEFORE any objective is scheduled, and
    the first refusal aborts the WHOLE batch -- nothing partially admitted
    is started, so a batch too large for the current admission state needs
    a smaller batch, not a partial one nobody asked for. This mirrors
    LazyCEO's own ``run_parallel`` batch-rollback behaviour; unlike that
    source, ``admission_gate`` here is a plain zero-argument callable with
    no reservation to take out and refund on a later refusal in the SAME
    batch -- this package has no quota/reservation model of its own, so
    there is nothing to roll back beyond simply not starting anything yet.

    ``run_parallel`` itself must be a COROUTINE function, not a plain one:
    :class:`~lazybridge.Tool` dispatches a synchronous tool's function
    through ``loop.run_in_executor``, on a worker thread with no running
    event loop -- ``asyncio.create_task`` inside ``_track`` would raise
    there. Declaring it ``async`` routes execution through the caller's
    own event loop instead, where a task can actually be created. Found
    by Codex review before this ever shipped.
    """
    resolved_factory = _resolve_engine_factory(engine_factory, workspace_root=workspace_root, gate=gate, model=model)

    async def run_parallel(objectives: list[str]) -> str:
        if not objectives:
            return "REJECTED: objectives is empty -- nothing to run"
        if len(objectives) > max_parallel_objectives:
            return f"REJECTED: {len(objectives)} objectives exceeds the cap of {max_parallel_objectives} per call"
        if len(background_tasks) + len(objectives) > max_in_flight_delegate_tasks:
            return (
                f"REJECTED: {len(background_tasks)} job(s) already in flight plus {len(objectives)} new objective(s) "
                f"exceeds the process-local cap of {max_in_flight_delegate_tasks}"
            )

        if admission_gate is not None:
            for index in range(1, len(objectives) + 1):
                rejection = await _admission_rejection_message(admission_gate, index=index, total=len(objectives))
                if rejection is not None:
                    return rejection
            # Re-checked immediately before scheduling, not just once at
            # the top of this call: admission_gate's own await can suspend
            # for real time, during which a CONCURRENT run_parallel call on
            # this same background_tasks set can pass the exact same
            # capacity check and also proceed, jointly exceeding
            # max_in_flight_delegate_tasks even though neither call's own
            # check ever saw a violation. This narrows that window
            # (there is still a gap between this read and the scheduling
            # loop below) rather than closing it outright -- the module's
            # own docstring already documents this cap as "process-local
            # and cooperative, not a fleet-wide guarantee"; a fully atomic
            # reservation would need a lock around the whole
            # check-then-schedule sequence, a bigger change than this
            # extraction takes on. Found by Codex review before this ever
            # shipped.
            if len(background_tasks) + len(objectives) > max_in_flight_delegate_tasks:
                return (
                    f"REJECTED: {len(background_tasks)} job(s) already in flight plus {len(objectives)} new "
                    f"objective(s) exceeds the process-local cap of {max_in_flight_delegate_tasks} -- capacity "
                    "was taken by another call while admission was being checked"
                )

        job_ids: list[str] = []
        for objective in objectives:
            job_id = str(uuid.uuid4())
            created_at = datetime.now(UTC).isoformat()
            registry.write(job_id, objective, tool_name="run_parallel", status="running", created_at=created_at)
            _track(
                background_tasks,
                _run_delegate_job(
                    job_id,
                    objective,
                    tool_name="run_parallel",
                    label="a parallel Claude Code sub-agent",
                    engine_factory=resolved_factory,
                    registry=registry,
                    notify=notify,
                    created_at=created_at,
                ),
            )
            job_ids.append(job_id[:8])

        preview = ", ".join(job_ids)
        _safe_notify(notify, f"Started {len(objectives)} parallel Claude Code sub-agent(s): {preview}")
        return (
            f"Started {len(objectives)} job(s) in parallel ({preview}) -- not blocking you, each is its own "
            "sub-agent with real write access. check_jobs() shows status/results as each one finishes."
        )

    run_parallel.__doc__ = doc
    return Tool.wrap(run_parallel, name="run_parallel")


def make_plan_delegate(
    *,
    engine_factory: Callable[[], Any] | None = None,
    registry: JobRegistry,
    background_tasks: set[asyncio.Task[Any]],
    board: Any,
    owner: str,
    notify: Callable[[str], None] | None = None,
    doc: str | None = None,
    max_parallel_objectives: int = 8,
    max_in_flight_delegate_tasks: int = 12,
    workspace_root: Path | None = None,
    gate: Any = None,
    model: str = "sonnet",
) -> Tool:
    """Build a tool that claims durable-plan tasks before delegation.

    Like ``run_parallel``, this must be a coroutine function -- see that
    function's docstring for why a synchronous one would break
    ``asyncio.create_task`` inside ``_track`` when dispatched through
    :class:`~lazybridge.Tool`'s executor path for plain functions.

    Does not accept ``run_parallel``'s own ``admission_gate`` -- a plan
    task's claim (``board.claim_task``) and its job record are written
    together per item, inside one loop, with no separate pre-pass that
    checks every item before any of them starts; retrofitting the same
    per-objective check-then-start-nothing-on-refusal shape here would mean
    unwinding already-claimed board tasks on a later refusal, which this
    extraction does not take on. Revisit together if a caller needs both.
    """
    resolved_factory = _resolve_engine_factory(engine_factory, workspace_root=workspace_root, gate=gate, model=model)
    plan_id = board.plan_id

    async def delegate_plan_tasks(delegations: list[dict[str, Any]]) -> str:
        if not delegations:
            return "REJECTED: delegations is empty -- nothing to run"
        if len(delegations) > max_parallel_objectives:
            return f"REJECTED: {len(delegations)} delegations exceeds the cap of {max_parallel_objectives} per call"
        if len(background_tasks) + len(delegations) > max_in_flight_delegate_tasks:
            return (
                f"REJECTED: {len(background_tasks)} job(s) already in flight plus {len(delegations)} new delegation(s) "
                f"exceeds the process-local cap of {max_in_flight_delegate_tasks}"
            )

        # Validate the entire batch before the first claim. Capacity/type
        # rejection must never leave a partial durable-plan mutation behind.
        outcomes: list[str | None] = [None] * len(delegations)
        validated: list[tuple[int, str, str] | None] = [None] * len(delegations)
        for item_number, delegation in enumerate(delegations, start=1):
            task_index = delegation.get("task_index")
            expected_text = delegation.get("expected_text")
            objective = delegation.get("objective")
            if not isinstance(task_index, int) or isinstance(task_index, bool):
                outcomes[item_number - 1] = (
                    f"- item {item_number}: REJECTED: task_index must be an int (bool is not accepted)"
                )
                continue
            if not isinstance(expected_text, str) or not expected_text.strip():
                outcomes[item_number - 1] = f"- item {item_number}: REJECTED: expected_text must be a non-empty string"
                continue
            if not isinstance(objective, str) or not objective.strip():
                outcomes[item_number - 1] = f"- item {item_number}: REJECTED: objective must be a non-empty string"
                continue
            validated[item_number - 1] = (task_index, expected_text, objective)

        for item_number, item in enumerate(validated, start=1):
            if item is None:
                continue
            task_index, expected_text, objective = item
            job_id = str(uuid.uuid4())
            claimed = board.claim_task(task_index, expected_text, owner=owner)
            if isinstance(claimed, str):
                outcomes[item_number - 1] = f"- item {item_number}: {claimed}"
                continue

            coroutine = None
            try:
                created_at = datetime.now(UTC).isoformat()
                registry.write(
                    job_id,
                    objective,
                    tool_name="delegate_plan_tasks",
                    status="running",
                    plan_id=plan_id,
                    task_index=task_index,
                    plan_task_text=expected_text,
                    created_at=created_at,
                )
                coroutine = _run_delegate_job(
                    job_id,
                    objective,
                    tool_name="delegate_plan_tasks",
                    label="a plan-linked Claude Code sub-agent",
                    engine_factory=resolved_factory,
                    registry=registry,
                    notify=notify,
                    plan_id=plan_id,
                    task_index=task_index,
                    plan_task_text=expected_text,
                    created_at=created_at,
                )
                _track(background_tasks, coroutine)
            except Exception as exc:
                if coroutine is not None:
                    coroutine.close()
                error = f"delegate_plan_tasks setup failed: {exc}"
                board.mark_failed(task_index, error, owner=owner)
                # Releasing the claim is only half the cleanup. The initial
                # record already says "running", and no coroutine now exists
                # to transition it; startup reclamation does not run here.
                registry.write(
                    job_id,
                    objective,
                    tool_name="delegate_plan_tasks",
                    status="failed",
                    plan_id=plan_id,
                    task_index=task_index,
                    plan_task_text=expected_text,
                    error=error,
                )
                outcomes[item_number - 1] = f"- item {item_number}: setup failed for task {task_index}: {exc}"
                continue

            outcomes[item_number - 1] = f"- item {item_number}: started job {job_id[:8]} for task {task_index}"

        return "\n".join(outcome for outcome in outcomes if outcome is not None)

    delegate_plan_tasks.__doc__ = doc or (
        "Claim and delegate specific plan tasks. Each item requires task_index, "
        "expected_text matching the current plan, and a self-contained objective."
    )
    return Tool.wrap(delegate_plan_tasks, name="delegate_plan_tasks")
