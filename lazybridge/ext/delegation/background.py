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
from pathlib import Path
from typing import Any

from lazybridge import Tool
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
) -> None:
    """Run one objective with a fresh engine and persist its outcome.

    A FRESH ``engine_factory()`` call per job is load-bearing for real
    concurrency. ClaudeCodeEngine serializes calls through its session lock,
    so N jobs sharing one engine would still execute one at a time despite N
    surrounding asyncio Tasks; separate instances create real concurrency.
    """
    from lazybridge import Agent

    try:
        worker = Agent(engine=engine_factory(), name=f"delegate-{tool_name}-{job_id[:8]}")
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
        )
        _safe_notify(notify, f"{label} job {job_id[:8]} FAILED: {objective[:100]}\n\n{exc}")
        return

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
        )
        preview = text if len(text) <= 500 else text[:497] + "..."
        _safe_notify(notify, f"{label} job {job_id[:8]} done: {objective[:100]}\n\n{preview}")
    else:
        message = result.error.message if result.error else "unknown error"
        registry.write(
            job_id,
            objective,
            tool_name=tool_name,
            status="failed",
            plan_id=plan_id,
            task_index=task_index,
            plan_task_text=plan_task_text,
            error=message,
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
) -> Tool:
    """Build a fire-and-forget delegate with durable status reporting.

    Fire-and-forget is intentional: with ``max_concurrent_inbound=1``, an
    in-turn await on a real delegated task (often the longest call an agent
    makes) blocks every other message. ``pre_confirm`` exists because once a
    sub-agent's own actions are not individually gated, the reliable human
    decision point is before the whole delegated objective starts.
    """

    async def _run_job(job_id: str, objective: str) -> None:
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
                )
                _safe_notify(notify, f"{label} job {job_id[:8]} DENIED before it started: {objective[:100]}")
                return
            registry.write(job_id, objective, tool_name=tool_name, status="running")

        await _run_delegate_job(
            job_id,
            objective,
            tool_name=tool_name,
            label=label,
            engine_factory=engine_factory,
            registry=registry,
            notify=notify,
        )

    async def delegate(objective: str) -> str:
        job_id = str(uuid.uuid4())
        initial_status = "awaiting_approval" if pre_confirm is not None else "running"
        registry.write(job_id, objective, tool_name=tool_name, status=initial_status)
        _track(background_tasks, _run_job(job_id, objective))
        preview = objective if len(objective) <= 300 else objective[:297] + "..."
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
        preview = result if len(result) <= 500 else result[:497] + "..."
        _safe_notify(notify, f"{tool_name} job {job_id[:8]} done: {question[:100]}\n\n{preview}")

    async def ask(question: str) -> str:
        job_id = str(uuid.uuid4())
        registry.write(job_id, question, tool_name=tool_name, status="running")
        _track(background_tasks, _run_job(job_id, question))
        preview = question if len(question) <= 200 else question[:197] + "..."
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
) -> Tool:
    """Build a capped, fire-and-forget parallel delegation tool.

    The per-call cap bounds the resource commitment one LLM tool call can
    make. The in-flight cap is deliberately process-local and cooperative,
    not a fleet-wide guarantee; it coordinates only callers sharing this
    ``background_tasks`` set.

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

        job_ids: list[str] = []
        for objective in objectives:
            job_id = str(uuid.uuid4())
            registry.write(job_id, objective, tool_name="run_parallel", status="running")
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
                registry.write(
                    job_id,
                    objective,
                    tool_name="delegate_plan_tasks",
                    status="running",
                    plan_id=plan_id,
                    task_index=task_index,
                    plan_task_text=expected_text,
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
