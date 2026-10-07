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
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lazybridge import Tool
from lazybridge._display import elide
from lazybridge.ext.delegation.admission import (
    refund_admission,
    release_admission,
)
from lazybridge.ext.delegation.admission import (
    rejection_text as _admission_rejection_text,
)
from lazybridge.ext.delegation.jobs import JobRegistry

# Matches the consultant handles emitted in practice: hex/dashes plus the
# ``repo#id`` shape used by Codex. Deliberately narrower than ``\S+`` so it
# cannot swallow punctuation or unrelated answer text from the first line.
_HANDLE = re.compile(r"[0-9a-zA-Z_#-]+")

#: Names :func:`make_background_delegate` resolves itself (``model``,
#: ``effort``) when their matching ``accept_*_override`` flag is set --
#: reserved so a caller's own ``extra_params`` cannot collide with them and
#: silently fight over who sets the engine_factory kwarg of that name.
_RESERVED_EXTRA_PARAM_NAMES = frozenset({"model", "effort"})


@dataclass(frozen=True)
class ExtraParam:
    """One additional per-call parameter a generated delegate tool accepts
    beyond ``objective``, exposed to the LLM with its own JSON Schema entry
    and forwarded verbatim as a keyword argument to ``engine_factory`` (and
    to ``guard``, if one is given) on every call.

    This is the generic answer to a caller whose own ``engine_factory``
    needs something resolved per call -- LazyCEO's motivating case was a
    ``repo`` argument, checked against its own policy on every call before
    anything is recorded or spawned (see ``guard`` on
    :func:`make_background_delegate`). This package has no opinion about
    what the parameter means: it only knows its JSON Schema type,
    description, and whether it is required.
    """

    #: JSON Schema ``"type"`` for this parameter, e.g. ``"string"``,
    #: ``"integer"``, ``"boolean"``.
    json_type: str = "string"
    description: str = ""
    required: bool = False


async def _maybe_await_value(value: Any) -> Any:
    """Await ``value`` if it is awaitable, otherwise pass it through.

    Lets a caller's ``guard`` hook be either a plain sync callable or an
    async one without this package needing to know which -- the same
    "maybe sync, maybe async" accommodation
    :func:`lazybridge.ext.delegation.admission._maybe_await` makes for
    ``release``/``refund``, kept local here rather than imported since
    that one is a private helper of a different module.
    """
    if inspect.isawaitable(value):
        return await value
    return value


async def _run_with_admission_release(coro: Any, admission: Any) -> None:
    """Run ``coro`` to completion, then release its admission however it
    ends -- success, failure, or cancellation.

    Shared by :func:`make_parallel_delegate` and :func:`make_plan_delegate`:
    both schedule one ``_run_delegate_job`` coroutine per objective/task
    under its own admission grant, and both need that reservation given
    back once the job is actually done, not held until some external TTL.
    A ``try/finally`` around the await covers cancellation too (the task
    this wraps is tracked in ``background_tasks`` and can be cancelled from
    outside).
    """
    try:
        await coro
    finally:
        await release_admission(admission)


def _schedule_with_admission_release(
    background_tasks: set[asyncio.Task[Any]],
    coro: Any,
    admission: Any,
    *,
    track: Callable[[set[asyncio.Task[Any]], Any], Any] | None = None,
) -> None:
    """Schedule ``coro`` wrapped in :func:`_run_with_admission_release`,
    owning BOTH coroutines if the scheduling call itself raises.

    The wrapper coroutine is created before ``track`` (``_track`` by
    default) gets a chance to consume it -- a ``track`` that raises
    (a broken scheduler, exercised by this package's own tests) leaves it
    un-awaited, which is a ``RuntimeWarning`` on every such failure.
    Closing the wrapper first, then the still-unstarted inner job
    coroutine, silences both without running either body. Mirrors the
    promoted source's own ``_start_released``/``_releasing`` pattern.
    """
    wrapper = _run_with_admission_release(coro, admission)
    try:
        (track or _track)(background_tasks, wrapper)
    except BaseException:
        wrapper.close()
        coro.close()  # closing an already-closed coroutine is a no-op
        raise


def _build_delegate_schema(
    *,
    accept_model_override: bool,
    accept_effort_override: bool,
    extra_params: Mapping[str, ExtraParam] | None,
) -> dict[str, Any]:
    """Hand-built JSON Schema for the dynamic-parameter ``delegate`` tool.

    Built explicitly (via :meth:`~lazybridge.Tool.from_schema`) rather than
    through signature introspection: the real Python callable underneath
    stays a stable ``(objective, **extra)``, with ``extra``'s *shape*
    varying per call site -- exactly the case ``Tool.from_schema``'s own
    docstring names ("signature introspection would ... produce the wrong
    shape").
    """
    properties: dict[str, Any] = {
        "objective": {"type": "string", "description": "The self-contained task to delegate."}
    }
    required = ["objective"]
    if accept_model_override:
        properties["model"] = {"type": "string", "description": "Override the model for this call only."}
    if accept_effort_override:
        properties["effort"] = {"type": "string", "description": "Override the reasoning effort for this call only."}
    for name, spec in (extra_params or {}).items():
        properties[name] = {"type": spec.json_type, "description": spec.description}
        if spec.required:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


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
    engine_factory: Callable[..., Any],
    registry: JobRegistry,
    background_tasks: set[asyncio.Task[Any]],
    notify: Callable[[str], None] | None,
    doc: str,
    pre_confirm: Callable[[str], Awaitable[bool]] | None = None,
    validate_model: Callable[[str | None], str | None] | None = None,
    model: str | None = None,
    effort: str | None = None,
    admission_gate: Callable[[], Awaitable[Any]] | None = None,
    accept_model_override: bool = False,
    accept_effort_override: bool = False,
    extra_params: Mapping[str, ExtraParam] | None = None,
    guard: Callable[[str, dict[str, Any]], Any] | None = None,
) -> Tool:
    """Build a fire-and-forget delegate with durable status reporting.

    Fire-and-forget is intentional: with ``max_concurrent_inbound=1``, an
    in-turn await on a real delegated task (often the longest call an agent
    makes) blocks every other message. ``pre_confirm`` exists because once a
    sub-agent's own actions are not individually gated, the reliable human
    decision point is before the whole delegated objective starts.

    ``validate_model`` and ``model`` together let a caller reject a model
    its OWN ``engine_factory`` was built to use (e.g. a non-Anthropic model
    handed to a Claude Code engine) before anything is recorded or spawned.
    By default ``model`` is not a per-call argument on the ``delegate`` tool
    this builds (the engine a caller wants is already fixed by whatever
    ``engine_factory`` closes over); it exists purely so this validation can
    run against the SAME value, once, up front. Passing
    ``accept_model_override=True`` (and likewise ``accept_effort_override``
    for ``effort``) exposes a per-call override instead -- see below. This
    package carries no opinion about which models are valid for which
    engine -- that policy belongs to the caller, passed in as
    ``validate_model``.

    ``accept_model_override``/``accept_effort_override``, when set, add an
    optional ``model``/``effort`` parameter to the generated ``delegate``
    tool. A call that omits it falls back to this function's own
    ``model``/``effort`` default -- the override is additive, never a
    requirement to pass one every call. The RESOLVED value (override, or
    the default) is what ``validate_model`` is run against and what is
    passed to ``engine_factory`` as a ``model=``/``effort=`` keyword (see
    ``extra_params`` below for how that call is shaped). This is the
    generic answer to a caller (LazyCEO's ``codex_write``/``claude_write``)
    whose own sub-agent engine needs a different model or reasoning effort
    on a given call, not only the one fixed at tool-build time.

    ``extra_params`` declares further per-call parameters beyond
    ``objective``/``model``/``effort`` -- LazyCEO's own motivating case is
    a ``repo`` argument, resolved and checked against a policy on every
    call. Each is forwarded verbatim as a keyword argument to
    ``engine_factory`` (which must then accept it) and, together with the
    resolved ``model``/``effort``, to ``guard``. When none of
    ``accept_model_override``, ``accept_effort_override``, or
    ``extra_params`` is set (the default), ``engine_factory`` is still
    called with zero arguments, exactly as before -- the generated tool's
    signature and JSON Schema are BYTE-IDENTICAL to the pre-1.8 shape. Once
    any of them is set, the tool is instead built via
    :meth:`~lazybridge.Tool.from_schema` with an explicit schema (see
    ``_build_delegate_schema``), because the real Python callable
    underneath has to accept an arbitrary ``**extra`` whose shape varies
    per call site -- signature introspection cannot describe that.

    ``guard``, when given, is called with ``(objective, call_kwargs)``
    -- ``call_kwargs`` being exactly what will be passed to
    ``engine_factory`` if this call proceeds (the resolved ``model``/
    ``effort`` when their overrides are enabled, plus every ``extra_params``
    value) -- and runs FIRST, before ``validate_model``, before
    ``pre_confirm``, before any job record is written or any task spawned.
    Returning a non-``None`` string rejects the call with that string
    (the same "REJECTED: ..." convention every other free precondition in
    this package uses); returning ``None`` allows it through. May be a sync
    or an async callable. This is the generic per-call validation hook a
    caller's own policy (e.g. LazyCEO's project-work gate) runs through --
    this package has no opinion about what it checks.

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
    either ``None`` -- treated as "allowed" -- or an
    :class:`~lazybridge.ext.delegation.admission.AdmissionDecision`-shaped
    object); that policy is entirely the caller's. When granted, the
    admission is released (via
    :func:`~lazybridge.ext.delegation.admission.release_admission`) exactly
    once, in a ``finally`` around the job's actual run, regardless of how
    it ends -- a reservation is never held past the attempt it was taken
    out for.
    """
    dynamic = accept_model_override or accept_effort_override or bool(extra_params)
    if extra_params and _RESERVED_EXTRA_PARAM_NAMES & extra_params.keys():
        raise ValueError(
            f"extra_params must not use the reserved names {sorted(_RESERVED_EXTRA_PARAM_NAMES)} -- "
            "those are resolved by accept_model_override/accept_effort_override instead"
        )

    async def _run_job(
        job_id: str, objective: str, created_at: str, *, engine_factory: Callable[[], Any] = engine_factory
    ) -> None:
        admission: Any = None
        try:
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
                    _safe_notify(
                        notify, f"{label} job {job_id[:8]} FAILED before it started: {objective[:100]}\n\n{exc}"
                    )
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
                        reason = _admission_rejection_text(admission)
                        registry.write(
                            job_id,
                            objective,
                            tool_name=tool_name,
                            status="failed",
                            error=reason,
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
                    job_id,
                    objective,
                    tool_name=tool_name,
                    status="running",
                    execution_started=False,
                    created_at=created_at,
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
        finally:
            # A no-op unless admission_gate actually granted something above
            # (admission stays None on every other exit -- denied, pre_confirm
            # failure, admission_gate failure/refusal, or no admission_gate at
            # all -- and release_admission() itself no-ops for None).
            await release_admission(admission)

    async def _start_and_describe(objective: str, *, engine_factory_for_job: Callable[[], Any]) -> str:
        job_id = str(uuid.uuid4())
        created_at = datetime.now(UTC).isoformat()
        initial_status = "awaiting_approval" if pre_confirm is not None else "running"
        registry.write(job_id, objective, tool_name=tool_name, status=initial_status, created_at=created_at)
        _track(background_tasks, _run_job(job_id, objective, created_at, engine_factory=engine_factory_for_job))
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

    if not dynamic:
        # The pre-1.8 shape, byte-for-byte: no extra params means no reason
        # for the generated tool's real signature (and therefore its JSON
        # Schema) to differ from what it has always been.
        async def delegate(objective: str) -> str:
            # Before anything else: no job record, no approval asked. A model
            # this engine can never run is refused for free, same discipline
            # applied to every other cheap-to-check precondition in this
            # package.
            if guard is not None:
                rejection = await _maybe_await_value(guard(objective, {}))
                if rejection is not None:
                    return rejection
            if validate_model is not None:
                rejection = validate_model(model)
                if rejection is not None:
                    return rejection
            return await _start_and_describe(objective, engine_factory_for_job=engine_factory)

        delegate.__doc__ = doc
        return Tool.wrap(delegate, name=tool_name)

    async def delegate_with_overrides(objective: str, **extra: Any) -> str:
        call_model = extra.pop("model", None) if accept_model_override else None
        call_effort = extra.pop("effort", None) if accept_effort_override else None
        resolved_model = call_model if call_model is not None else model
        resolved_effort = call_effort if call_effort is not None else effort
        call_kwargs: dict[str, Any] = dict(extra)
        if accept_model_override:
            call_kwargs["model"] = resolved_model
        if accept_effort_override:
            call_kwargs["effort"] = resolved_effort

        if guard is not None:
            rejection = await _maybe_await_value(guard(objective, dict(call_kwargs)))
            if rejection is not None:
                return rejection
        if validate_model is not None:
            rejection = validate_model(resolved_model)
            if rejection is not None:
                return rejection

        def _engine_factory_for_job() -> Any:
            return engine_factory(**call_kwargs)

        return await _start_and_describe(objective, engine_factory_for_job=_engine_factory_for_job)

    delegate_with_overrides.__doc__ = doc
    schema = _build_delegate_schema(
        accept_model_override=accept_model_override,
        accept_effort_override=accept_effort_override,
        extra_params=extra_params,
    )
    return Tool.from_schema(tool_name, doc, schema, delegate_with_overrides)


def make_persistent_consultant(
    build: Any,
    *,
    tool_name: str,
    registry: JobRegistry,
    background_tasks: set[asyncio.Task[Any]],
    notify: Callable[[str], None] | None = None,
    doc_suffix: str = "",
    accept_fresh: bool = False,
    accept_model_override: bool = False,
    accept_effort_override: bool = False,
) -> Tool:
    """Build one serialized, handle-remembering consultant conversation.

    The lock guards the background job rather than the tool call, preserving
    low turn latency. Without it, overlapping calls can both read the same
    stale handle, open separate conversations, and leave only the final
    reply's handle remembered.

    ``accept_fresh``, when set, adds an optional ``fresh: bool = False``
    parameter to the generated ``ask`` tool: a call with ``fresh=True``
    starts a brand-new thread (passes ``None`` for the remembered handle
    instead of the current one) rather than continuing the running
    conversation -- whatever handle that new thread gets back still becomes
    the one subsequent calls continue, exactly like today's single running
    conversation.

    ``accept_model_override``/``accept_effort_override``, when set, add an
    optional ``model``/``effort`` parameter the same way, forwarded to the
    wrapped consultant call for that one call only. Each requires the
    wrapped consultant function to actually accept a ``model``/``effort``
    parameter of its own -- checked once at build time (like the
    ``thread_id``/``session_id`` detection just below), raising
    :class:`TypeError` immediately rather than failing on the first call
    that tries to use it.

    When none of ``accept_fresh``, ``accept_model_override``, or
    ``accept_effort_override`` is set (the default), the generated tool's
    signature is exactly ``ask(question: str)``, byte-identical to the
    pre-1.8 shape.
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
    if accept_model_override and "model" not in params:
        raise TypeError(f"{tool_name}: accept_model_override=True but the wrapped function has no `model` parameter")
    if accept_effort_override and "effort" not in params:
        raise TypeError(f"{tool_name}: accept_effort_override=True but the wrapped function has no `effort` parameter")
    state: dict[str, str | None] = {"handle": None}
    lock = asyncio.Lock()

    async def _run_job(
        job_id: str, question: str, *, fresh: bool = False, model: str | None = None, effort: str | None = None
    ) -> None:
        async with lock:
            call_kwargs: dict[str, Any] = {id_field: None if fresh else state["handle"]}
            if accept_model_override and model is not None:
                call_kwargs["model"] = model
            if accept_effort_override and effort is not None:
                call_kwargs["effort"] = effort
            try:
                result = await inner.func(question=question, **call_kwargs)
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

    ask: Callable[..., Awaitable[str]]
    if not (accept_fresh or accept_model_override or accept_effort_override):
        # Byte-identical to the pre-1.8 shape: no override flag means no
        # reason for the generated tool's real signature to differ from
        # what it has always been.
        async def _ask_plain(question: str) -> str:
            job_id = str(uuid.uuid4())
            registry.write(job_id, question, tool_name=tool_name, status="running")
            _track(background_tasks, _run_job(job_id, question))
            preview = elide(question)
            _safe_notify(notify, f"Consulting {tool_name} (job {job_id[:8]}): {preview}")
            return (
                f"Started job {job_id[:8]} -- {tool_name} is answering in the background, not blocking you. "
                "check_jobs() for the answer when it's ready."
            )

        ask = _ask_plain
    else:
        # A separate function (not a second ``def ask`` in this same scope)
        # so the two variants' different signatures never look like one
        # function redefined -- mypy flags that as an error, and it would
        # be a real one: callers of each branch expect a different shape.
        async def _ask_with_overrides(
            question: str, fresh: bool = False, model: str | None = None, effort: str | None = None
        ) -> str:
            if fresh and not accept_fresh:
                return "REJECTED: this consultant does not support fresh=True"
            if model is not None and not accept_model_override:
                return "REJECTED: this consultant does not support a per-call model override"
            if effort is not None and not accept_effort_override:
                return "REJECTED: this consultant does not support a per-call effort override"
            job_id = str(uuid.uuid4())
            registry.write(job_id, question, tool_name=tool_name, status="running")
            _track(background_tasks, _run_job(job_id, question, fresh=fresh, model=model, effort=effort))
            preview = elide(question)
            _safe_notify(notify, f"Consulting {tool_name} (job {job_id[:8]}): {preview}")
            return (
                f"Started job {job_id[:8]} -- {tool_name} is answering in the background, not blocking you. "
                "check_jobs() for the answer when it's ready."
            )

        ask = _ask_with_overrides

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
    a smaller batch, not a partial one nobody asked for. Every admission
    already granted earlier in THIS batch is given back (via
    :func:`~lazybridge.ext.delegation.admission.refund_admission`) the
    moment a later objective is refused, or the post-check capacity race
    below fires -- the batch starts nothing, so nothing it already
    reserved may stay reserved either. This mirrors LazyCEO's own
    ``run_parallel`` batch-rollback behaviour. Once an objective's own job
    is actually scheduled, its admission is released (not refunded) when
    that job finishes, success or failure -- see
    :func:`_run_with_admission_release`.

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

        admissions: list[Any] = [None] * len(objectives)
        if admission_gate is not None:
            for index in range(len(objectives)):
                admission = await admission_gate()
                if admission is not None and not getattr(admission, "allowed", True):
                    # Nothing partially admitted may stay reserved once the
                    # whole batch is held back -- every admission granted
                    # for an EARLIER objective in this same loop is given
                    # back before returning.
                    for granted in admissions[:index]:
                        await refund_admission(granted)
                    rejection = _admission_rejection_text(admission)
                    return (
                        f"REJECTED: {rejection} (refused at objective {index + 1} of {len(objectives)}; "
                        "the whole batch was held back rather than started in part)"
                    )
                admissions[index] = admission
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
                for granted in admissions:
                    await refund_admission(granted)
                return (
                    f"REJECTED: {len(background_tasks)} job(s) already in flight plus {len(objectives)} new "
                    f"objective(s) exceeds the process-local cap of {max_in_flight_delegate_tasks} -- capacity "
                    "was taken by another call while admission was being checked"
                )

        job_ids: list[str] = []
        for objective, admission in zip(objectives, admissions, strict=True):
            job_id = str(uuid.uuid4())
            created_at = datetime.now(UTC).isoformat()
            registry.write(job_id, objective, tool_name="run_parallel", status="running", created_at=created_at)
            _schedule_with_admission_release(
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
                admission,
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
    admission_gate: Callable[[], Awaitable[Any]] | None = None,
) -> Tool:
    """Build a tool that claims durable-plan tasks before delegation.

    Like ``run_parallel``, this must be a coroutine function -- see that
    function's docstring for why a synchronous one would break
    ``asyncio.create_task`` inside ``_track`` when dispatched through
    :class:`~lazybridge.Tool`'s executor path for plain functions.

    Unlike ``run_parallel``, ``admission_gate`` here is checked PER ITEM,
    immediately before that item's own ``board.claim_task`` -- not in a
    separate whole-batch pre-pass -- because each item's claim and job
    record are already independent of every other item's in this loop.  A
    refusal costs that one item nothing: nothing was claimed yet, so
    nothing needs unwinding on the board. The unwind this function HAS
    always had to do -- a claim that succeeds but whose job then fails to
    even get scheduled (``registry.write``/``_track`` raising) -- now also
    refunds that item's granted admission (via
    :func:`~lazybridge.ext.delegation.admission.refund_admission`)
    alongside releasing the board claim (``board.mark_failed``): the
    attempt that admission was reserved for never actually ran. Once a
    job IS scheduled, its admission is released (not refunded) when that
    job finishes, success or failure -- see
    :func:`_run_with_admission_release`.
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

            admission: Any = None
            if admission_gate is not None:
                # Ask BEFORE claiming: a refusal must cost this item
                # nothing -- no claim taken out, nothing to unwind.
                admission = await admission_gate()
                if admission is not None and not getattr(admission, "allowed", True):
                    outcomes[item_number - 1] = f"- item {item_number}: {_admission_rejection_text(admission)}"
                    continue

            job_id = str(uuid.uuid4())
            claimed = board.claim_task(task_index, expected_text, owner=owner)
            if isinstance(claimed, str):
                outcomes[item_number - 1] = f"- item {item_number}: {claimed}"
                # Granted but the claim lost the race -- this attempt never
                # ran at all.
                await refund_admission(admission)
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
                _schedule_with_admission_release(background_tasks, coroutine, admission)
            except Exception as exc:
                if coroutine is not None:
                    coroutine.close()
                error = f"delegate_plan_tasks setup failed: {exc}"
                board.mark_failed(task_index, error, owner=owner)
                # The job never ran -- give back its admission alongside the
                # board claim, same "never attempted" reasoning as the
                # lost-the-claim-race branch above.
                await refund_admission(admission)
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
