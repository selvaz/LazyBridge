"""Optional, replaceable worker phases with an explicit execution boundary."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from lazybridge.ext.delegation.admission import refund_admission, release_admission
from lazybridge.ext.delegation.jobs import JobRegistry


async def _resolve(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


@dataclass
class JobContext:
    """One attempt. Callbacks may attach setup/provenance data to ``metadata``.

    ``started`` flips only after register succeeds, immediately before execute.
    A custom register must return False when its CAS loses; never rewrite the
    winning record. Rollback receives the phase, error, and this same context.
    """

    job_id: str
    objective: str
    tool_name: str
    label: str
    engine_factory: Callable[[], Any]
    registry: JobRegistry
    notify: Callable[[str], None] | None = None
    created_at: str | None = None
    plan_id: str | None = None
    task_index: int | None = None
    plan_task_text: str | None = None
    admission: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    phase: str = "prepare"
    started: bool = False
    prepared: Any = None
    engine: Any = None
    result: Any = None
    error: BaseException | None = None
    attempts_before: int | None = None
    claim_owner: str | None = None
    claimed: bool = False
    registered: bool = False
    _rolled_back: bool = field(default=False, init=False, repr=False)
    _settled: bool = field(default=False, init=False, repr=False)

    def begin_execution(self) -> bool:
        """Acquire the registry CAS and remember ownership even if setup later raises.

        Custom register callbacks can call this before adding their own setup.
        A callback using its own CAS should set ``registered=True`` on success.
        """
        from lazybridge.ext.delegation.background import _engine_identity

        engine = self.engine if self.engine is not None else getattr(self.prepared, "engine", None)
        fields: dict[str, Any] = {"cost_unknown": True}
        if engine is not None:
            engine_name, model, effort = _engine_identity(engine)
            fields.update(
                {
                    name: value
                    for name, value in {"engine": engine_name, "model": model, "effort": effort}.items()
                    if value is not None
                }
            )
        self.registered = self.registry.begin_execution(self.job_id, extra=fields)
        return self.registered


def _prepare(context: JobContext) -> Any:
    from lazybridge import Agent

    context.engine = context.engine_factory()
    return Agent(engine=context.engine, name=f"delegate-{context.tool_name}-{context.job_id[:8]}")


def _register(context: JobContext) -> bool:
    return context.begin_execution()


async def _execute(context: JobContext) -> Any:
    return await context.prepared.run(context.objective)


def _finalize(context: JobContext) -> None:
    from lazybridge._display import elide
    from lazybridge.ext.delegation.background import _result_cost_usd, _safe_notify

    result = context.result
    cost = _result_cost_usd(result)
    changes: dict[str, Any] = {
        "status": "done" if result.ok else "failed",
        "finished_at": datetime.now(UTC).isoformat(),
        "cost_unknown": not result.ok and cost == 0.0,
    }
    if result.ok:
        changes["result"] = result.text()
    else:
        changes["error"] = result.error.message if result.error else "unknown error"
    if result.ok or cost:
        changes["cost_usd"] = cost
    context.registry.update(context.job_id, changes, expected_started=True)
    if result.ok:
        message = (
            f"{context.label} job {context.job_id[:8]} done: {context.objective[:100]}\n\n{elide(changes['result'])}"
        )
    else:
        message = f"{context.label} job {context.job_id[:8]} FAILED: {context.objective[:100]}\n\n{changes['error']}"
    _safe_notify(context.notify, message)


@dataclass
class JobRunner:
    """Run prepare → register (CAS) → execute → finalize.

    All callbacks accept JobContext and may be sync or async. Register must
    return a bool. Rollback runs exactly once on refusal, exception, or
    cancellation, including partial prepare/finalize failures. It may restore
    caller-owned claims using captured metadata; reservation settlement stays
    here. Errors propagate after cleanup. A lost CAS leaves the record intact.
    """

    # Install plain callables on each instance; a class-level function default
    # looks like a bound method to static analyzers and gains a spurious self.
    prepare: Callable[[JobContext], Any] = field(default_factory=lambda: _prepare)
    register: Callable[[JobContext], Any] = field(default_factory=lambda: _register)
    execute: Callable[[JobContext], Any] = field(default_factory=lambda: _execute)
    finalize: Callable[[JobContext], Any] = field(default_factory=lambda: _finalize)
    rollback: Callable[[JobContext], Any] | None = None

    async def __call__(self, context: JobContext) -> None:
        refused = False
        context.phase = "prepare"
        try:
            context.prepared = await _resolve(self.prepare(context))
            context.phase = "register"
            registered = await _resolve(self.register(context))
            if not isinstance(registered, bool):
                raise TypeError("register must return bool")
            if not registered:
                refused = True
                return
            context.registered = True
            context.started = True
            context.phase = "execute"
            context.result = await _resolve(self.execute(context))
            context.phase = "finalize"
            await _resolve(self.finalize(context))
        except BaseException as exc:
            context.error = exc
            raise
        finally:
            await self._cleanup(context, refused=refused)

    async def abort(self, context: JobContext, error: BaseException, *, phase: str = "schedule") -> None:
        """Unwind a failed handoff before this runner has entered any phase."""
        context.phase = phase
        context.error = error
        await self._cleanup(context)

    async def _cleanup(self, context: JobContext, *, refused: bool = False) -> None:
        try:
            if (refused or context.error is not None) and not context._rolled_back:
                context._rolled_back = True
                try:
                    if self.rollback is not None:
                        await _resolve(self.rollback(context))
                finally:
                    if not refused:
                        try:
                            recorded = context.registry.update(
                                context.job_id,
                                {
                                    "status": "failed",
                                    "error": f"{context.phase} failed: {context.error}",
                                    "execution_started": context.started,
                                    "finished_at": datetime.now(UTC).isoformat(),
                                },
                                only_active=True,
                                expected_started=context.started or context.registered,
                            )
                        except Exception:
                            # Best-effort, as before, but never silent: a failed write
                            # here leaves the job looking "running". Found by review.
                            logging.getLogger(__name__).exception(
                                "could not record the failure of job %s", context.job_id
                            )
                            recorded = True  # still surface the failure if its Store write failed
                        if recorded:
                            from lazybridge.ext.delegation.background import _safe_notify

                            before_start = " before it started" if not context.started else ""
                            _safe_notify(
                                context.notify,
                                f"{context.label} job {context.job_id[:8]} FAILED{before_start}: "
                                f"{context.objective[:100]}\n\n{context.error}",
                            )
        finally:
            if not context._settled:
                context._settled = True
                if context.started:
                    await release_admission(context.admission)
                else:
                    await refund_admission(context.admission)
