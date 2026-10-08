"""Optional, replaceable worker phases with an explicit execution boundary."""

from __future__ import annotations

import contextlib
import inspect
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
    result: Any = None
    error: BaseException | None = None


def _prepare(context: JobContext) -> Any:
    from lazybridge import Agent

    return Agent(engine=context.engine_factory(), name=f"delegate-{context.tool_name}-{context.job_id[:8]}")


def _register(context: JobContext) -> bool:
    return context.registry.begin_execution(context.job_id)


async def _execute(context: JobContext) -> Any:
    return await context.prepared.run(context.objective)


def _finalize(context: JobContext) -> None:
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
    context.registry.update(context.job_id, changes)
    _safe_notify(context.notify, f"{context.label} job {context.job_id[:8]} {changes['status']}")


@dataclass
class JobRunner:
    """Run prepare → register (CAS) → execute → finalize.

    All callbacks accept JobContext and may be sync or async. Register must
    return a bool. Rollback runs exactly once on refusal, exception, or
    cancellation, including partial prepare/finalize failures. It may restore
    caller-owned claims using captured metadata; reservation settlement stays
    here. Errors propagate after cleanup. A lost CAS leaves the record intact.
    """

    prepare: Callable[[JobContext], Any] = _prepare
    register: Callable[[JobContext], Any] = _register
    execute: Callable[[JobContext], Any] = _execute
    finalize: Callable[[JobContext], Any] = _finalize
    rollback: Callable[[JobContext], Any] | None = None

    async def __call__(self, context: JobContext) -> None:
        refused = False
        try:
            context.prepared = await _resolve(self.prepare(context))
            context.phase = "register"
            registered = await _resolve(self.register(context))
            if not isinstance(registered, bool):
                raise TypeError("register must return bool")
            if not registered:
                refused = True
                return
            context.started = True
            context.phase = "execute"
            context.result = await _resolve(self.execute(context))
            context.phase = "finalize"
            await _resolve(self.finalize(context))
        except BaseException as exc:
            context.error = exc
            raise
        finally:
            try:
                if refused or context.error is not None:
                    try:
                        if self.rollback is not None:
                            await _resolve(self.rollback(context))
                    finally:
                        if not refused:
                            # Best effort if the Store itself caused the failure.
                            # Preserve metadata and any recovery/other worker's terminal record.
                            with contextlib.suppress(Exception):
                                context.registry.update(
                                    context.job_id,
                                    {
                                        "status": "failed",
                                        "error": f"{context.phase} failed: {context.error}",
                                        "execution_started": context.started,
                                        "finished_at": datetime.now(UTC).isoformat(),
                                    },
                                    only_active=True,
                                )
            finally:
                if context.started:
                    await release_admission(context.admission)
                else:
                    await refund_admission(context.admission)
