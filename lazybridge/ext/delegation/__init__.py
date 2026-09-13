"""Durable, fire-and-forget delegation for long-running agents.

This is an **extension** package under the LazyBridge core-vs-ext policy
(see ``docs/guides/core-vs-ext.md``). API may change between minor
releases; consult the per-module CHANGELOG before pinning.

Promoted from LazyCEO's generic background-delegation infrastructure after
live production use. The package keeps durable job records separate from
the process-local asyncio tasks that execute them, supports persistent
consultant conversations, fans independent objectives out in parallel, and
can link delegated work to a :class:`~lazybridge.ext.planners.DurableBlackboard`
task without making any of those policies part of LazyBridge core.

Every Tool this package builds must be awaited via
:meth:`~lazybridge.Tool.run` from within the delegating agent's own,
genuinely persistent event loop -- never invoked through
:meth:`~lazybridge.Tool.run_sync`, whose fresh-loop-per-call semantics
would cancel the fire-and-forget background work before it can do
anything. See :func:`~lazybridge.ext.delegation.background._track`'s own
docstring for the full explanation.

    from lazybridge import Store
    from lazybridge.ext.delegation import JobRegistry, make_background_delegate

    registry = JobRegistry(Store(db="delegation.sqlite"))
    background_tasks = set()
    delegate = make_background_delegate(
        tool_name="delegate",
        label="a worker",
        engine_factory=build_engine,
        registry=registry,
        background_tasks=background_tasks,
        notify=print,
        doc="Delegate one objective in the background.",
    )
"""

from lazybridge.ext.delegation.background import (
    make_background_delegate,
    make_claude_delegate_engine_factory,
    make_parallel_delegate,
    make_persistent_consultant,
    make_plan_delegate,
)
from lazybridge.ext.delegation.jobs import (
    DEFAULT_JOB_PREFIX,
    DEFAULT_SESSION_KEY_PREFIX,
    JobRegistry,
    session_id_key,
)
from lazybridge.ext.delegation.writers import make_claude_writer, make_codex_writer

__all__ = [
    "DEFAULT_JOB_PREFIX",
    "DEFAULT_SESSION_KEY_PREFIX",
    "JobRegistry",
    "make_background_delegate",
    "make_claude_delegate_engine_factory",
    "make_claude_writer",
    "make_codex_writer",
    "make_parallel_delegate",
    "make_persistent_consultant",
    "make_plan_delegate",
    "session_id_key",
]
