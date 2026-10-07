"""Codex- and Claude-Code-specific background writer factories.

These provider adapters sit in the delegation extension under the policy in
``docs/guides/core-vs-ext.md``; the durable job and scheduling machinery in
``background`` remains engine-agnostic.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

from lazybridge import Tool
from lazybridge._display import elide
from lazybridge.ext.delegation.background import ExtraParam, make_background_delegate
from lazybridge.ext.delegation.jobs import JobRegistry


def make_codex_writer(
    *,
    workspace_root: Path,
    gate: Any,
    channel: Any,
    registry: JobRegistry,
    background_tasks: set[asyncio.Task[Any]],
    model: str | None = None,
    effort: str | None = None,
    notify: Callable[[str], None] | None = None,
    doc: str,
    confirmation_prompt: str = "About to delegate to Codex: {preview}\n\nProceed?",
    validate_model: Callable[[str | None], str | None] | None = None,
    admission_gate: Callable[[], Awaitable[Any]] | None = None,
    writable_roots: Sequence[str] | None = None,
    accept_model_override: bool = False,
    accept_effort_override: bool = False,
    accept_session_override: bool = False,
) -> Tool:
    """Build a one-shot-confirmed Codex workspace writer.

    ``validate_model`` and ``admission_gate`` are passed straight through to
    :func:`~lazybridge.ext.delegation.background.make_background_delegate` --
    see that function's own docstring for what each hook does and when it
    runs. This module carries no model-validation or admission POLICY of
    its own; both are entirely the caller's.

    ``writable_roots``, when given, extends the Codex sandbox's writable
    paths beyond ``workspace_root`` itself via
    :class:`~lazybridge.engines.coding.CodexPolicy`'s own field of that
    name -- the prime use case is ``workspace_root`` being a git
    *worktree*, whose index and refs live under the main repository's
    ``.git`` (``git rev-parse --git-common-dir``), outside the worktree
    directory and otherwise unwritable. Resolving which extra path(s) a
    given worktree needs is left entirely to the caller (this module has
    no git-topology opinion of its own); pass the result here.

    ``accept_model_override``/``accept_effort_override``/
    ``accept_session_override`` expose an optional per-call ``model``/
    ``effort``/``session`` parameter on the generated tool -- see
    ``make_background_delegate``'s own docstring for ``accept_model_override``/
    ``accept_effort_override``. ``session`` is new here: when accepted, it
    is forwarded as ``CodexEngine(thread_id=...)``, resuming that thread
    instead of starting a new one. All three default to ``False``, keeping
    today's fixed-at-build-time behaviour the default.
    """
    from lazybridge.engines.coding import CodingAgentConfig

    # `gate` does NOT protect this the way it protects the agent's own Bash calls or claude_write's sub-agent — Codex's sandbox is a hard permission boundary, not a per-call tier check, so `channel` is used directly (not through `gate`'s tier matching) to ask ONE question — "delegate this whole task to Codex at all?" — before the sub-agent starts; a live commit was observed landing with zero calls to `gate`.
    config = CodingAgentConfig.writer(gate)
    if writable_roots:
        config = dataclasses.replace(
            config, codex=dataclasses.replace(config.codex, writable_roots=tuple(writable_roots))
        )

    async def _confirm(objective: str) -> bool:
        preview = elide(objective)
        return await channel.ask(confirmation_prompt.format(preview=preview))

    def _engine_factory(*, model: str | None = model, effort: str | None = effort, session: str | None = None) -> Any:
        from lazybridge.engines.codex import CodexEngine

        # effort/session are passed only when actually set -- an
        # engine_factory call with neither overridden must build CodexEngine
        # with EXACTLY the same keyword arguments as before this existed,
        # so a caller's own stand-in engine with the old, narrower
        # constructor signature keeps working unchanged.
        extra: dict[str, Any] = {}
        if effort is not None:
            extra["reasoning_effort"] = effort
        if session is not None:
            extra["thread_id"] = session
        # No request deadline: live multi-file work hit the same blank-error
        # timeout failure as the dispatcher until this construction site was
        # explicitly made unbounded.
        return CodexEngine(model=model, cwd=str(workspace_root), config=config, request_timeout=None, **extra)

    extra_params = (
        {"session": ExtraParam(description="Resume this Codex thread id instead of starting a new one.")}
        if accept_session_override
        else None
    )

    return make_background_delegate(
        tool_name="codex_write",
        label="Codex",
        engine_factory=_engine_factory,
        registry=registry,
        background_tasks=background_tasks,
        notify=notify,
        pre_confirm=_confirm,
        doc=doc,
        model=model,
        effort=effort,
        validate_model=validate_model,
        admission_gate=admission_gate,
        accept_model_override=accept_model_override,
        accept_effort_override=accept_effort_override,
        extra_params=extra_params,
    )


def make_claude_writer(
    *,
    workspace_root: Path,
    gate: Any,
    registry: JobRegistry,
    background_tasks: set[asyncio.Task[Any]],
    model: str = "sonnet",
    effort: str | None = None,
    notify: Callable[[str], None] | None = None,
    doc: str,
    validate_model: Callable[[str | None], str | None] | None = None,
    max_turns: int = 60,
    accept_model_override: bool = False,
    accept_effort_override: bool = False,
    accept_session_override: bool = False,
) -> Tool:
    """Build a per-action-gated Claude Code workspace writer.

    No ``admission_gate`` parameter here, unlike :func:`make_codex_writer`:
    that hook only ever fires on the ``pre_confirm`` path (see
    :func:`~lazybridge.ext.delegation.background.make_background_delegate`'s
    docstring), and this writer has none -- its own actions are already
    gated per-call through ``gate``, so there is no human wait after which
    admission would need re-checking.

    ``max_turns`` defaults to ``60`` (the SDK's own default is ``20`` --
    too low for a real multi-file objective; see the comment on the
    engine construction below). A caller whose own sub-agents need more
    headroom still (LazyCEO settled on ``100`` after ``20`` proved too low
    live) passes a larger value here.

    ``accept_model_override``/``accept_effort_override``/
    ``accept_session_override`` expose an optional per-call ``model``/
    ``effort``/``session`` parameter on the generated tool -- see
    ``make_background_delegate``'s own docstring for
    ``accept_model_override``/``accept_effort_override``. ``session`` is
    new here: when accepted, it is forwarded as
    ``ClaudeCodeEngine(session_id=...)``, resuming that session instead of
    starting a new one. All three default to ``False``, keeping today's
    fixed-at-build-time behaviour the default.
    """
    from lazybridge.engines.coding import ClaudeCodePolicy, CodingAgentConfig

    config = CodingAgentConfig(
        claude=ClaudeCodePolicy(
            permission_mode="default",
            preapprove_application_tools=False,
            extra_tools=("Write", "Edit", "Bash"),
        ),
        approval_gate=gate,
    )

    def _engine_factory(*, model: str = model, effort: str | None = effort, session: str | None = None) -> Any:
        from lazybridge.engines.claude_code import ClaudeCodeEngine

        # effort/session are passed only when actually set -- see the
        # matching comment in make_codex_writer's own _engine_factory for
        # why (an engine_factory call with neither overridden must build
        # ClaudeCodeEngine with EXACTLY the same keyword arguments as
        # before this existed).
        extra: dict[str, Any] = {}
        if effort is not None:
            extra["reasoning_effort"] = effort
        if session is not None:
            extra["session_id"] = session
        # SDK default is 20, and a real live objective hit "Reached maximum
        # number of turns" while still investigating, before writing a single file.
        return ClaudeCodeEngine(
            model=model, cwd=str(workspace_root), config=config, request_timeout=None, max_turns=max_turns, **extra
        )

    extra_params = (
        {"session": ExtraParam(description="Resume this Claude Code session id instead of starting a new one.")}
        if accept_session_override
        else None
    )

    return make_background_delegate(
        tool_name="claude_write",
        label="a Claude Code sub-agent",
        engine_factory=_engine_factory,
        registry=registry,
        background_tasks=background_tasks,
        notify=notify,
        doc=doc,
        model=model,
        effort=effort,
        validate_model=validate_model,
        accept_model_override=accept_model_override,
        accept_effort_override=accept_effort_override,
        extra_params=extra_params,
    )
