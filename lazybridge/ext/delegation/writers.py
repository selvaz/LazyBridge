"""Codex- and Claude-Code-specific background writer factories.

These provider adapters sit in the delegation extension under the policy in
``docs/guides/core-vs-ext.md``; the durable job and scheduling machinery in
``background`` remains engine-agnostic.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from lazybridge import Tool
from lazybridge._display import elide
from lazybridge.engines.sessions import SessionRegistry
from lazybridge.ext.delegation.background import ExtraParam, make_background_delegate
from lazybridge.ext.delegation.jobs import JobRegistry
from lazybridge.ext.delegation.lifecycle import JobRunner


class _DefaultAlias:
    """Distinguish an omitted per-call alias from an explicit alias or None."""


_DEFAULT_ALIAS = _DefaultAlias()


def _resolve_alias(session: str | None, alias: str | _DefaultAlias | None, configured: str | None) -> str | None:
    if isinstance(alias, _DefaultAlias):
        return configured if session is None else None
    if session is not None and alias is not None:
        raise ValueError("choose a native session id or a session_alias")
    return alias


def make_codex_writer_engine_factory(
    *,
    workspace_root: Path,
    gate: Any,
    model: str | None = None,
    effort: str | None = None,
    session_alias: str | None = None,
    session_registry: SessionRegistry | None = None,
    writable_roots: Sequence[str] | None = None,
) -> Callable[..., Any]:
    """Reusable Codex writer factory; each call constructs a fresh engine.

    The returned callable accepts per-call cwd/model/effort/session_alias,
    native ``session`` (thread id), and writable_roots. Aliases are scoped by cwd.
    """
    from lazybridge.engines.coding import CodingAgentConfig

    config = CodingAgentConfig.writer(gate)
    configured_alias = session_alias

    def factory(
        *,
        cwd: str | Path = workspace_root,
        model: str | None = model,
        effort: str | None = effort,
        session: str | None = None,
        session_alias: str | _DefaultAlias | None = _DEFAULT_ALIAS,
        writable_roots: Sequence[str] | None = writable_roots,
    ) -> Any:
        from lazybridge.engines.codex import CodexEngine

        resolved_alias = _resolve_alias(session, session_alias, configured_alias)
        selected_config = config
        if writable_roots:
            selected_config = dataclasses.replace(
                config, codex=dataclasses.replace(config.codex, writable_roots=tuple(writable_roots))
            )
        extra: dict[str, Any] = {}
        if effort is not None:
            extra["reasoning_effort"] = effort
        if session is not None:
            extra["thread_id"] = session
        if resolved_alias is not None:
            extra["session_alias"] = resolved_alias
        if session_registry is not None:
            extra["session_registry"] = session_registry
        return CodexEngine(model=model, cwd=str(cwd), config=selected_config, request_timeout=None, **extra)

    return factory


def make_claude_writer_engine_factory(
    *,
    workspace_root: Path,
    gate: Any,
    model: str = "sonnet",
    effort: str | None = None,
    session_alias: str | None = None,
    session_registry: SessionRegistry | None = None,
    max_turns: int = 60,
) -> Callable[..., Any]:
    """Reusable Claude writer factory with per-call cwd/model/effort/alias/max_turns."""
    from lazybridge.engines.coding import ClaudeCodePolicy, CodingAgentConfig

    config = CodingAgentConfig(
        claude=ClaudeCodePolicy(
            permission_mode="default", preapprove_application_tools=False, extra_tools=("Write", "Edit", "Bash")
        ),
        approval_gate=gate,
    )
    configured_alias = session_alias

    def factory(
        *,
        cwd: str | Path = workspace_root,
        model: str = model,
        effort: str | None = effort,
        session: str | None = None,
        session_alias: str | _DefaultAlias | None = _DEFAULT_ALIAS,
        max_turns: int = max_turns,
    ) -> Any:
        from lazybridge.engines.claude_code import ClaudeCodeEngine

        resolved_alias = _resolve_alias(session, session_alias, configured_alias)
        extra: dict[str, Any] = {}
        if effort is not None:
            extra["reasoning_effort"] = effort
        if session is not None:
            extra["session_id"] = session
        if resolved_alias is not None:
            extra["session_alias"] = resolved_alias
        if session_registry is not None:
            extra["session_registry"] = session_registry
        return ClaudeCodeEngine(
            model=model, cwd=str(cwd), config=config, request_timeout=None, max_turns=max_turns, **extra
        )

    return factory


def _writer_params(
    extra_params: Mapping[str, ExtraParam] | None,
    *,
    accept_session_override: bool,
    accept_session_alias_override: bool,
    provider: str,
) -> dict[str, ExtraParam]:
    params = dict(extra_params or {})
    if {"session", "session_alias"} & params.keys():
        raise ValueError("declare writer session parameters using the accept_session flags")
    if accept_session_override:
        params["session"] = ExtraParam(description=f"Resume this native {provider} session id.")
    if accept_session_alias_override:
        params["session_alias"] = ExtraParam(
            description="Resume a durable named conversation scoped to this workspace."
        )
    return params


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
    session_alias: str | None = None,
    session_registry: SessionRegistry | None = None,
    accept_session_alias_override: bool = False,
    engine_factory: Callable[..., Any] | None = None,
    extra_params: Mapping[str, ExtraParam] | None = None,
    guard: Callable[[str, dict[str, Any]], Any] | None = None,
    job_runner: JobRunner | None = None,
) -> Tool:
    """Build a one-shot-confirmed Codex workspace writer.

    ``session_alias`` and ``session_registry`` resume a durable conversation by
    name; ``accept_session_alias_override`` exposes that distinct per-call name.
    Native ``session`` still means a provider id. ``engine_factory``, ``extra_params``,
    ``guard``, and ``job_runner`` pass through the background builder's hooks.

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
    factory = engine_factory or make_codex_writer_engine_factory(
        workspace_root=workspace_root,
        gate=gate,
        model=model,
        effort=effort,
        session_alias=session_alias,
        session_registry=session_registry,
        writable_roots=writable_roots,
    )

    async def _confirm(objective: str) -> bool:
        preview = elide(objective)
        return await channel.ask(confirmation_prompt.format(preview=preview))

    params = _writer_params(
        extra_params,
        accept_session_override=accept_session_override,
        accept_session_alias_override=accept_session_alias_override,
        provider="Codex",
    )

    return make_background_delegate(
        tool_name="codex_write",
        label="Codex",
        engine_factory=factory,
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
        extra_params=params or None,
        guard=guard,
        job_runner=job_runner,
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
    admission_gate: Callable[[], Awaitable[Any]] | None = None,
    validate_model: Callable[[str | None], str | None] | None = None,
    max_turns: int = 60,
    accept_model_override: bool = False,
    accept_effort_override: bool = False,
    accept_session_override: bool = False,
    session_alias: str | None = None,
    session_registry: SessionRegistry | None = None,
    accept_session_alias_override: bool = False,
    engine_factory: Callable[..., Any] | None = None,
    extra_params: Mapping[str, ExtraParam] | None = None,
    guard: Callable[[str, dict[str, Any]], Any] | None = None,
    job_runner: JobRunner | None = None,
) -> Tool:
    """Build a per-action-gated Claude Code workspace writer.

    ``session_alias`` and ``session_registry`` resume a durable conversation by
    name, distinct from the native ``session`` id. See the Codex writer for the
    optional per-call alias and factory/guard/lifecycle hooks.

    ``admission_gate`` is consulted at scheduling time before registration.
    Its grant is refunded for work that never starts and released on completion.

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
    factory = engine_factory or make_claude_writer_engine_factory(
        workspace_root=workspace_root,
        gate=gate,
        model=model,
        effort=effort,
        session_alias=session_alias,
        session_registry=session_registry,
        max_turns=max_turns,
    )
    params = _writer_params(
        extra_params,
        accept_session_override=accept_session_override,
        accept_session_alias_override=accept_session_alias_override,
        provider="Claude Code",
    )

    return make_background_delegate(
        tool_name="claude_write",
        label="a Claude Code sub-agent",
        engine_factory=factory,
        registry=registry,
        background_tasks=background_tasks,
        notify=notify,
        doc=doc,
        model=model,
        effort=effort,
        validate_model=validate_model,
        admission_gate=admission_gate,
        accept_model_override=accept_model_override,
        accept_effort_override=accept_effort_override,
        extra_params=params or None,
        guard=guard,
        job_runner=job_runner,
    )
