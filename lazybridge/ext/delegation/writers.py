"""Codex- and Claude-Code-specific background writer factories.

These provider adapters sit in the delegation extension under the policy in
``docs/guides/core-vs-ext.md``; the durable job and scheduling machinery in
``background`` remains engine-agnostic.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lazybridge import Tool
from lazybridge._display import elide
from lazybridge.ext.delegation.background import make_background_delegate
from lazybridge.ext.delegation.jobs import JobRegistry


def make_codex_writer(
    *,
    workspace_root: Path,
    gate: Any,
    channel: Any,
    registry: JobRegistry,
    background_tasks: set[asyncio.Task[Any]],
    model: str | None = None,
    notify: Callable[[str], None] | None = None,
    doc: str,
    confirmation_prompt: str = "About to delegate to Codex: {preview}\n\nProceed?",
) -> Tool:
    """Build a one-shot-confirmed Codex workspace writer."""
    from lazybridge.engines.coding import CodingAgentConfig

    # `gate` does NOT protect this the way it protects the agent's own Bash calls or claude_write's sub-agent — Codex's sandbox is a hard permission boundary, not a per-call tier check, so `channel` is used directly (not through `gate`'s tier matching) to ask ONE question — "delegate this whole task to Codex at all?" — before the sub-agent starts; a live commit was observed landing with zero calls to `gate`.
    config = CodingAgentConfig.writer(gate)

    async def _confirm(objective: str) -> bool:
        preview = elide(objective)
        return await channel.ask(confirmation_prompt.format(preview=preview))

    def _engine_factory() -> Any:
        from lazybridge.engines.codex import CodexEngine

        # No request deadline: live multi-file work hit the same blank-error
        # timeout failure as the dispatcher until this construction site was
        # explicitly made unbounded.
        return CodexEngine(model=model, cwd=str(workspace_root), config=config, request_timeout=None)

    return make_background_delegate(
        tool_name="codex_write",
        label="Codex",
        engine_factory=_engine_factory,
        registry=registry,
        background_tasks=background_tasks,
        notify=notify,
        pre_confirm=_confirm,
        doc=doc,
    )


def make_claude_writer(
    *,
    workspace_root: Path,
    gate: Any,
    registry: JobRegistry,
    background_tasks: set[asyncio.Task[Any]],
    model: str = "sonnet",
    notify: Callable[[str], None] | None = None,
    doc: str,
) -> Tool:
    """Build a per-action-gated Claude Code workspace writer."""
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

        # SDK default is 20, and a real live objective hit "Reached maximum
        # number of turns" while still investigating, before writing a single file.
        return ClaudeCodeEngine(
            model=model,
            cwd=str(workspace_root),
            config=config,
            request_timeout=None,
            max_turns=60,
        )

    return make_background_delegate(
        tool_name="claude_write",
        label="a Claude Code sub-agent",
        engine_factory=_engine_factory,
        registry=registry,
        background_tasks=background_tasks,
        notify=notify,
        doc=doc,
    )
