"""Declarative, tiered approval policies for coding-agent tool calls.

:class:`~lazybridge.ext.approval.tiered.TieredGate` implements
:class:`~lazybridge.engines.coding.ApprovalGate` over an ordered table of
:class:`~lazybridge.ext.approval.tiered.Rule` objects, each assigning one of
four tiers (``allow``/``session``/``ask``/``deny``) to a tool-name pattern
(and optionally an argument pattern). Promoted from the ``approval-lab``
prototype after live verification against a real coding agent — see
``lazybridge.ext.approval.tiered`` for the full design rationale, the
compound-Bash-command caveat, and the three gaps fixed on promotion (session
grant scoping, structured audit log, confirmed default-deny).

Lives in ``ext`` rather than core for the same reason as ``ext.hil``: it's an
opinionated policy pattern layered on the ``ApprovalGate`` protocol, not a
primitive every agent needs. Nothing here is provider-specific — one
``TieredGate`` answers requests from either Claude Code or Codex.

    from lazybridge.ext.approval import Channel, Rule, TerminalChannel, TieredGate

    gate = TieredGate(
        channel=TerminalChannel(),
        rules=(
            Rule("allow", "Read"), Rule("allow", "Bash", "git status*"),
            Rule("session", "Write"),
            Rule("ask", "Bash", "git commit*"),
            Rule("deny", "Bash", "git push*"),
        ),
    )
"""

from __future__ import annotations

from lazybridge.ext.approval.queue import (
    ApprovalQueue,
    ApprovalTicket,
    StoreApprovalChannel,
    TicketKind,
    ticket_gist,
)
from lazybridge.ext.approval.tiered import (
    AuditRecord,
    Channel,
    Rule,
    TerminalChannel,
    Tier,
    TieredGate,
    run_gate_sync,
)

__all__ = [
    "ApprovalQueue",
    "ApprovalTicket",
    "AuditRecord",
    "Channel",
    "Rule",
    "StoreApprovalChannel",
    "TerminalChannel",
    "TicketKind",
    "Tier",
    "TieredGate",
    "run_gate_sync",
    "ticket_gist",
]
