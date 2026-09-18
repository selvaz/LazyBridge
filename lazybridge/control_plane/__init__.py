"""One durable store for work, its claims, and what happened to it.

Today a fleet's state is scattered: Pulse tasks, WorkItems, blackboard
tasks, job dicts, contracts, tickets and specialist records, each with its
own identity and its own state machine, none of them correlated. Nobody
can answer "what ran, for which project, under whose authority, and what
did it cost" without joining by hand across databases that do not share a
key.

This package is the replacement: one physical database, one transaction
boundary, and three things that are hard to get right and worse to get
wrong.

**Atomic claim.** A queue item goes to exactly one worker, decided inside
a single write transaction. Read-then-write with a check in between is
what lets ten agents all believe they got the same item, and the fleet
this replaces had precisely that shape.

**Fencing.** Every claim carries a monotonically increasing token. A
worker that stalls, loses its lease, and comes back to report success is
rejected, because its token is no longer current. Without this, a
partition or a long pause produces two live writers for one piece of
work and the second one silently overwrites the first.

**An append-only ledger.** Every transition is an event, never an
overwrite. The state of a run is derivable from its events, so a
disagreement between what a row says and what happened is visible rather
than being the only thing left.

The storage engine is a decision, not an assumption: see ``probe``. SQLite
in WAL mode is the candidate because it is already everywhere in this
ecosystem, and it stays only if it survives real concurrency on the real
filesystem.
"""

from __future__ import annotations

from .ledger import LedgerProblem, verify_ledger
from .store import ClaimedItem, ControlStore, FenceRejected, ScopeDenied

__all__ = [
    "ClaimedItem",
    "ControlStore",
    "FenceRejected",
    "LedgerProblem",
    "ScopeDenied",
    "verify_ledger",
]
