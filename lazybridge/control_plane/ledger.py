"""Checking that the rows and their history still agree.

The ledger is append-only, so a queue row is a cache of what its events
already say. That makes one class of bug checkable rather than
speculative: if the row and the events disagree, something wrote state
without recording it, and that is exactly the kind of fault that hides
until an incident needs reconstructing.

``verify_ledger`` is run in the probes and by the operator CLI, and its
answer is a list of problems rather than a boolean -- "something is
wrong" is not an actionable result at three in the morning.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import ControlStore

_TERMINAL = {"done", "failed"}


@dataclass(frozen=True)
class LedgerProblem:
    kind: str
    item_id: str
    detail: str

    def __str__(self) -> str:
        return f"{self.kind}: {self.item_id} -- {self.detail}"


def verify_ledger(store: ControlStore) -> list[LedgerProblem]:
    """Every way the rows and the events can contradict each other."""
    problems: list[LedgerProblem] = []
    by_item: dict[str, list[dict]] = defaultdict(list)
    # One snapshot for both reads. As two autocommit statements they can
    # straddle a worker finishing an item, and a healthy item then reads as
    # terminal-without-event.
    with store._consistent_read():
        all_events = store._every_event()
        items = {item["item_id"]: item for item in store._every_item()}
    for event in all_events:
        by_item[event["item_id"]].append(event)

    for item_id, item in items.items():
        events = by_item.get(item_id, [])
        kinds = [e["event_type"] for e in events]

        if "queued" not in kinds:
            problems.append(
                LedgerProblem("no_creation_event", item_id, "the item exists but nothing recorded its arrival")
            )

        terminals = [k for k in kinds if k in _TERMINAL]
        if len(terminals) > 1:
            problems.append(
                LedgerProblem(
                    "two_terminal_events",
                    item_id,
                    f"reached a terminal state more than once: {terminals}",
                )
            )
        if item["status"] in _TERMINAL and not terminals:
            problems.append(
                LedgerProblem(
                    "terminal_without_event",
                    item_id,
                    f"the row says {item['status']!r} but no event says how it got there",
                )
            )
        if terminals and item["status"] in _TERMINAL and terminals[-1] != item["status"]:
            # Both terminal is not the same as agreeing. A failed item
            # overwritten to done leaves the row terminal and the events
            # terminal, so a check that only asks "are both terminal"
            # hands the operator a clean ledger for the wrong outcome.
            # Found by Codex review on PR #173.
            problems.append(
                LedgerProblem(
                    "terminal_events_disagree",
                    item_id,
                    f"the row says {item['status']!r} but the event says {terminals[-1]!r}",
                )
            )
        if terminals and item["status"] not in _TERMINAL:
            problems.append(
                LedgerProblem(
                    "event_without_terminal",
                    item_id,
                    f"an event says {terminals[0]!r} but the row still says {item['status']!r}",
                )
            )

        claims = [e for e in events if e["event_type"] == "claimed"]
        if item["status"] == "ready" and (claims or int(item["fence"]) > 0):
            # Nothing ever sets a claimed item back to "ready" -- a lapsed
            # lease is reclaimed straight to "claimed" under a higher
            # fence. A row that says ready while its own history shows it
            # was taken was reset behind the ledger's back, and the fence
            # check alone cannot see it because the fence still matches.
            problems.append(
                LedgerProblem(
                    "ready_after_claim",
                    item_id,
                    f"the row says 'ready' but the history shows {len(claims)} claim(s) at fence {item['fence']}",
                )
            )
        if claims:
            fences = [int(e["fence"]) for e in claims if e["fence"] is not None]
            if fences != sorted(fences) or len(set(fences)) != len(fences):
                problems.append(
                    LedgerProblem(
                        "fence_not_monotonic",
                        item_id,
                        f"claim fences must rise and never repeat, got {fences}",
                    )
                )
            if int(item["fence"]) != max(fences, default=0):
                problems.append(
                    LedgerProblem(
                        "fence_disagrees",
                        item_id,
                        f"the row is at fence {item['fence']} but the last claim recorded {max(fences, default=0)}",
                    )
                )

    for item_id in by_item:
        if item_id not in items:
            problems.append(LedgerProblem("orphan_events", item_id, "events reference an item that no longer exists"))

    return problems
