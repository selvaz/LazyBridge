# Control plane: one durable store for work and its claims

*New in 1.5.0.* `lazybridge.control_plane` is a small SQLite-backed store for a
fleet that has work to hand out: a queue of items, an atomic claim on each,
and an append-only record of everything that happened to it.

It is a **single-machine** substrate -- many processes on one disk -- not a
distributed queue. If your workers live on different hosts, this is the wrong
tool.

## What it gets right

Three things that are hard to get right and worse to get wrong:

**Atomic claim.** An item goes to exactly one worker, decided inside a single
write transaction. Read-then-write with a check in between is what lets ten
agents all believe they got the same item.

**Fencing.** Every claim carries a monotonically increasing token (`fence`). A
worker that stalls, loses its lease, and comes back to report success is
rejected, because its token is no longer current. Without this, a long pause
produces two live writers for one piece of work and the second one silently
overwrites the first.

**An append-only ledger.** Every transition is an event, never an overwrite.
The state of an item is derivable from its events, so a row that disagrees with
its own history is *visible* rather than being the only copy left.

## A minimal round trip

```python
from lazybridge.control_plane import ControlStore, FenceRejected, verify_ledger

store = ControlStore("control.db", lease_seconds=900)
store.create_project("alpha", "Project Alpha", status="open")
item_id = store.enqueue("alpha", {"task": "summarise the report"})

claim = store.claim(owner="worker-1")      # a ClaimedItem, or None if nothing is available
assert claim is not None and claim.item_id == item_id
assert store.claim(owner="worker-2") is None   # it already belongs to worker-1

try:
    ...  # do the work
    store.finish(claim.item_id, fence=claim.fence, status="done")   # or "failed"
except FenceRejected:
    ...  # the claim was taken over while this worker was away: discard its result

assert [e["event_type"] for e in store.events(item_id)] == ["queued", "claimed", "done"]
assert verify_ledger(store) == []
```

`claim()` returns a `ClaimedItem` with `item_id`, `project_id`, `payload`,
`fence` and `attempts`. An item whose lease has lapsed becomes claimable again
with the fence incremented, which is exactly what makes the previous holder's
late `finish()` fail with `FenceRejected`. `finish()` also refuses an item that
is not currently claimed, so "done" cannot be reported before the work was ever
taken, or twice.

## Scope belongs to the store, not the caller

Pass `actor_id=` and the **repository** enforces what that actor may see; a
filter in the UI is a display choice, this is the boundary:

```python
store.assign("alpha", "specialist-a")
store.enqueue("beta", {"task": "not yours"})            # the control plane itself may enqueue anywhere

store.visible_projects("specialist-a")                  # ['alpha']
store.claim(owner="specialist-a", actor_id="specialist-a")   # None: beta's work is not theirs
store.enqueue("beta", {}, actor_id="specialist-a")      # raises ScopeDenied
```

A denial does not distinguish "that project does not exist" from "that project
is not yours", and `item()` / `events()` raise `ScopeDenied` rather than
answering empty -- an empty list is indistinguishable from "no history yet", so
the caller could not tell it had been refused. An item id is not a secret and is
never treated as one.

## Checking the ledger

`verify_ledger(store)` returns a **list of problems**, each naming the kind, the
item and the detail, not a boolean:

| Kind | Meaning |
|---|---|
| `no_creation_event` | the item exists but nothing recorded its arrival |
| `two_terminal_events` | it reached a terminal state (`done`/`failed`) more than once |
| `terminal_without_event` | the row is terminal but no event says how it got there |
| `terminal_events_disagree` | row and events are both terminal, but for different outcomes |
| `event_without_terminal` | an event says it finished but the row still says otherwise |
| `fence_not_monotonic` | claim fences repeated or fell instead of rising |
| `fence_disagrees` | the row's fence differs from the last claim recorded |
| `ready_after_claim` | the row says `ready` but the history shows it was claimed -- nothing moves a claimed item back to `ready`, so it was reset behind the ledger's back |
| `orphan_events` | events reference an item that no longer exists |

An empty list means the rows and their history agree. It cannot say the *right*
thing happened, only that nothing wrote state without recording it.

## Qualifying the storage engine

SQLite in WAL mode is the candidate substrate, and it stays only if it survives
real contention on the real filesystem. `lazybridge.control_plane.probe` runs
that check with separate **processes** -- threads in one interpreter share a
connection pool and a GIL and would pass while the deployed shape fails:

```bash
python -m lazybridge.control_plane.probe --store C:/tmp/probe.db \
    --scenario concurrent-queue --workers 16 --items 10000

python -m lazybridge.control_plane.probe --store C:/tmp/probe.db \
    --scenario kill-and-reclaim
```

It exits 0 only when every count is exactly what it must be: no duplicate
claim, no lost item, no `database is locked` error, a clean ledger. On the
maintainer's Windows machine (local disk, 18 September 2026) sixteen processes
drained ten thousand items in about 35 seconds with none of those, and
`kill-and-reclaim` showed a killed worker's stale token being refused with
exactly one terminal event recorded. That is a measurement on one machine, not
a guarantee about yours -- run the probe on the disk you will deploy on. If it
fails there, the answer is a server database, not a retry loop that hides an
unsuitable substrate.
