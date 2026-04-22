# Store

`Store` is the blackboard. When two or more agents need to share state
— intermediate results, configuration, flags — they write to a common
`Store`. Unlike `Memory`, which is conversation history attached to a
single agent, `Store` is explicitly shared and addressable by key.

Two common patterns:

1. **Hand-off via store**: Agent A writes `store["research"] = results`;
   Agent B reads `store.read("research")` in a later step. Useful with
   `Plan(Step(..., writes="research"))` where Plan does the write for you.
2. **Live context**: `Agent(sources=[store])` injects the current store
   content (as text) into every call's context. The agent sees the
   blackboard as part of its prompt, automatically updated each turn.

Pick ``db="file.sqlite"`` when you need durability — a Plan's
checkpoint/resume, cross-process sharing, or survival of crashes.
Pick ``db=None`` for tests and single-process ephemeral runs.

## Example

```python
from lazybridge import Agent, LLMEngine, Store, Plan, Step

# db="research.sqlite"  → persistent SQLite file (WAL mode, thread-safe).
#                         Pass db=None for an in-memory dict instead.
store = Store(db="research.sqlite")

# Plan step writes its result into the store automatically.
plan = Plan(
    # name="search"   → step identifier (referenced by sentinels like
    #                   from_step("search"), checkpoints, and the graph).
    # writes="hits"   → after the step runs, Plan calls store.write("hits", payload).
    Step(researcher, name="search", writes="hits"),
    Step(writer,     name="write"),
)
Agent.from_engine(plan)("AI trends")
print(store.read("hits"))

# Agent with sources= sees the live store on every call.
# ``system=`` lives on the engine, not Agent — build an LLMEngine.
monitor = Agent(
    engine=LLMEngine("claude-opus-4-7",
                     system="Report what's currently in the blackboard."),
    name="monitor",           # label in Session.graph / event logs
    sources=[store],          # store.to_text() is injected into context each call
)
print(monitor("status?").text())
```

## Atomic updates with `compare_and_swap`

`write()` is fire-and-forget — two writers that read, modify, and
write the same key will clobber each other.  `compare_and_swap` gives
you optimistic concurrency: only commit if the stored value still
matches what you read.

```python
# What this shows: incrementing a counter shared across concurrent
# agents (or processes, for db="file.sqlite") without losing updates.
# Why CAS: the classic read-modify-write is not atomic. Two agents
# read "3", both compute "4", both write "4" — one increment is lost.
# compare_and_swap detects the race and returns False so the caller
# can retry.

from lazybridge import Store

store = Store(db="counters.sqlite")   # SQLite path: CAS uses BEGIN IMMEDIATE
                                      #  so concurrent writers serialise.

def increment(key: str) -> int:
    while True:
        current = store.read(key, default=0)
        new = current + 1
        # Returns True iff the on-disk value is still ``current``.
        # Works for Pydantic models / dicts too — comparison is
        # JSON-normalised so runtime shape equals on-disk shape.
        if store.compare_and_swap(key, current, new):
            return new
        # Another writer beat us; loop and re-read.

# CAS semantics for "must not exist":
#   store.compare_and_swap("lock", expected=None, new="held")
# Returns True only if "lock" has never been written (or was deleted).
```

Use `compare_and_swap` when multiple agents share a Store key in a
pipeline and the value depends on its current state — counters, lock
handles, checkpoint pointers.  For append-only blackboards where each
agent owns a distinct key, plain `write()` is enough.

## Inspection: `read_entry`, `read_all`, `to_text`

`read(key)` returns the raw value.  When you need the **provenance**
— who wrote it and when — reach for `read_entry`.  Provenance is what
turns a Store from a dumb dict into an audit-friendly blackboard.

```python
# What this shows: all three read shapes side-by-side, and how
# agent_id threads through them.
# Why: read_entry/read_all are the right tools for debugging "who
# wrote this value and when?" questions in a multi-agent pipeline.
# agent_id is an optional string tag the caller passes on write()
# and gets back on read_entry().

from lazybridge import Store

store = Store()  # in-memory for the demo

# Write with provenance. agent_id is a free-form string; the
# convention is the producing Agent's name so read_entry later tells
# you which step wrote the value.
store.write("hits", ["p1", "p2"],       agent_id="researcher")
store.write("ranked", ["p2", "p1"],     agent_id="ranker")
store.write("draft", "short article",   agent_id="writer")

# Plain value:
store.read("hits")                         # ["p1", "p2"]

# StoreEntry with provenance:
entry = store.read_entry("hits")
assert entry.key == "hits"
assert entry.agent_id == "researcher"
assert entry.written_at > 0                # monotonic time on write

# Snapshot of every key — useful for dashboards, debug panels,
# test assertions. Returns a plain dict keyed by store key.
snapshot = store.read_all()
assert set(snapshot) == {"hits", "ranked", "draft"}

# Slice of keys rendered as "key: JSON" lines for sources=[store].
# Pass keys=[...] to restrict to a subset — critical for stores with
# thousands of keys, where injecting them all would blow the prompt
# context budget.
context = store.to_text(keys=["hits", "ranked"])
# → "hits: [\"p1\", \"p2\"]\nranked: [\"p2\", \"p1\"]"
```

`agent_id` has no framework-enforced meaning — it's whatever string
you tag values with.  The convention is the producing Agent's
`.name`, so `read_entry(key).agent_id` answers "which step wrote
this".  `Plan(Step(..., writes="k"))` auto-sets `agent_id` to the
step's agent name when it writes.

## Thread safety & durability

Two execution paths with different guarantees:

| Mode | Set up | Thread safe | Cross-process | Durable |
|---|---|---|---|---|
| In-memory dict | `Store()` (default) | Yes — internal lock | No | No (lost on exit) |
| SQLite file | `Store(db="path.sqlite")` | Yes — WAL + busy timeout | Yes (multiple processes can read/write) | Yes |

- The SQLite path uses **WAL mode** so concurrent readers never block
  writers; the file is safe to share between a Plan that checkpoints
  to it and an unrelated process that reads from it.
- `compare_and_swap` on SQLite wraps read-check-write in `BEGIN
  IMMEDIATE`, so two concurrent writers serialise on the reserved
  lock instead of interleaving.
- Thread-local connections: each worker thread gets its own SQLite
  connection from the pool; closing the Store (or using it as a
  context manager) releases all of them.

```python
from lazybridge import Store

# Context-manager usage — guarantees connection cleanup on exit.
with Store(db="run.sqlite") as store:
    store.write("seed", 42)
    # ... rest of the pipeline
# Connections closed automatically here.
```

## Pitfalls

- ``Store(db=":memory:")`` is NOT the same as ``Store()`` — the former
  opens an in-memory SQLite (connection-scoped), the latter uses a
  Python dict. Use ``Store()`` unless you specifically need SQLite
  semantics in-process.
- Large binary blobs go through JSON serialisation; for files use a
  filesystem path as the value and read the file when you need it.
- ``store.to_text()`` can be expensive for stores with thousands of keys;
  pass ``keys=[...]`` to limit the slice.

!!! note "API reference"

    Store(db: str | None = None) -> Store
      # db=None   → in-memory (lost on exit)
      # db="file" → SQLite WAL-mode, thread-safe, persistent
    
    store.write(key: str, value: Any, *, agent_id: str | None = None) -> None
    store.read(key: str, default: Any = None) -> Any
    store.read_entry(key: str) -> StoreEntry | None
    store.read_all() -> dict[str, Any]
    store.keys() -> list[str]
    store.delete(key: str) -> None
    store.clear() -> None
    store.to_text(keys: list[str] | None = None) -> str   # for sources=
    
    StoreEntry = dataclass(key, value, written_at, agent_id)

!!! warning "Rules & invariants"

    - Values are JSON-encoded on write (via ``json.dumps(default=str)``),
      so non-JSON types are stringified. Prefer primitives + Pydantic models
      (use ``.model_dump()`` before writing).
    - ``to_text()`` renders the store as ``key: <json>`` lines, designed for
      ``Agent(sources=[store])`` live-view injection.
    - Store is thread-safe via a lock (in-memory) or SQLite WAL mode + busy
      timeout (persistent). Safe to share across concurrent agents.
    - Store is not transactional; each write commits immediately.

## See also

[memory](memory.md), [session](session.md),
[plan](plan.md), [checkpoint](checkpoint.md),
decision tree: [state_layer](../decisions/state-layer.md)
