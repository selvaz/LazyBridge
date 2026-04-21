## signature
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

## rules
- Values are JSON-encoded on write (via ``json.dumps(default=str)``),
  so non-JSON types are stringified. Prefer primitives + Pydantic models
  (use ``.model_dump()`` before writing).
- ``to_text()`` renders the store as ``key: <json>`` lines, designed for
  ``Agent(sources=[store])`` live-view injection.
- Store is thread-safe via a lock (in-memory) or SQLite WAL mode + busy
  timeout (persistent). Safe to share across concurrent agents.
- Store is not transactional; each write commits immediately.

## narrative
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

## example
```python
from lazybridge import Agent, Store, Plan, Step

store = Store(db="research.sqlite")

# Plan step writes a result into the store automatically.
plan = Plan(
    Step(researcher, name="search", writes="hits"),
    Step(writer,     name="write"),
)
Agent.from_engine(plan)("AI trends")
print(store.read("hits"))

# Agent with sources= sees the live store on every call.
monitor = Agent("claude-opus-4-7", name="monitor", sources=[store],
                system="Report what's currently in the blackboard.")
print(monitor("status?").text())
```

## pitfalls
- ``Store(db=":memory:")`` is NOT the same as ``Store()`` — the former
  opens an in-memory SQLite (connection-scoped), the latter uses a
  Python dict. Use ``Store()`` unless you specifically need SQLite
  semantics in-process.
- Large binary blobs go through JSON serialisation; for files use a
  filesystem path as the value and read the file when you need it.
- ``store.to_text()`` can be expensive for stores with thousands of keys;
  pass ``keys=[...]`` to limit the slice.

## see-also
[memory](memory.md), [session](session.md),
[plan](plan.md), [checkpoint](checkpoint.md),
decision tree: [state_layer](../decisions/state-layer.md)
