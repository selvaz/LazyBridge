# Store

**Use `Store`** for cross-process or cross-run state — pipeline
checkpoints, computed artefacts a downstream agent should read live, or
shared scratch space across a fan-out.  SQLite mode is durable and
thread-safe.

**Don't use `Store`** for in-prompt conversation context — that's
`Memory`'s job.  `Store` is "what should survive a crash"; `Memory` is
"what should the model see in the next turn".

## Example

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
    - ``Store()`` (in-memory) returns a **deep copy** from ``read()`` and
      stores a deep copy on ``write()``, matching the SQLite path's
      copy-on-write semantics. Do not rely on reference identity from
      ``store.read()`` — mutating the returned value does not affect the store.

## See also

- [Memory](memory.md) — separate concept (in-prompt conversation context).
- [Checkpoint & resume](checkpoint.md) — `Plan` uses `Store` under the hood.
