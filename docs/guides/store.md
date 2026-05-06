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
from lazybridge import Agent, LLMEngine, Store, Plan, Step, from_agent

store = Store(db="research.sqlite")

# 1) Plan step writes a result into the store via Step(writes=).
researcher = Agent(engine=LLMEngine("claude-opus-4-7"), name="research")
writer = Agent(engine=LLMEngine("gpt-4o"), name="write")

pipeline = Agent(
    engine=Plan(
        Step("research", writes="hits"),  # stores result under key "hits"
        Step("write"),
    ),
    tools=[researcher.as_tool("research"), writer.as_tool("write")],
    store=store,  # store= is a first-class Agent parameter
)
pipeline("AI trends")
print(store.read("hits"))

# 2) Agents write their output to store automatically after each run.
#    Key: "__agent_output__:{agent.name}"
standalone = Agent(engine=LLMEngine("claude-opus-4-7"), store=store, name="research")
standalone("AI trends 2026")
print(store.read("__agent_output__:research"))  # the agent's last output

# 3) from_agent("name") reads that output in a Plan step.
editor = Agent(engine=LLMEngine("claude-opus-4-7"), name="edit")
plan2 = Agent(
    engine=Plan(
        Step("research"),
        Step("edit", context=from_agent("research")),  # reads from store at runtime
    ),
    tools=[researcher.as_tool("research"), editor.as_tool("edit")],
    store=store,
)

# 4) Agent with sources= sees the live store on every call (LLM reads the whole store).
monitor = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    name="monitor",
    sources=[store],
)
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

    - ``store=`` is a **first-class Agent parameter** (like ``memory=`` and
      ``session=``). Pass the same ``Store`` instance to multiple agents to
      create a shared blackboard.
    - Every Agent automatically writes its last successful output to
      ``"__agent_output__:{agent.name}"`` after each run. Use
      ``from_agent("name")`` in a Plan step to read that value at runtime.
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
- [Sentinels](sentinels.md) — `from_agent` and `from_memory` read from Store/Memory.
- [Checkpoint & resume](checkpoint.md) — `Plan` uses `Store` under the hood.
