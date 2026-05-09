# Store

A thread-safe key-value blackboard, in-memory by default and SQLite-backed
when you want durability. The Store is what `Plan` checkpoints land in,
what `from_agent("alias")` reads from, and what you reach for whenever
state must survive a crash or a process boundary.

## Signature

```python
from lazybridge import Store

Store(db=None)
  # db=None    → in-memory (deep-copied dict; lost on exit)
  # db="path"  → SQLite, WAL mode, thread-safe, persistent

# Methods
store.write(key, value, *, agent_id=None)
store.read(key, default=None)
store.read_entry(key)            # StoreEntry | None — value plus metadata
store.read_all()                 # dict[str, Any]
store.keys()                     # list[str]
store.delete(key)
store.clear()
store.to_text(keys=None)         # render as "key: <json>" lines for sources=
```

`StoreEntry` is a dataclass `(key, value, written_at, agent_id)`.

## Synopsis

`Store` is the durable / shared counterpart to [`Memory`](memory.md).
Where `Memory` is "what the model should see in the next prompt", `Store`
is "what should survive a crash, or be read by another agent / process /
machine".

Three things flow through it automatically:

- **Pipeline checkpoints.** A `Plan` with `store=` and a
  `checkpoint_key=` writes step results so a crashed run can resume.
- **Agent outputs.** Every agent writes its last result under
  `"__agent_output__:{name}"` after a successful run, so
  `from_agent("name")` sentinels (and code that wants to peek
  out-of-band) can read it.
- **Plan step `writes=` declarations.** `Step("research", writes="hits")`
  copies the step's payload to the key `"hits"` once it succeeds.

You can also use it as a plain blackboard — write whatever keys you
like, read them from anywhere that holds the same `Store`. SQLite mode
makes it safe to share across threads and concurrent agent runs.

## When to use it

- **Crash-resumable pipelines.** Pass `store=Store(db="run.sqlite")` and
  a `checkpoint_key=` to a `Plan`; `resume=True` on the next run picks
  up at the failed step.
- **Cross-agent shared blackboard.** A fan-out of researchers writes
  intermediate facts; a downstream synthesiser reads them via
  `sources=[store]` (live view) or `from_agent("…")` (per-agent
  output).
- **Out-of-band inspection.** A monitor / dashboard process opens the
  same SQLite file read-only and queries the live state without
  joining the agent run.
- **Cross-process artefact handoff.** One service writes computed
  artefacts; another service reads them. Use a shared filesystem path
  or a pointer (URL, file path) as the value.

## When NOT to use it

- **Conversation context for an LLM call.** That's `Memory`'s job.
  Don't shovel the whole `Store` into prompts every turn — pass the
  store as a `sources=` (live view) or read specific keys with
  `from_agent` / `from_step`.
- **High-throughput counters or queues.** Each write commits
  immediately; there's no transactional batch. For that workload reach
  for a real database or queue.
- **Large binary blobs.** Values are JSON-encoded on write. Store a
  filesystem path / URL as the value and let the consumer read the
  blob directly.

## Example

```python
from lazybridge import (
    Agent,
    LLMEngine,
    Plan,
    Step,
    Store,
    from_agent,
)


store = Store(db="research.sqlite")


# 1) Plan step writes a result via Step(writes=).
researcher = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    store=store,                       # required for from_agent later
    name="research",
)
writer = Agent(
    engine=LLMEngine("gpt-4o"),
    name="write",
)

pipeline = Agent(
    engine=Plan(
        Step("research", writes="hits"),   # store["hits"] = step payload
        Step("write"),
    ),
    tools=[researcher, writer],
    store=store,
)
pipeline("AI trends in 2026")
print(store.read("hits"))


# 2) Agents auto-write their output to Store after each run.
#    Key: "__agent_output__:{name}".
researcher("AI trends 2026")
print(store.read("__agent_output__:research"))


# 3) from_agent("name") reads that output in a Plan step.
#    IMPORTANT: store= must be on the SOURCE AGENT (researcher), not
#    just the pipeline.
editor = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    name="edit",
)
plan_with_handoff = Agent(
    engine=Plan(
        Step("research"),
        Step("edit", context=from_agent("research")),
    ),
    tools=[researcher, editor],
)
plan_with_handoff("Topic: bees")


# 4) Agent with sources=[store] sees the live store on every call.
monitor = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    sources=[store],
)
print(monitor("what's the current state?").text())
```

## Pitfalls

- **`Store(db=":memory:")`** is **not** the same as `Store()`. The
  former opens an in-memory SQLite (connection-scoped); the latter
  uses a Python dict. Use `Store()` for in-process state and the
  filename form for durability.
- **Auto-write key uses the alias, not `agent.name`.** When you wrap
  an agent as `agent.as_tool("alias")`, the auto-write key is
  `"__agent_output__:alias"`. `from_agent("alias")` reads the same
  key. If you query the store directly, use the alias.
- **`store=` must be on the source agent**, not just on the pipeline.
  `from_agent("research")` reads `__agent_output__:research`; that
  key is only written if the researcher itself was constructed with
  `store=store`. PlanCompiler rejects `from_agent(...)` references
  whose source agent has no store attached.
- **Reads return deep copies.** Mutating `store.read("k")` does not
  change the stored value. The in-memory backend matches the SQLite
  copy-on-write semantics on purpose.
- **Values are JSON-encoded** via `json.dumps(default=str)`. Non-JSON
  types are stringified on the way in. Prefer primitives and
  `pydantic_model.model_dump()` over raw class instances.
- **`store.to_text()` materialises the entire keyspace** by default;
  pass `keys=[...]` to limit the slice when the store has thousands
  of keys.
- **Each write commits immediately.** There's no transactional batch.
  If you need atomic multi-key updates, layer them yourself or use a
  real RDBMS.

## See also

- [Memory](memory.md) — the in-prompt counterpart.
- [Session](session.md) — observability of writes (events emitted on
  agent finish carry the auto-write metadata).
- *Guides → Full → Checkpoint & resume* (Phase 3) — how `Plan` uses
  `Store` for crash-resilient pipelines.
- *Guides → Full → Sentinels* (Phase 3) — `from_agent("name")` and
  `from_step("name")` resolve against the Store at run time.
