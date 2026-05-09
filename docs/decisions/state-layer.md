# State layer

> **State: `Memory`, `Store`, or `sources=`?**

Three composable mechanisms. Pick by *scope* and *lifetime*; they
also stack freely on the same agent.

## Decision tree

```text
Conversation history for one agent across multiple calls?
    → Memory(strategy="auto")
      Agent(engine=LLMEngine("…"), memory=memory)

Shared key-value blackboard across multiple agents or runs?
    → Store(db="path.sqlite")    # or Store() for in-process
      Agent(engine=LLMEngine("…"), store=store)
      # writes via Step(writes="key") or agent auto-write

Static documents / live views injected into context at every call?
    → Agent(engine=LLMEngine("…"), sources=[memory, store, "policy text"])

Multiple patterns at once?
    → Yes — they compose freely.
      Agent(
          engine=LLMEngine("…"),
          memory=conversation_memory,
          store=team_store,
          sources=[policy_text, live_metric_provider],
      )
```

## Quick reference

| Use | When |
|---|---|
| `Memory` | Per-agent conversation history; auto-compresses with `strategy="auto"` |
| `Store` | Cross-agent / cross-run state; `db=` for SQLite persistence |
| `sources=[…]` | Live view injected into the system prompt; accepts callables, `Memory`, `Store`, raw strings |

## Notes

- **`Memory` is per-agent by default.** Pass the same `Memory`
  instance to two agents to share their conversation; the second
  agent typically reads via `sources=[shared_memory]` so it
  doesn't accidentally write turns it didn't author.
- **`Store` is the durable counterpart to `Memory`.** Memory is
  "what should the model see in the next prompt"; Store is "what
  should survive a crash". They're complementary, not
  substitutable.
- **`sources=` is a live view.** Each item with `.text()` is
  re-evaluated at call time, so a `Store` passed in `sources=`
  reflects the most-recent values without any explicit refresh.
- **Inside a `Plan`, prefer step writes (`Step(writes="key")`)
  + `from_step("step_name")` sentinels** for in-pipeline data
  flow. Keep `Memory` and `sources=` for cross-call state and
  live document injection.

## See also

- [Memory](../guides/mid/memory.md) — strategies (auto / sliding
  / summary / none), summariser configuration.
- [Store](../guides/mid/store.md) — in-memory vs SQLite,
  auto-write keys, `from_agent("…")` reads.
- [Sentinels](../guides/full/sentinels.md) — `from_step` /
  `from_agent` / `from_memory` for plumbing state into Plan
  steps.
