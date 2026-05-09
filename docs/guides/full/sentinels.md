# Sentinels

How a `Step` says where its input comes from. Without sentinels you'd
thread arguments manually through every step; with them, the data
flow is one declaration per step. All sentinel references are
validated at `Plan` construction time — typos become `PlanCompileError`
before any LLM call.

## Signature

```python
from lazybridge import (
    from_prev,                     # singleton: previous step's output (default)
    from_start,                    # singleton: original user task
    from_step,                     # callable: from_step("name") — named prior step
    from_parallel,                 # callable: from_parallel("name") — alias for from_step
    from_parallel_all,             # callable: from_parallel_all("name") — aggregate a parallel band
    from_memory,                   # callable: from_memory("name") — agent's live conversation history
    from_agent,                    # callable: from_agent("name") — agent's last output from Store
)


# Valid placements on a Step:
Step(target, task=<sentinel>)
Step(target, context=<sentinel>)
Step(target, context=[<sentinel>, <sentinel>, "literal string"])
```

### The seven sentinels

| Sentinel | Reads | Resolved | Compile-time validation |
|---|---|---|---|
| `from_prev` | The previous step's output | Plan execution history | — (always available) |
| `from_start` | The Plan's original input | Plan execution history | — |
| `from_step("n")` | A named prior step's output | Plan execution history | `"n"` must name an earlier step |
| `from_parallel("n")` | Same as `from_step("n")` — different name signals intent | Plan execution history | **Not separately validated.** A typo'd `from_parallel("foo")` surfaces at runtime as a `warnings.warn(...)` fallback to the start envelope, not a `PlanCompileError`. Prefer `from_step` when you want compile-time safety. |
| `from_parallel_all("n")` | Every consecutive parallel sibling starting at `"n"`, joined as labelled text | Plan execution history | `"n"` must exist, come earlier, be `parallel=True`, AND be the FIRST member of its band (the step immediately before it must be non-parallel) |
| `from_memory("n")` | The live `Memory` of the agent registered under `"n"` | Step execution time (live) | The named tool must be an agent with `memory=` attached |
| `from_agent("n")` | The last output of agent `"n"` from a shared `Store` | Step execution time (live) | The tool must be an agent (`returns_envelope=True`) AND the source agent must have `store=` attached |

## Synopsis

Sentinels split into two categories:

**Plan-only** — resolve against the Plan's execution history at
step dispatch time. They cannot reach outside the current `Plan.run`:

- `from_prev` — the workhorse. The default for `task=`. Each step
  reads the one before it.
- `from_start` — the original user task. Use it when a step needs
  the input regardless of intermediate processing (verification,
  re-routing, fresh framing).
- `from_step("name")` — name-keyed access to any earlier step's
  output. The compiler validates the name; a typo fails at
  construction.
- `from_parallel("name")` — alias for `from_step` that reads better
  at the call site when the referenced step ran concurrently with
  siblings.
- `from_parallel_all("name")` — aggregator. Folds every consecutive
  `parallel=True` step starting at `name` into one envelope whose
  payload is a labelled-text join (`"[step_a]\n<text>\n\n[step_b]\n<text>"`).
  See *Parallel plan steps* (Phase 3b) for the full mechanics.

**Universal** — resolve at step *execution* time and work both
inside a `Plan` and standalone:

- `from_memory("name")` — reads the *live* `Memory` of the agent
  registered under `name`. Always reflects the most recent
  conversation history; absent or empty memory contributes
  nothing (silent no-op).
- `from_agent("name")` — reads the *last output* of agent `name`
  from a shared `Store`. Every successful agent run writes to
  `__agent_output__:{alias}`; `from_agent` reads it back. Works
  across runs and outside the current `Plan`.

The store key is always the **alias** passed to `as_tool("alias")`,
not the agent's internal `name=`.

## When to use which

- **Inside the same Plan, prefer `from_step("name")`** over
  `from_agent("name")`. `from_step` reads from in-memory step
  history (no Store required), is validated more tightly at
  compile time, and is cheaper.
- **Use `from_agent("name")` only when** you need the agent's
  output independent of the current Plan's history: across Plan
  runs, in a standalone LLM orchestrator with no step history,
  or in a step that needs the output of an agent invoked
  *outside* this Plan.
- **Use `from_memory("name")` for** conversation continuity —
  when a downstream agent should see what an upstream agent has
  *been talking about* (memory), not just its last output
  (Store).
- **Use `from_start` when** the original user task is the right
  prompt for a step that's deep in the pipeline (verifier,
  apology branch, fresh-framing summariser).

## Example

```python
from pydantic import BaseModel

from lazybridge import (
    Agent,
    LLMEngine,
    Memory,
    Plan,
    Step,
    Store,
    from_agent,
    from_memory,
    from_prev,
    from_start,
    from_step,
)


store = Store(db="pipeline.sqlite")
mem = Memory(strategy="summary")


# Researcher with memory + store — feeds both `from_memory` and
# `from_agent` references downstream.
researcher = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    memory=mem,
    store=store,
    name="research",
)
fact_checker = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    store=store,
    name="fact_check",
)
writer = Agent(
    engine=LLMEngine("gpt-4o"),
    name="write",
)


# 1) Mixed sentinels in a single Plan.
plan = Agent(
    engine=Plan(
        Step("research"),
        # fact_checker sees the researcher's output as task,
        # the original user task as context.
        Step("fact_check",
             task=from_prev,
             context=from_start),
        # writer sees the researcher's live memory PLUS the fact_checker output.
        Step("write",
             context=[from_memory("research"), from_step("fact_check")]),
    ),
    tools=[researcher, fact_checker, writer],
    store=store,
)
plan("AI trends April 2026")


# 2) Multi-source synthesis via context=[...] — no combiner step needed.
class Brief(BaseModel):
    title: str
    body: str


policy_loader = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    name="policy",
)
synthesiser = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    name="synth",
    output=Brief,
)


plan2 = Agent(
    engine=Plan(
        Step("research"),
        Step("policy"),
        Step("synth",
             task="Draft a brief citing both sources.",
             context=[
                 from_step("research"),
                 from_step("policy"),
                 "Style: neutral, third person, no superlatives.",
             ]),
    ),
    tools=[researcher, policy_loader, synthesiser],
)


# 3) from_agent across runs — read what the researcher produced
#    in a previous Plan execution.
standalone = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    tools=[researcher],
    store=store,
)
standalone("find AI trends")

# Later, in a different Plan / process, with the same store:
later_plan = Agent(
    engine=Plan(
        Step("write",
             task="Write a follow-up brief based on prior research.",
             context=from_agent("research")),
    ),
    tools=[writer],
    store=store,
)
```

## Pitfalls

- **`from_prev` after a parallel band returns the *join step's*
  output, not one of the branches.** Use
  `from_parallel("<branch-name>")` for a specific branch or
  `from_parallel_all("<first-branch-name>")` for the aggregate.
- **Sentinels are module-level imports.** Don't shadow them with
  local variables of the same name (`from_prev = "literal"` is a
  bug waiting to happen).
- **A `str` passed as `task=` is a LITERAL, not a sentinel
  reference.** `task="from_prev"` sets the step's task to the
  string `"from_prev"`. Use the imported `from_prev` symbol.
- **`from_memory("n")` is a silent no-op when the agent hasn't
  run.** Empty memory contributes nothing, no error. If you want
  fail-fast behaviour, validate the memory yourself before the
  Plan dispatches the step.
- **`from_agent("n")` requires `store=` on the source agent.**
  PlanCompiler rejects it at construction time when the named
  agent has no store attached. The error message names the
  offending agent — pass `store=...` to that agent, not just to
  the Plan.
- **`from_agent("n")` requires the tool to be an agent**
  (`returns_envelope=True`), not a plain function. PlanCompiler
  rejects plain-function targets at construction time.
- **The Store key is the *alias*, not the agent's `name=`.**
  When the agent is wrapped via `agent.as_tool("alias")`, the
  auto-write key is `__agent_output__:alias`. `from_agent("alias")`
  reads it back. Mixing these up is one of the most common
  sources of confusion — keep aliases stable across runs that
  share a Store.
- **`from_parallel_all("n")` requires `n` to be the FIRST step of
  its parallel band.** Mid-band references fail at construction
  with a clear error.

## See also

- [Plan](plan.md) — the engine that interprets sentinels.
- [Step](step.md) — the surface that consumes sentinels via
  `task=` and `context=`.
- [Routing](routing.md) — sentinels are about *data flow*;
  routing is about *control flow*. They don't overlap.
- [Store](../mid/store.md) — the backing store for `from_agent`
  and the receiver of `Step(writes=...)`.
- [Memory](../mid/memory.md) — the conversation-history layer
  that `from_memory` reads live.
- *Guides → Full → Parallel plan steps* (Phase 3b) —
  `from_parallel_all` aggregation in full.
