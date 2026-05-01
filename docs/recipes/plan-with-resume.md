# Recipe: Plan with typed steps and crash resume

**Tier:** Full  
**Goal:** Run a multi-step pipeline where each step has a declared output type, steps can
conditionally route to each other, and the whole thing resumes from the failing step after a crash.

!!! note "Two orthogonal features in one recipe"
    This recipe combines **routing** (`next: Literal[...]` on a Pydantic
    output, decides which step runs next) and **crash-resume** (`store=`
    + `checkpoint_key=` + `resume=True`, decides what survives a process
    restart).  They are independent — see the
    [Plan guide](../guides/plan.md#two-control-mechanisms-that-are-easy-to-confuse)
    for each on its own.

## The pattern

```python
from lazybridge import Agent, Plan, Step, Store, from_prev, from_step
from pydantic import BaseModel
from typing import Literal

# --- Declare output types for each step that needs them ---

# "next" field → Plan routes to the matching step name after this step completes.
class SearchResult(BaseModel):
    items: list[str]
    next: Literal["analyse", "no_results"] = "analyse"

class Analysis(BaseModel):
    summary: str
    confidence: float

# --- Agents for each step ---

searcher = Agent("claude-opus-4-7", name="searcher", tools=[web_search])
analyser = Agent("claude-opus-4-7", name="analyser")
writer   = Agent("claude-opus-4-7", name="writer")
apology  = Agent("claude-opus-4-7", name="apology")   # fallback if search found nothing

# --- Persistent store for checkpointing ---

store = Store(db="pipeline.sqlite")

# --- Plan construction validates the DAG — no LLM call happens yet ---

plan = Plan(
    Step(searcher, name="search",
         writes="results",
         output=SearchResult),
    # Idiomatic shape: an explicit ``task=`` instruction; upstream data
    # flows through ``context=``.  The agent doesn't have to guess what
    # to do with the envelope it received.
    Step(analyser, name="analyse",
         task="Analyse the search results; assign a confidence in [0,1].",
         context=from_prev,
         output=Analysis),
    Step(writer,   name="write",
         task="Write a 250-word brief from the analyser's findings; cite items.",
         context=from_step("analyse")),
    Step(apology,  name="no_results",
         task="Apologise that no results were found and suggest broader queries."),
    store=store,
    checkpoint_key="weekly-brief",
    resume=True,           # skip steps that already completed successfully
    max_iterations=20,
)

# --- Agent.from_engine wraps the plan; ("task") starts execution ---

result = Agent.from_engine(plan)("AI developments this week")
print(result.text())
```

## What resume does

| Run | Scenario | Behaviour |
|---|---|---|
| Run 1 | "analyse" crashes | Checkpoint saved: `status="failed"`, `next_step="analyse"` |
| Run 2 | `resume=True` | "search" is skipped (already done); "analyse" retries |
| Run 3 | Plan already `"done"` | Returns cached last-step result immediately |

The checkpoint key (`"weekly-brief"`) namespaces state in the Store — change it to start fresh.

## Parallel branches

Mark steps with `parallel=True` to run them concurrently, then join with `from_parallel`:

```python
from lazybridge import from_parallel, from_prev

plan = Plan(
    Step(searcher,     name="search"),
    Step(us_analyser,  name="us",
         task="Score the US-relevant items.",
         context=from_prev,
         parallel=True),
    Step(eu_analyser,  name="eu",
         task="Score the EU-relevant items.",
         context=from_prev,
         parallel=True),
    # Join: explicit instruction + list-context that pulls both branches
    # without an intermediate combiner.
    Step(summariser,   name="join",
         task="Summarise US and EU findings into one global report.",
         context=[from_parallel("us"), from_parallel("eu")]),
)
```

See [Parallel plan steps](../guides/parallel-steps.md) for the full reference.

## Pitfalls

- **`PlanCompileError` fires at `Plan(...)` time.** A misspelled step
  name, a forward reference in `from_step`, or a parallel-band misuse
  surfaces *before any LLM call*. Treat the construction error as the
  contract: fix the DAG, don't paper over it at runtime.
- **Concurrent runs on the same `checkpoint_key` are serialised.**
  `on_concurrent="fail"` (default) raises `ConcurrentPlanRunError` on
  collision — pick distinct keys per run, or pass
  `on_concurrent="fork"` to namespace each run under a uid suffix.
  `resume=True` is incompatible with `fork`.
- **`writes` is the only persisted state.** In-memory `history` does
  not survive across processes — only values written into named
  `writes` buckets. Design steps so that the bucket carries everything
  a future resume needs.
- **Parallel bands are atomic.** If any branch in a band errors, the
  framework applies *no* writes from that band. A future `resume=True`
  re-runs the whole band cleanly rather than double-applying side
  effects from the siblings that succeeded.

## Next

- [Sentinels](../guides/sentinels.md) — `from_prev`, `from_start`, `from_step`, `from_parallel` in detail
- [Checkpoint & resume](../guides/checkpoint.md) — store mechanics and `PlanState` shape
- [Parallel plan steps](../guides/parallel-steps.md) — concurrent branches and joins
- [SupervisorEngine](../guides/supervisor.md) — add a human REPL step to any pipeline
- [Operations checklist](../guides/operations.md) — production knobs (timeout, fallback, resume)
