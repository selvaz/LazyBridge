# Recipe: Plan with typed steps and crash resume

**Tier:** Full  
**Goal:** Run a multi-step pipeline where each step has a declared output type, steps can
conditionally route to each other, and the whole thing resumes from the failing step after a crash.

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
    Step(searcher, name="search",     writes="results", output=SearchResult),
    Step(analyser, name="analyse",    task=from_prev,   output=Analysis),   # receives search's Envelope
    Step(writer,   name="write",      task=from_step("analyse")),           # jumps to analyse's Envelope by name
    Step(apology,  name="no_results"),                                      # reached only if SearchResult.next == "no_results"
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
from lazybridge import from_parallel

plan = Plan(
    Step(searcher,     name="search"),
    Step(us_analyser,  name="us",  task=from_prev,         parallel=True),
    Step(eu_analyser,  name="eu",  task=from_prev,         parallel=True),
    Step(summariser,   name="join", task=from_parallel("us")),  # receives list[Envelope]
)
```

See [Parallel plan steps](../guides/parallel-steps.md) for the full reference.

## Next

- [Sentinels](../guides/sentinels.md) — `from_prev`, `from_start`, `from_step`, `from_parallel` in detail
- [Checkpoint & resume](../guides/checkpoint.md) — store mechanics and `PlanState` shape
- [Parallel plan steps](../guides/parallel-steps.md) — concurrent branches and joins
- [SupervisorEngine](../guides/supervisor.md) — add a human REPL step to any pipeline
