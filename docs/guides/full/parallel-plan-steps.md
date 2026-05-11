# Parallel plan steps

`Step(parallel=True)` marks a step as a member of a concurrent band:
the engine bundles consecutive `parallel=True` steps and dispatches
them via `asyncio.gather`. After the band finishes, the next non-
parallel step acts as the join. Use a `from_parallel(...)` /
`from_parallel_all(...)` sentinel on the join to read the branch
outputs.

For application-level scripted fan-out → `list[Envelope]` (no Plan,
no aggregation), use [`Agent.parallel`](../mid/parallel.md) instead.

## Signature

```python
from lazybridge import Step, from_parallel, from_parallel_all

# Mark a step as a band member.
Step(target, *, parallel=True, name="...", writes=None, output=None, ...)


# Sentinels for the join step that reads branch outputs.
from_parallel("branch_name")          # one specific branch
from_parallel_all("first_branch")     # aggregate the whole band as labelled text
```

## Synopsis

A "parallel band" is one or more **consecutive** `Step(parallel=True)`
declarations. The engine groups them into a single dispatch unit and
runs them via `asyncio.gather`; control flow waits until every branch
finishes (success, error, or timeout). The first non-parallel step
that follows the band is the **join** — it reads branch outputs via
the parallel sentinels.

The idiomatic shape is:

```python
Plan(
    Step(a, name="a", parallel=True),
    Step(b, name="b", parallel=True),
    Step(c, name="c", parallel=True),
    Step(join, name="join",
         task="Synthesise the three branches.",
         context=[from_parallel("a"), from_parallel("b"), from_parallel("c")]),
)
```

The list-context form lets you mix branches with literal strings
(e.g. style notes); `from_parallel_all("a")` is the one-line
equivalent that produces a single labelled-text join.

### Atomicity

If any branch in the band errors, **no `writes=` from the band are
applied** — not even those of succeeded siblings. The first-error
envelope propagates as the band's outcome, and the checkpoint points
to the band's **first** step. A subsequent `resume=True` re-runs the
whole band cleanly. This is intentional: resuming mid-band would
leave earlier branches' Store keys stale relative to the re-run
ones.

## When to use it

- **Independent steps that can run concurrently** and the next step
  needs all of their results — multi-source research, multi-region
  fetches, multi-model ensembles.
- **Map-reduce shapes.** N similar branches process N inputs in
  parallel, then a summariser folds the results.
- **Parallel side-effects with shared visibility.** Three searchers
  each `writes="findings_<x>"` — downstream steps with
  `sources=[store]` see all three live.

## When NOT to use it

- **Application-level fan-out where you just want
  `list[Envelope]`.** Use [`Agent.parallel`](../mid/parallel.md) —
  no Plan, no aggregation.
- **Conditional concurrency.** Routing primitives are silently
  ignored on parallel branches — the whole band runs every time.
  If only one branch should run conditionally, route to a
  non-parallel step instead and decide there.
- **Branches with data dependencies on each other.** If branch B
  needs branch A's output, they belong in sequence. Parallel
  bands are for genuinely independent work.
- **Side effects that aren't crash-safe to re-run.** Atomicity
  re-runs the entire band on any branch failure. If a branch
  writes to an external system that doesn't tolerate duplicate
  writes, gate it with idempotency keys or run it sequentially.

## Example

```python
from pydantic import BaseModel

from lazybridge import (
    Agent,
    LLMEngine,
    Plan,
    Step,
    Store,
    from_parallel,
    from_parallel_all,
)


def search_anthropic(query: str) -> str:
    """Search Anthropic's bulletins."""
    return "..."


def search_openai(query: str) -> str:
    """Search OpenAI's bulletins."""
    return "..."


def search_google(query: str) -> str:
    """Search Google's bulletins."""
    return "..."


anthropic_search = Agent(engine=LLMEngine("deepseek-v4-flash"), tools=[search_anthropic], name="search_a")
openai_search = Agent(engine=LLMEngine("deepseek-v4-flash"), tools=[search_openai], name="search_o")
google_search = Agent(engine=LLMEngine("deepseek-v4-flash"), tools=[search_google], name="search_g")
synthesiser = Agent(engine=LLMEngine("deepseek-v4-flash"), name="synth")


# 1) Three branches → join with explicit per-branch sentinels.
store = Store(db="monitor.sqlite")
plan = Agent(
    engine=Plan(
        Step("search_a", parallel=True, writes="findings_a"),
        Step("search_o", parallel=True, writes="findings_o"),
        Step("search_g", parallel=True, writes="findings_g"),
        Step("synth",
             task="Compare the three sources; flag agreement and disagreement.",
             context=[
                 from_parallel("search_a"),
                 from_parallel("search_o"),
                 from_parallel("search_g"),
                 "Style: terse, factual, no superlatives.",
             ],
             writes="brief"),
        store=store,
    ),
    tools=[anthropic_search, openai_search, google_search, synthesiser],
)
plan("framework update — April 2026")


# 2) Same shape, one-line aggregation via from_parallel_all.
plan = Agent(
    engine=Plan(
        Step("search_a", parallel=True, writes="findings_a"),
        Step("search_o", parallel=True, writes="findings_o"),
        Step("search_g", parallel=True, writes="findings_g"),
        Step("synth",
             task="Compare the three sources; flag agreement and disagreement.",
             context=from_parallel_all("search_a"),     # the FIRST branch's name
             writes="brief"),
    ),
    tools=[anthropic_search, openai_search, google_search, synthesiser],
)


# 3) Map-reduce — N items processed in parallel, summarised at the end.
class ItemResult(BaseModel):
    item: str
    score: float


class Report(BaseModel):
    summary: str
    items: list[ItemResult]


def make_pipeline(items: list[str]) -> Plan:
    branches = [
        Step(item_processor, name=f"proc_{i}", parallel=True,
             task=f"Run end-of-day analysis on {item}.",
             writes=f"out_{i}", output=ItemResult)
        for i, item in enumerate(items)
    ]
    return Plan(
        *branches,
        Step(summariser, name="summary",
             task="Summarise the per-ticker analyses.",
             context=from_parallel_all(branches[0].name),
             output=Report),
    )


agent = Agent(engine=make_pipeline(["AAPL", "GOOG", "MSFT"]))
agent("end-of-day market scan")
```

## Pitfalls

- **Only *consecutive* `parallel=True` steps are bundled.** A
  non-parallel step in between starts a new band. Keep parallel
  siblings contiguous.
- **`from_parallel_all("X")` requires X to be the FIRST step of
  its band.** Mid-band references fail at construction.
  PlanCompiler walks forward from the named step; if an earlier
  parallel sibling exists, the reference is rejected.
- **The join is implicit — the first non-parallel step after the
  band.** If you forget to read branches via `from_parallel(...)`
  or `from_parallel_all(...)`, the join sees only `from_prev`,
  which resolves to the last completed branch (timing-dependent
  and rarely what you want).
- **Routing is ignored on parallel branches.** A `parallel=True`
  step's `routes=` / `routes_by=` is silently dropped — parallel
  bands have their own control flow. Set routing on the join
  instead.
- **Atomicity re-runs the whole band on any branch failure.**
  Branches that write to external systems (HTTP POSTs, DB
  inserts) need idempotency keys to be safe under retry.
- **`max_iterations` counts each branch.** A wide band of N
  parallel steps consumes N from the iteration budget. Raise
  `Plan(max_iterations=...)` accordingly for very wide fan-outs.
- **Per-branch errors don't propagate as exceptions.** The first
  failing branch's error envelope becomes the band's outcome;
  every other branch's result is discarded along with its
  `writes=`. If you need partial-success semantics, run the
  branches as `Agent.parallel(...)` and inspect the
  `list[Envelope]` yourself.

## See also

- [Plan](plan.md) — the engine that orchestrates parallel bands.
- [Step](step.md) — `parallel=True` is one of the step's three
  concurrency-and-naming fields.
- [Sentinels](sentinels.md) — `from_parallel("name")` and
  `from_parallel_all("name")` semantics.
- [Parallel](../mid/parallel.md) — application-level scripted
  fan-out with `list[Envelope]` return; complementary, not
  redundant.
- [Checkpoint & resume](checkpoint.md) — band-level atomicity
  drives the "next_step points to the band's first step" rule on
  failure.
