# Nested pipelines — horizontal composition

Most introductions to LazyBridge compose agents **vertically** —
`Agent.chain(a, b, c)` for a one-shot linear pipeline, or
`Plan(Step("a"), Step("b"), Step("c"))` when you need named steps,
typed hand-offs, sentinels, or routing.  That's enough for many
applications.

This page is the **horizontal** counterpart: how to compose
pipelines side-by-side, and how to nest one pipeline inside another.
The same `Agent = Engine + Tools + State` mental model applies — the
trick is that `Plan` itself is an engine, so an agent whose engine
is a `Plan` is a perfectly valid `Step.target` for an **outer**
plan.  Pipelines compose recursively.

## Vertical recap (one paragraph)

Vertical composition produces a single sequence of steps:

```python
from lazybridge import Agent, LLMEngine, Plan, Step

researcher = Agent(engine=LLMEngine("claude-opus-4-7"), name="research")
writer     = Agent(engine=LLMEngine("gpt-5.4-mini"),    name="write")
editor     = Agent(engine=LLMEngine("claude-opus-4-7"), name="edit")

pipeline = Agent(
    engine=Plan(Step("research"), Step("write"), Step("edit")),
    tools=[researcher, writer, editor],
    name="vertical_pipeline",
)
```

This is the shape covered by [Chain](../mid/chain.md) and
[Plan](plan.md).  Everything below assumes you've read those.

## Horizontal: a Plan whose step is a Plan

The simplest horizontal composition: one `Step` in an outer `Plan`
targets an **agent whose own engine is a `Plan`**.  The outer plan
sees a single tool call; under the hood, the sub-plan runs its own
multi-step pipeline.

```python
from lazybridge import Agent, LLMEngine, Plan, Step, from_step

# --- Sub-pipeline: a self-contained "research" capability ---
search    = Agent(engine=LLMEngine("claude-haiku-4-5"), name="search")
summarise = Agent(engine=LLMEngine("claude-haiku-4-5"), name="summarise")

research_pipeline = Agent(
    engine=Plan(
        Step("search"),
        Step("summarise"),
    ),
    tools=[search, summarise],
    name="research",         # outer plan references this name
)

# --- Outer pipeline: research → write → edit ---
writer = Agent(engine=LLMEngine("gpt-5.4-mini"),    name="write")
editor = Agent(engine=LLMEngine("claude-opus-4-7"), name="edit")

pipeline = Agent(
    engine=Plan(
        Step("research"),                                           # nested plan
        Step("write", context=from_step("research")),
        Step("edit"),
    ),
    tools=[research_pipeline, writer, editor],
    name="article_pipeline",
)

result = pipeline("AI agent frameworks, April 2026")
print(result.text())
print(f"total cost: ${result.metadata.cost_usd}")
```

What's happening:

- `research_pipeline` is a regular `Agent`.  Its `engine` happens
  to be a `Plan`, but from the outer plan's perspective it's just a
  named tool — `Step("research")` resolves to it via the
  `tools=[research_pipeline, ...]` map.
- Cost / token telemetry from the sub-plan rolls up into the outer
  envelope's `metadata.nested_*` fields.  The single
  `result.metadata.cost_usd` you read at the end is the **whole tree**.
- Sentinels (`from_step("research")`) work transparently — the outer
  plan sees `research`'s final envelope, not its intermediate steps.

### When to nest a plan vs flatten the steps

| Shape | When |
|---|---|
| Flat `Plan(Step("search"), Step("summarise"), Step("write"))` | The steps are co-evolving — one team owns them, the data flow is straightforward, and you don't want a separate test surface for "research". |
| Nested `Plan(Step("research"), Step("write"))` where `research` is its own `Agent(engine=Plan(...))` | You want **isolation**: the research sub-pipeline ships as a reusable unit, has its own tests, its own checkpoint key if needed, and can be swapped for a different implementation (e.g. a `SupervisorEngine` or a custom engine) without touching the outer plan. |

## Horizontal + parallel: parallel bands of sub-pipelines

`Step(..., parallel=True)` runs sibling steps concurrently.  When
each "step" is itself a sub-pipeline, you get **N independent
pipelines running side by side**, with their outputs aggregated via
`from_parallel_all(...)`.

```python
from lazybridge import Agent, LLMEngine, Plan, Step, from_parallel_all

# Three independent research pipelines, each a Plan of its own.
def make_research_pipeline(name: str, source_agent: Agent) -> Agent:
    return Agent(
        engine=Plan(
            Step("search", target=source_agent),
            Step("summarise"),
        ),
        tools=[source_agent, Agent(engine=LLMEngine("claude-haiku-4-5"), name="summarise")],
        name=name,
    )

web_research      = make_research_pipeline("web",      Agent(engine=LLMEngine("claude-opus-4-7"), name="web_search"))
academic_research = make_research_pipeline("academic", Agent(engine=LLMEngine("claude-opus-4-7"), name="academic_search"))
internal_research = make_research_pipeline("internal", Agent(engine=LLMEngine("claude-opus-4-7"), name="internal_search"))

synthesiser = Agent(engine=LLMEngine("claude-opus-4-7"), name="synthesise")

pipeline = Agent(
    engine=Plan(
        Step("web",      parallel=True),   # branch 1: full sub-pipeline
        Step("academic", parallel=True),   # branch 2: full sub-pipeline
        Step("internal", parallel=True),   # branch 3: full sub-pipeline
        Step("synthesise",
             context=from_parallel_all("web")),   # joins all 3 parallel siblings
    ),
    tools=[web_research, academic_research, internal_research, synthesiser],
    name="multi_source_brief",
)
```

What this gives you:

- **Three independent sub-pipelines** run concurrently via
  `asyncio.gather`.  The framework caps concurrency at
  `Plan(max_parallel_steps=…)` (defaults to unbounded).
- **`from_parallel_all("web")`** resolves to the labelled-text join
  of every contiguous `parallel=True` sibling starting at `web` — so
  `synthesise` sees one input containing all three branches' outputs,
  each labelled with its branch name.
- **First-error short-circuit**: if any branch errors, the outer
  envelope carries that error and the remaining branches' results
  are dropped.  Use a `verify=` judge or `fallback=` agent on
  individual branches if you need graceful degradation.

### When to use parallel bands vs `Agent.parallel(...)`

| Shape | When |
|---|---|
| `Agent.parallel(a, b, c)` | You want a **single envelope** out (labelled-text join).  No further plan structure needed.  See [Parallel](../mid/parallel.md). |
| `Step("a", parallel=True) … Step("d", context=from_parallel_all("a"))` | The parallel work is **one stage of a larger plan** — there's setup before, aggregation after, sentinels across, and you want crash-resume on the whole thing.  This page. |

## Horizontal: agent-as-tool with LLM-decided dispatch

The previous two patterns are **deterministic** — the plan decides
which sub-pipelines run.  The third horizontal shape hands the
decision to an LLM: pass sub-pipelines in the **outer agent's
`tools=[...]`** and let the model pick.

```python
from lazybridge import Agent, LLMEngine

# Same three sub-pipelines as above (each is an Agent(engine=Plan(...))).
orchestrator = Agent(
    engine=LLMEngine(
        "claude-opus-4-7",
        system=(
            "You have three research sub-pipelines available.  Call only "
            "the ones that match the user's question; combine their results."
        ),
    ),
    tools=[web_research, academic_research, internal_research],
    name="adaptive_research",
)

orchestrator("Compare LangGraph and CrewAI for our use case.")
```

What's different:

- **The LLM chooses** which sub-pipelines to call, in what order,
  and whether to call multiple in parallel (the engine emits parallel
  tool calls automatically when the model requests them in the same
  turn).
- **No `Plan` at the outer layer** — the engine is `LLMEngine`, so
  there's no compile-time DAG validation, no checkpointing on the
  outer call, no sentinels.  You trade auditability for adaptability.
- **Sub-pipeline internals stay deterministic.** Each sub-agent's
  own `Plan` still validates at construction, still produces
  predictable token cost, still respects its own `checkpoint_key=`.

### Choosing between the three horizontal shapes

```text
Do all sub-pipelines always need to run?
├── Yes, sequentially  → outer Plan with one Step per sub-pipeline
├── Yes, concurrently  → outer Plan with parallel=True bands + from_parallel_all
└── No — let the model decide → outer LLMEngine with sub-pipelines in tools=[...]
```

## Diagram — vertical vs horizontal

```text
VERTICAL (chain / linear Plan)
─────────────────────────────
  start → [search] → [summarise] → [write] → [edit] → end
  one process, one path, total latency = sum of steps


HORIZONTAL — Plan-of-Plans
──────────────────────────
  start → ╔════════════ research ════════════╗ → [write] → [edit] → end
          ║ [search] → [summarise]           ║
          ╚══════════════════════════════════╝
  sub-pipeline is one tool to the outer plan; sentinels traverse the boundary


HORIZONTAL — parallel bands of sub-pipelines
────────────────────────────────────────────
              ╔════════ web ════════╗
              ║ [search]→[summarise]║
              ╠══════ academic ═════╣
  start  →    ║ [search]→[summarise]║ →  [synthesise(from_parallel_all)]  → end
              ╠══════ internal ═════╣
              ║ [search]→[summarise]║
              ╚═════════════════════╝
  three sub-pipelines run concurrently; aggregator joins their outputs


HORIZONTAL — LLM-decided dispatch
─────────────────────────────────
                     ┌── web ────────┐
                     │ [Plan...]     │
  start → [LLM] ──── │ academic      │ ── (LLM may call 1-N, may parallel)
                     │ [Plan...]     │
                     │ internal      │
                     │ [Plan...]     │
                     └───────────────┘
  no outer Plan; engine emits tool calls based on the user prompt
```

## Cost and observability across nested boundaries

Every horizontal shape preserves the cost roll-up:

- Inner pipelines write their tokens + cost + latency to their own
  envelope's `metadata.*`.
- When the outer plan invokes the inner agent as a tool, the inner
  envelope's metadata is folded into the outer envelope's
  `metadata.nested_*` fields.
- `Session.usage_summary()` walks the whole tree and gives you one
  number per provider / model.

There is no "I called three sub-pipelines and have no idea what it
cost" failure mode — the rollup is automatic.

## Pitfalls

- **Naming collisions are silent.**  If two sub-agents share the same
  `name=`, the outer tool map registers a single entry (with a
  `UserWarning`) and the outer plan resolves both `Step("search")`
  references to the same target.  Always name sub-agents distinctly,
  even when they live in separate sub-pipelines.
- **`Plan(max_iterations=N)` is per-plan, not transitive.** A nested
  `Plan` that loops forever is invisible to the outer plan's
  iteration counter.  Set sensible caps on every level.
- **Checkpoint keys must be unique across the tree.**  If both the
  outer plan and an inner plan write to the same `Store` with the
  same `checkpoint_key=`, their state collides.  Namespace them
  (e.g. `"article/research"`, `"article/write"`).
- **Sentinels are per-plan.** `from_step("research")` in the outer
  plan resolves to the inner agent's **final** envelope — you cannot
  reach into `from_step("research.search")` to read the inner step's
  output.  If the outer plan needs an inner step's value, surface it
  via the inner plan's `writes=` to a shared `Store` and read it
  with `from_agent("research.search")`.
- **First-error short-circuit at each level.** A failing branch in
  an inner parallel band aborts the inner plan, which aborts the
  outer step, which aborts the outer plan — unless one of the
  intermediate agents has a `fallback=` or the failing step is
  wrapped with `verify=`.

## See also

- [Chain](../mid/chain.md) — the vertical baseline.
- [Parallel](../mid/parallel.md) — single-level `Agent.parallel(...)`
  fan-out; the lighter sibling of parallel plan bands.
- [Plan](plan.md) — the outer-pipeline surface this page composes.
- [Parallel plan steps](parallel-plan-steps.md) — the
  `parallel=True` mechanism in depth.
- [Sentinels](sentinels.md) — `from_step`, `from_parallel_all`,
  `from_agent` semantics across plan boundaries.
- [Checkpoint & resume](checkpoint.md) — applies independently per
  plan in the tree; namespace your `checkpoint_key=`.
- [Recipes → Supervisor pattern](../../recipes/supervisor-pattern.md)
  — an LLM-decided dispatch over sub-agents, the runnable form of
  the third horizontal shape.
