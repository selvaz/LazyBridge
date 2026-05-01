# Plan

**`Plan` is the engine for declared, multi-step pipelines.**  Every
step has a named target, a typed input/output, an explicit data source
(the sentinel system), and optionally writes its payload into a `Store`
bucket the rest of the pipeline can read.  All of that is validated at
**construction time** — `PlanCompileError` fires before any LLM call.

**Five data-flow primitives** cover almost every shape:

* **`task=`** — the prompt instruction.  Use a literal string for a
  specialised step ("Rank these hits by relevance") and let the data
  flow through `context=`.
* **`context=`** — the data source(s).  A sentinel, a string, or a
  list mixing both.  Multi-source synthesis with no combiner step.
* **`output=Model`** — declares the step's payload type.  Pure typing;
  no longer overloaded with routing.
* **`writes="key"`** — persists the step's payload to `store["key"]`.
* **`parallel=True`** — concurrent band membership.

For **conditional flow** (one step runs OR another, depending on
state), see the dedicated routing section below.

A typical pipeline shape — solid arrows are linear progression,
dashed arrows are conditional routes:

```mermaid
flowchart LR
    A[search\nroutes={...}] -.->|predicate true| F[apology]
    A --> B[rank]
    B --> C[fact_check]
    C --> D[write\nwrites=draft]
    D --> E[(Store\nkey=draft)]
    style A fill:#e8f5e9
    style B fill:#e8f5e9
    style C fill:#fff3e0
    style D fill:#e8f5e9
    style F fill:#ffebee
```

**Stay at `Agent.chain`** if your pipeline is a straight line of text
hand-offs — `chain` is sugar for the simplest `Plan` shape.  Move to
`Plan` once you need typed hand-offs, conditional routing, parallel
bands, named writes, or crash-resume.

### Routing — choose the path explicitly

Routing is **declared on the `Step`**, not hidden in the output model.
That means you can tell at the call site whether a step branches and
where, without reading any Pydantic class.

**Two forms, exactly one (or neither) per step:**

#### Form A — `routes={...}` (predicate map)

Use when **your code** decides the branch.  Each key is the name of a
target step; each value is a callable `(Envelope) -> bool`.  After the
step runs, predicates are evaluated in declared order; the first one
that returns truthy makes Plan jump to that target.  If none match,
linear progression continues.

```python
Step(searcher, name="search", output=Hits,
     routes={
         # If search found nothing, jump to the apology step.
         "empty": lambda env: not env.payload.items,
     })
```

The branch table is right there at the call site — no need to inspect
`Hits`.

#### Form B — `routes_by="field"` (LLM-decided via Literal field)

Use when **the LLM** decides the branch.  Pass the name of an
attribute on the structured output; the framework reads
`env.payload.<name>` and, if it's a string matching a step name,
jumps there.

```python
class Decision(BaseModel):
    kind: Literal["urgent", "normal"] | None = None  # field type IS the contract

Step(classifier, name="classify", output=Decision,
     routes_by="kind")              # reads env.payload.kind
```

Compile-time validation:

* The field must exist on the output model.
* It must be `Literal[...]` (or `Literal[...] | None`) — anything
  else fails at construction.
* Every Literal value must match a declared step name.
* `None` (or any value not matching a step) means "don't route" —
  linear progression continues.

`routes_by` is the right tool when you want the *LLM* to read the
input and pick a branch, e.g. classification or self-correction
loops.

#### Behaviour after a route — *detour*, not "no fall-through"

Routing is a **detour**.  When step A routes to step X, Plan runs X
and then **resumes linear progression from X's declared position**.
There is no "no fall-through" mode.  Practical implications:

* **Terminal early-out** = put the routed-to step **last** in the
  declared step list.  After it runs, linear progression has nowhere
  left to go → Plan ends.
* **Mid-pipeline detour** = route to a step in the middle.  After it
  runs, the pipeline continues from there to the end.
* **Loop / self-correction** = route back to an earlier step.  When
  the route condition stops firing, the loop exits and linear
  progression resumes.  `max_iterations` is the safety net.

#### Three rules to memorise

* **Visible at the call site** — every routing decision lives on the
  `Step(...)` line via `routes=` or `routes_by=`.
* **Compile-time validated** — unknown target names, malformed
  Literal types, predicate-not-callable: all caught by
  `PlanCompiler` before any LLM call.
* **Detour, not termination** — the routed-to step runs, then linear
  progression resumes from its declared position.  Place terminal
  steps last in declared order.

### Crash-resume — *what survives across runs* (separate concern)

Crash-resume is **durability across run boundaries**, completely
independent of routing.  Configure with
`Plan(store=..., checkpoint_key="...", resume=True)`: after every step
the plan state is written to the store via `compare_and_swap`.
Re-running with the same `checkpoint_key` and `resume=True` picks up
at the failed step.

```python
plan = Plan(
    Step(extract,   name="extract",   writes="raw"),
    Step(transform, name="transform", writes="clean"),
    Step(load,      name="load"),
    store=Store(db="etl.sqlite"),
    checkpoint_key="run-2026-04-30",
    resume=True,
)
```

Three rules:

* **No control-flow effect** — checkpointing has nothing to do with
  routing.  It only decides which steps a *future* run skips, never
  which step runs now.
* **Only `writes=` survives** — in-memory `history` is rebuilt empty
  on resume.  Any state a future run needs must be persisted via
  `writes="key"`.
* **Concurrent runs serialised by CAS** — `on_concurrent="fail"`
  (default) raises `ConcurrentPlanRunError` on collision.  For
  fan-out workflows, use `on_concurrent="fork"` (incompatible with
  `resume=True`).

### Routing vs crash-resume — side-by-side

|                    | Routing                                       | Crash-resume                                            |
| ---                | ---                                           | ---                                                     |
| Controls           | Which step runs **next**                      | What survives a **process restart**                     |
| Trigger            | `Step(routes={...})` or `Step(routes_by="…")` | `Plan(store=…, checkpoint_key=…, resume=True)`          |
| Time scale         | Within one `Plan.run`                         | Across separate `Plan.run` invocations                  |
| Validation surface | `PlanCompileError` at construction            | `ConcurrentPlanRunError` at runtime CAS                 |
| Required           | A predicate map OR a Literal-typed field      | An instantiated `Store` + a checkpoint key              |
| Combinable         | Yes — production pipelines normally use both  | Yes                                                     |

## Example

The patterns below cover the full surface.  Each is a minimal,
self-contained shape; combine them as needed in real pipelines.

### 1. Linear typed pipeline with terminal-fork routing

The everyday shape: search → rank → write, with an early-out to
``apology`` when the searcher returns nothing.  ``routes=`` on the
search step makes the branch explicit at the call site; ``apology``
is last in declared order so linear fall-through never reaches it.

```python
from pydantic import BaseModel
from lazybridge import Agent, Plan, Step, Store, from_prev, from_step

class Hits(BaseModel):
    items: list[str]

class Ranked(BaseModel):
    top: list[str]

store = Store(db="research.sqlite")

plan = Plan(
    Step(searcher, name="search",
         task="Search the web for the user's topic.",
         writes="hits", output=Hits,
         # The branch table is explicit at the call site.
         # The predicate gets the FULL Envelope; .payload is typed.
         routes={
             "empty": lambda env: not env.payload.items,
         }),
    Step(ranker,   name="rank",
         task="Rank these search hits by relevance; return the top 5.",
         context=from_prev,
         output=Ranked),
    Step(writer,   name="write",
         task="Write a 200-word brief from the ranked items below.",
         context=from_step("rank")),
    Step(apology,  name="empty",                    # ← terminal: last in order
         task="Apologise that no results were found and suggest broader terms."),
    store=store, checkpoint_key="research", resume=True,
)
print(Agent.from_engine(plan)("AI trends April 2026").text())
```

### 2. LLM-decided routing via a Literal field

When the LLM should pick the branch — e.g. classification — use
``routes_by="<field>"``.  The output model declares the legal values
as a ``Literal[...]``; the compiler checks they're real step names.

```python
from typing import Literal
from pydantic import BaseModel

class Triage(BaseModel):
    summary: str
    severity: Literal["urgent", "normal", "spam"] | None = None

plan = Plan(
    Step(classifier, name="classify",
         task="Classify the incoming ticket. Set severity to "
              "'urgent' for outages, 'spam' for marketing, otherwise 'normal'.",
         output=Triage,
         routes_by="severity"),    # reads env.payload.severity
    Step(escalator,  name="urgent",
         task="Page the on-call team and open a P0."),
    Step(triager,    name="normal",
         task="Add to the support backlog with the summary."),
    Step(closer,     name="spam",  # last in order → terminal
         task="Close the ticket as spam."),
)
```

If the LLM sets ``severity=None`` (or omits it), no routing happens
and linear progression continues — useful when "I'm not sure" should
fall through to a default branch.

### 3. Self-correction loop

A reviewer step routes back to the writer when the draft fails
quality checks.  ``max_iterations`` is the safety net.  Feedback is
threaded back to the writer via a shared `Store` (the reviewer
writes its verdict, the writer reads it through `sources=[store]`)
because Plan sentinels can't reference forward steps.

```python
from pydantic import BaseModel
from lazybridge import Plan, Step, Store, from_start, from_prev, from_step

class Verdict(BaseModel):
    feedback: str
    approved: bool

store = Store()      # in-memory; the reviewer's writes flow live to writer's sources

plan = Plan(
    Step(writer,    name="write",
         task="Draft a 200-word answer.  If a 'verdict' is in the store, "
              "rewrite the previous draft addressing the feedback.",
         context=from_start,
         sources=[store],                      # reads any prior reviewer verdict live
         writes="draft"),
    Step(reviewer,  name="review",
         task="Score the draft for accuracy, tone, length.  "
              "Set approved=True only if all three pass.",
         context=from_prev,
         output=Verdict,
         writes="verdict",                     # writer reads this via sources= next loop
         routes={
             # Loop back to the writer when the reviewer rejected.
             "write": lambda env: not env.payload.approved,
         }),
    Step(publisher, name="publish",
         task="Final-format and publish the approved draft.",
         context=from_step("write")),
    store=store,
    max_iterations=8,                          # cap the loop
)
```

When ``approved=True`` the predicate is False → linear → publish.
When ``approved=False`` the predicate is True → route back to write
→ writer sees the feedback through `sources=[store]` and re-drafts.

### 4. Fan-out + fan-in with explicit aggregation

Three independent searchers run concurrently; the join step reads
ALL of them via `from_parallel_all`.  Routing primitives are not
involved — parallel bands have their own control flow.

```python
from lazybridge import Plan, Step, Store, from_parallel_all

plan = Plan(
    Step(anthropic_search, name="search_a", parallel=True, writes="findings_a"),
    Step(openai_search,    name="search_o", parallel=True, writes="findings_o"),
    Step(google_search,    name="search_g", parallel=True, writes="findings_g"),

    Step(synthesiser, name="synth",
         task="Compare the three search results; flag agreement and disagreement.",
         context=from_parallel_all("search_a"),
         writes="brief"),
    store=Store(db="weekly.sqlite"),
)
```

### 5. Map-reduce — N items processed in parallel, then summarised

```python
def make_pipeline(items: list[str]) -> Plan:
    branches = [
        Step(item_processor, name=f"proc_{i}", parallel=True,
             task=f"Run end-of-day analysis on {item}.",
             writes=f"out_{i}",
             output=ItemResult)
        for i, item in enumerate(items)
    ]
    return Plan(
        *branches,
        Step(summariser, name="summary",
             task="Summarise the per-ticker analyses into a bulleted report.",
             context=from_parallel_all(branches[0].name),
             output=Report),
    )

agent = Agent.from_engine(make_pipeline(["AAPL", "GOOG", "MSFT"]))
agent("end-of-day market scan")
```

### 6. Crash-resume after a failed step (no routing)

Pure crash-resume — every step runs in declared order; ``resume=True``
picks up at the failed step.

```python
class ValidationReport(BaseModel):
    rejected_rows: list[int]
    accepted_rows: int

store = Store(db="pipeline.sqlite")
plan = Plan(
    Step(extract,   name="extract",   writes="raw"),
    Step(transform, name="transform",
         task="Transform raw records to the canonical schema; drop nulls.",
         context=from_prev,
         writes="clean"),
    Step(validate,  name="validate",
         task="Verify business rules; flag rows that fail.",
         context=from_prev,
         writes="verdict",
         output=ValidationReport),
    Step(load,      name="load",
         task="Load the cleaned records into the warehouse.",
         context=from_step("transform")),
    store=store,
    checkpoint_key="etl-2026-04-30",
    resume=True,
)
Agent.from_engine(plan)("today's batch")
```

### 7. Concurrent fan-out runs with `on_concurrent="fork"`

Many runs of the same pipeline (one per ticker / seed / variant) on
the same `Store` without colliding — each run claims its own
isolated keyspace.

```python
from concurrent.futures import ThreadPoolExecutor

store = Store(db="backtest.sqlite")
plan = Plan(
    Step(load_data,    name="load",  writes="prices"),
    Step(run_strategy, name="run",
         task="Execute the strategy over the price series; emit a trade log.",
         context=from_prev,
         writes="trades"),
    Step(score,        name="score",
         task="Compute Sharpe, max-drawdown, and total return.",
         context=from_prev,
         output=Metrics),
    store=store,
    checkpoint_key="backtest",
    on_concurrent="fork",
)

agent = Agent.from_engine(plan)
with ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(lambda t: agent(t), ["AAPL", "GOOG", "MSFT", "AMZN"]))
```

### 8. Multi-source synthesis with `context=[...]`

```python
from lazybridge import Plan, Step, from_step, from_prev

plan = Plan(
    Step(searcher,      name="search",   writes="hits"),
    Step(policy_loader, name="policy",   task="Load the 2026 acceptable-use policy."),
    Step(competitor,    name="bench",    task="Find three relevant prior posts."),

    # Pulls from three upstream steps PLUS a literal style note.
    Step(synthesiser, name="synth",
         task="Draft a 300-word brief; cite each source explicitly.",
         context=[
             from_step("search"),
             from_step("policy"),
             from_step("bench"),
             "Style: neutral, third-person, no superlatives.",
         ],
         output=Brief),

    Step(publisher, name="publish", task=from_prev),
)
```

### 9. Mixed step targets — agents, tools, callables

`Step.target` is uniform: anything that can be a tool can be a step.

```python
def normalise(text: str) -> str:
    """Pure Python — no LLM, instant."""
    return text.strip().lower()

plan = Plan(
    Step(researcher, name="search"),                  # Agent
    Step(normalise,  name="clean", task=from_prev),   # plain callable: task=sentinel is natural
    Step("score",    name="score", task=from_prev),   # tool name (resolved on the wrapping Agent)
    Step(writer,     name="write",
         task="Write a 150-word brief.",
         context=from_step("clean")),
)

Agent.from_engine(plan, tools=[score_tool])("…")
```

## Pitfalls

- Forgetting ``output=Model`` on a step where you want typed payload
  hand-off — the next step sees a plain string.  Declare ``output=``
  where you need types; ``routes_by`` requires it for compile-time
  validation.
- Cyclic step references or unknown step names → ``PlanCompileError``
  at construction.  This includes ``routes={"ghost": ...}`` and
  ``routes_by`` Literal values that don't match any step name.
- Routing CYCLES (``A → B → A``) are NOT a compile error (they may be
  intentional, e.g. self-correction loops).  They surface at runtime
  as ``MaxIterationsExceeded`` once ``max_iterations`` fires.
- ``resume=True`` without ``store=`` is a silent no-op (no checkpoint
  to read or write). Pass both, and pick a ``checkpoint_key``.
- ``on_concurrent="fork"`` + ``resume=True`` is a configuration error
  (raises at construction). Fork mode gives each run its own key, so
  there's no shared checkpoint to resume from.
- ``from_parallel_all("X")`` requires ``X`` to be the FIRST member of
  its parallel band — the engine walks forward from there.
- A step that fails persists a ``status="failed"`` checkpoint pointing
  back at itself.  Subsequent ``resume=True`` runs retry that step.
- Plan writes go through the *same* store as application writes —
  namespace your keys (e.g. prefix with the pipeline name) so a
  step's ``writes="results"`` doesn't collide with an unrelated
  agent's stored value.

!!! note "API reference"

    Plan(
        *steps: Step,
        max_iterations: int = 100,
        store: Store | None = None,
        checkpoint_key: str | None = None,
        resume: bool = False,
        on_concurrent: Literal["fail", "fork"] = "fail",
    ) -> Engine
    
    Step(
        target: str | Callable | Agent,                    # tool name, function, or Agent
        task: Sentinel | str = from_prev,                  # where my input comes from
        context: Sentinel | str
               | list[Sentinel | str] | None = None,        # one OR many side-context sources
        sources: list = (),                                 # live-view objects with .text()
        writes: str | None = None,                         # Store key under which payload is saved
        input: type = Any,
        output: type = str,
        parallel: bool = False,
        name: str | None = None,
        # ── Routing — exactly one (or neither): ───────────────────────────
        routes: dict[str, Callable[[Envelope], bool]] | None = None,
        routes_by: str | None = None,
    )
    
    # Sentinels — see the dedicated guide for full semantics.
    from_prev                    # previous step's output (default)
    from_start                   # original user task
    from_step("name")            # named prior step
    from_parallel("name")        # named parallel branch
    from_parallel_all("name")    # aggregate every branch in a parallel band, labelled-text join
    
    PlanCompileError             # raised at Agent construction if the DAG is invalid
    ConcurrentPlanRunError       # raised by CAS when two runs share a checkpoint_key
    PlanState                    # checkpoint shape: plan_id, current_step, next_step, store, history, status
    StepResult                   # single step record: step_name, envelope, ts
    
    Usage: Agent(engine=Plan(Step(a), Step(b)))

!!! warning "Rules & invariants"

    - ``max_iterations`` caps total step executions per ``run`` to guard
      against runaway routing loops (default 100). Raise it for legitimate
      long plans; lower it during dev to fail fast. Hitting the cap returns
      a ``MaxIterationsExceeded`` error envelope (not a crash).
    - Step names are unique. ``PlanCompileError`` fires at Agent construction
      on duplicate names, dangling ``from_step`` / ``from_parallel`` /
      ``from_parallel_all`` references, forward references, mid-band
      ``from_parallel_all`` start, unknown ``routes=`` targets, or
      malformed ``routes_by=`` fields.
    - ``output=SomeModel`` is **only** for typing the step's payload.  It
      does NOT control routing — a `next` field on the model is just a
      regular field.  Routing is declared **on the Step**, see below.
    - ``parallel=True`` marks a branch in a concurrent band.  The engine
      bundles consecutive ``parallel=True`` steps and dispatches them via
      ``asyncio.gather``.  **Atomicity:** if any branch errors, no
      ``writes`` from the band are applied — a future ``resume=True`` re-runs
      the whole band cleanly.  Routing primitives (`routes` / `routes_by`)
      are ignored on parallel branches.
    - ``writes="key"`` stores the step's payload into ``store[key]``.
      Required for checkpoint data and for downstream agents reading via
      ``sources=[store]``. Plan writes go through the same store as
      application writes; namespace your keys.
    - ``checkpoint_key`` + ``store`` enable state persistence after every
      step via ``compare_and_swap``. ``resume=True`` reads the checkpoint
      and picks up at the next un-run step (failed runs restart from the
      failing step, not the next).
    - Concurrent runs sharing a ``checkpoint_key`` are serialised via CAS:
        * ``on_concurrent="fail"`` (default) — second run raises
          ``ConcurrentPlanRunError`` on collision (single-writer; pair with
          ``resume=True`` for crash recovery).
        * ``on_concurrent="fork"`` — each run claims an isolated
          ``f"{checkpoint_key}:{run_uid}"`` keyspace (fan-out workflows).
          Incompatible with ``resume=True``.
    - ``context=`` accepts **a single sentinel/string OR a list of them**.
      A list lets a step pull data from N upstream steps without an
      intermediate combiner — items resolve independently and the parts
      are joined with blank-line separators (same shape as ``sources``).
      Each list item is validated at compile time; mixing sentinels with
      literal strings is supported.

## See also

- [Sentinels](sentinels.md) — full semantics of `from_prev` / `from_step` / `from_parallel` / `from_parallel_all`.
- [Parallel plan steps](parallel-steps.md) — concurrent bands and join shapes.
- [Checkpoint & resume](checkpoint.md) — store mechanics and `PlanState`.
- [Plan serialization](plan-serialize.md) — round-trip a Plan through JSON for cross-process re-use.
- [SupervisorEngine](supervisor.md) — alternative engine for HIL pipelines.
- [verify=](verify.md) — judge placement at the engine level.
- [Operations checklist](../guides/operations.md) — production knobs (timeout, fallback, on_concurrent).
