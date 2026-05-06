# LazyBridge — Full tier
**Use this when** you need a declared, multi-step pipeline: typed hand-offs, conditional routing, crash recovery via checkpoint/resume, or OTel/JSON observability.

**Stay at Mid** if your pipeline is a straight line with no typed models between steps and you don't need resume semantics.

## Plan

**signature**

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
    after_branches: str | None = None,                 # exclusive-branch rejoin point
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

**rules**

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

**example**

The patterns below cover the full surface.  Each is a minimal,
self-contained shape; combine them as needed in real pipelines.

### 1. Linear typed pipeline with terminal-fork routing

The everyday shape: search → rank → write, with an early-out to
``apology`` when the searcher returns nothing.  ``routes=`` on the
search step makes the branch explicit at the call site; ``apology``
is last in declared order so linear fall-through never reaches it.

```python
from pydantic import BaseModel
from lazybridge import Agent, Plan, Step, Store, from_prev, from_step, when

class Hits(BaseModel):
    items: list[str]

class Ranked(BaseModel):
    top: list[str]

store = Store(db="research.sqlite")

plan = Plan(
    Step(searcher, name="search",
         task="Search the web for the user's topic.",
         writes="hits", output=Hits,
         # ``when`` DSL: when ``items`` is empty, route to "apology"
         # instead of falling through to "rank".  No lambda, no
         # ``env.payload.<name>`` plumbing.
         routes={"apology": when.field("items").empty()}),
    Step(ranker,        name="rank",
         task="Rank these search hits by relevance; return the top 5.",
         context=from_prev,
         output=Ranked),
    Step(writer,        name="write",
         task="Write a 200-word brief from the ranked items below.",
         context=from_step("rank")),
    Step(apology_agent, name="apology",                # ← terminal: last in declared order
         task="Apologise that no results were found and suggest broader terms."),
    store=store, checkpoint_key="research", resume=True,
)
print(Agent.from_engine(plan)("AI trends April 2026").text())
```

### 2. LLM-decided routing via a Literal field (exclusive branches)

When the LLM should pick the branch — e.g. classification — use
``routes_by="<field>"``.  The output model declares the legal values
as a ``Literal[...]``; the compiler checks they're real step names.
Add ``after_branches="step"`` so only the matched branch runs and
execution resumes at the named rejoin point; without it the routed-to
step would run and then linear progression would continue through the
remaining branch steps.

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
         routes_by="severity",     # reads env.payload.severity
         after_branches="archive"), # skip siblings; always land at archive
    Step(escalator,  name="urgent",
         task="Page the on-call team and open a P0."),
    Step(triager,    name="normal",
         task="Add to the support backlog with the summary."),
    Step(closer,     name="spam",
         task="Close the ticket as spam."),
    Step(archiver,   name="archive",  # always runs after whichever branch ran
         task="Log the resolved ticket to the audit archive."),
)
```

If the LLM sets ``severity=None`` (or omits it), no routing happens
and linear progression continues to ``urgent`` (next declared step).
Use ``after_branches`` whenever you want exactly one branch to run;
omit it only when detour/fall-through behaviour is intentional (e.g.
self-correction loops).

### 3. Self-correction loop

A reviewer step routes back to the writer when the draft fails
quality checks.  ``max_iterations`` is the safety net.  Feedback is
threaded back to the writer via a shared `Store` (the reviewer
writes its verdict, the writer reads it through `sources=[store]`)
because Plan sentinels can't reference forward steps.

```python
from pydantic import BaseModel
from lazybridge import Plan, Step, Store, from_start, from_prev, from_step, when

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
         # Loop back to the writer when the reviewer rejected.
         routes={"write": when.field("approved").is_(False)}),
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

**pitfalls**

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
- A sequential step that fails persists a ``status="failed"`` checkpoint
  pointing back at itself; subsequent ``resume=True`` runs retry that
  step.  A parallel band that fails points the checkpoint at the band's
  **first** step — the whole band re-runs so all sibling ``writes`` are
  produced consistently.
- Plan writes go through the *same* store as application writes —
  namespace your keys (e.g. prefix with the pipeline name) so a
  step's ``writes="results"`` doesn't collide with an unrelated
  agent's stored value.

## Sentinels (from_prev / from_start / from_step / from_parallel)

**signature**

from_prev                    # singleton — previous step's output (default)
from_start                   # singleton — original user task
from_step(name: str)         # named prior step's output
from_parallel(name: str)     # named parallel branch's output
from_parallel_all(name: str) # aggregate every branch in a parallel band;
                             # payload is a labelled-text string, same as task

# Used on Step(..., task=<sentinel>) or Step(..., context=<sentinel>).

**rules**

- ``from_prev`` (default): the previous step's output becomes the next
  step's task. This is real chain semantics — each step sees what its
  predecessor produced, not the original user input.
- ``from_start``: explicit reference to the initial envelope. Use it
  when you want a step to operate on the original user request
  regardless of what preceded it.
- ``from_step("n")``: reach back to a specific prior step's result.
  PlanCompiler verifies ``"n"`` names an earlier step, else raises.
- ``from_parallel("n")``: alias for ``from_step`` intended for parallel
  branch joins. Indicates to readers that the step being referred to
  ran concurrently with siblings.
- ``from_parallel_all("n")``: aggregates every consecutive parallel step
  starting at ``"n"`` into one Envelope whose ``task`` and ``payload``
  are both a labelled-text join (``"[name]\\n<text>\\n\\n..."``).
  ``"n"`` must be the FIRST step of the band; the compile-time check
  rejects mid-band references.
- A plain string passed as ``task=`` is used verbatim — useful for
  hard-coded prompts at intermediate steps.
- ``context=`` accepts a single sentinel/string OR a **list** of them.
  Each list item resolves independently; the parts join with
  blank-line separators in the step's ``Envelope.context``.  Mix
  sentinels with literal strings to inject fixed boilerplate alongside
  upstream data without an intermediate combiner step.

**example**

```python
from lazybridge import Plan, Step, from_prev, from_start, from_step

plan = Plan(
    Step(researcher,    name="research",  output=Hits),
    # Each step has an explicit task instruction; the upstream data
    # flows through ``context=`` (the idiomatic shape).
    Step(fact_checker,  name="check",
         task="Score each item for factual correctness; list any rejects.",
         context=from_prev),                                  # check researcher's output
    Step(writer,        name="write",
         task="Draft a 200-word answer to the user's task.",
         context=from_start),                                  # writer sees ORIGINAL user task
    Step(editor,        name="edit",
         task="Polish the draft; remove items the fact-checker flagged.",
         context=[from_step("write"), from_step("check")]),   # multi-source via list-context
)

# context= can carry MANY sources at once (no combiner step needed).
plan2 = Plan(
    Step(researcher,    name="research"),
    Step(policy_loader, name="policy"),
    Step(synthesiser,   name="synth",
         task="Draft a brief that cites both sources and follows the style note.",
         context=[
             from_step("research"),
             from_step("policy"),
             "Style: neutral, third person, no superlatives.",
         ]),
)
```

**pitfalls**

- ``from_prev`` after a parallel branch returns the join step's output,
  not one of the branches. Use ``from_parallel("<branch-name>")`` for a
  specific branch.
- Sentinels are module-level imports; don't shadow them with local
  variables of the same name.
- When passing a ``str`` as ``task=``, it's treated as a LITERAL, not a
  sentinel. Don't write ``task="from_prev"`` expecting the sentinel.

## Parallel plan steps

**signature**

Step(target, *, parallel: bool = False, name: str | None = None, ...)
from_parallel(name: str) -> Sentinel

# Typical shape: N parallel branches followed by a join step.
# Idiomatic: ``task=`` is the join's instruction (a literal); upstream
# branch outputs flow through ``context=`` — a list of sentinels reads
# all branches without an intermediate combiner.
Plan(
    Step(a, name="a", parallel=True),
    Step(b, name="b", parallel=True),
    Step(c, name="c", parallel=True),
    Step(join, name="join",
         task="Synthesise the three branches into one report.",
         context=[from_parallel("a"), from_parallel("b"), from_parallel("c")]),
)

**rules**

- ``parallel=True`` marks a step as a branch that runs concurrently
  with other consecutive parallel steps in the plan.
- The plan engine dispatches all consecutive ``parallel=True`` steps
  via ``asyncio.gather`` before proceeding.
- A non-parallel step immediately after parallel steps acts as an
  implicit join: it sees ``from_prev`` as the last completed branch's
  output; use ``from_parallel("name")`` to reach a specific branch.
- Parallel steps may have their own ``writes=`` — each branch's
  payload is persisted under the respective Store key.
- **Atomicity on failure**: if any branch errors, no ``writes`` from
  the band are applied (not even those of succeeded siblings), the
  first-error ``Envelope`` is returned, and the checkpoint points to
  the band's first step so a future ``resume=True`` re-runs the whole
  band cleanly.

**example**

```python
from lazybridge import Agent, Plan, Step, from_parallel, from_parallel_all, Store

store = Store(db="monitor.sqlite")

plan = Plan(
    # Three independent searchers fan out in parallel.
    Step(anthropic_search, name="search_a", parallel=True, writes="findings_a"),
    Step(openai_search,    name="search_o", parallel=True, writes="findings_o"),
    Step(google_search,    name="search_g", parallel=True, writes="findings_g"),

    # Join — explicit task instruction; the three branch outputs flow
    # in via the list-context.  ``from_parallel_all("search_a")`` would
    # also work and produces a single labelled-text join, but the list
    # form is more flexible (e.g. add a literal style note: see below).
    Step(synthesiser, name="synth",
         task="Compare the three sources; flag agreement and disagreement.",
         context=[from_parallel("search_a"),
                  from_parallel("search_o"),
                  from_parallel("search_g"),
                  "Style: terse, factual, no superlatives."],
         writes="plan"),

    store=store,
)
Agent.from_engine(plan)("framework update — April 2026")
```

**pitfalls**

- Interleaving parallel and sequential steps without care: the engine
  only bundles CONSECUTIVE ``parallel=True`` steps. Insert them in a
  run.
- Forgetting the join step — after N parallel steps the next
  non-parallel step IS the join. If you want all three outputs you
  must read them via ``from_parallel("…")`` on the join step;
  otherwise only ``from_prev`` (last completed) is visible.
- Checkpointing across a parallel block is coarse-grained: on failure the
  checkpoint's ``next_step`` is set to the band's **first** step, not the
  failing step. If branch A succeeds but B crashes, resume re-runs the
  whole band from the start so all sibling ``writes`` are produced
  consistently — resuming mid-band would leave earlier branches' kv stale.

## SupervisorEngine (ext.hil)

**signature**

SupervisorEngine(
    *,
    tools: list[Tool | Callable | Agent] = None,
    agents: list[Agent] = None,         # agents the human can retry
    store: Store | None = None,
    input_fn: Callable[[str], str] | None = None,
    ainput_fn: Callable[[str], Awaitable[str]] | None = None,
    timeout: float | None = None,
    default: str | None = None,
) -> Engine

Usage: Agent(engine=SupervisorEngine(tools=[...], agents=[researcher]))

REPL commands:
  continue [optional text]        accept; return to caller
  retry <agent>: <feedback>       re-run a registered agent with feedback
  store <key>                     print store[key]
  <tool>(<args>)                  invoke a registered tool

**rules**

- ``tools=`` accepts functions, Tool instances, and Agent instances
  uniformly — normalised to ``Tool`` at construction. Same contract as
  ``Agent(tools=...)``.
- The REPL runs on a worker thread so the caller's event loop is not
  blocked. ``input_fn`` is called there; use scripted inputs in tests.
- ``retry <agent>: <feedback>`` re-runs the named agent with the
  feedback appended to the task. The output replaces the current
  supervisor buffer.
- Unknown commands print help and re-prompt. ``continue`` is the only
  terminator.
- Session propagation: an Agent wrapping a SupervisorEngine receives
  session events for AGENT_START / AGENT_FINISH like any other engine.

**example**

```python
from lazybridge import Agent, Tool, Store
from lazybridge.ext.hil import SupervisorEngine

def search(query: str) -> str:
    """Search the web for query."""
    return f"hits for {query}"

researcher = Agent("claude-opus-4-7", name="researcher", tools=[search])
store = Store()
store.write("policy", "publish only peer-reviewed sources")

supervisor = Agent(
    engine=SupervisorEngine(
        tools=[search],
        agents=[researcher],
        store=store,
    ),
    name="supervisor",
)

writer = Agent("claude-opus-4-7", name="writer")

# Pipeline: researcher drafts → supervisor inspects / revises → writer finalises.
agents = [researcher, supervisor, writer]
pipeline = Agent.chain(*agents)
pipeline("AI policy brief")
```

**pitfalls**

- ``input_fn`` is called from a worker thread. If it accesses
  thread-unsafe state (like ``readline`` history), guard it.
- ``agents=`` expects v1 ``Agent`` instances. Duck-typed objects work
  if they expose ``__call__`` / ``run`` and a ``name`` attribute.
- The REPL blocks the human user — if ``timeout=None`` (the default),
  an unattended pipeline hangs forever. Set ``timeout=``+``default=``
  for unattended runs.
- Tool calls in the REPL go via ``run_sync``. If a tool's ``func`` is
  async, it's driven to completion automatically (post-v1 fix).

## Checkpoint & resume

**signature**

Plan(
    *steps,
    store: Store,
    checkpoint_key: str,
    resume: bool = False,
) -> Engine

# Persisted shape at store[checkpoint_key]:
#   {
#     "next_step": str | None,
#     "kv": {"writes_key": payload, ...},
#     "completed_steps": [str],
#     "status": "running" | "failed" | "done",
#   }

**rules**

- Checkpoint fires after each successful step and after each failed step.
- Success path: ``status="running"`` (next step pending) →
  ``status="done"`` when ``next_step is None``.
- Fail path: the failing step is NOT added to ``completed_steps``;
  ``status="failed"`` is written.  For a sequential step, ``next_step``
  points to the failing step itself so resume retries it.  For a
  parallel band, ``next_step`` points to the **band's first step** —
  the whole band must re-run cleanly so all sibling ``writes`` are
  produced; resuming mid-band would leave earlier siblings' kv stale.
- Success + ``resume=True`` + ``status="done"`` → short-circuit: Plan
  returns an Envelope with payload = cached ``kv``, without re-running.
- Checkpoint is JSON-encoded via ``Store.write``; ``writes=`` payloads
  must be JSON-serialisable (string, dict, Pydantic model via
  ``.model_dump()``).

**example**

```python
from lazybridge import Agent, Plan, Step, Store

store = Store(db="pipeline.sqlite")

def build_plan():
    return Plan(
        Step(researcher, name="search",  writes="hits"),
        Step(ranker,     name="rank",    writes="ranked"),
        Step(writer,     name="write",   writes="draft"),
        store=store,
        checkpoint_key="pipeline",
        resume=True,
    )

# Run 1 — crashes after rank: status="failed", next_step="write".
try:
    Agent.from_engine(build_plan())("AI trends")
except KeyboardInterrupt:
    pass

# Run 2 — resumes from the failing step; search+rank are not re-run.
Agent.from_engine(build_plan())("AI trends")

# Run 3 — plan is already "done": short-circuits, returns cached kv.
result = Agent.from_engine(build_plan())("AI trends")
print(result.payload)  # {"hits": ..., "ranked": ..., "draft": ...}
```

**pitfalls**

- Changing the Plan definition (adding/removing/renaming steps) and
  resuming from an old checkpoint will fail: the saved ``next_step``
  may no longer exist. Delete the checkpoint
  (``store.delete(checkpoint_key)``) after refactoring steps.
- Non-JSON-serialisable ``writes`` values (e.g. a file handle) are
  stringified silently via ``default=str``. Prefer primitives and
  Pydantic models.
- Resume does not re-inject the original session or exporters; pass the
  same ``session=`` + ``store=`` on every run for continuity.

## Exporters

**signature**

# Protocol
class EventExporter(Protocol):
    def export(self, event: dict) -> None: ...
    # Optional: close() is called by Session.close() when present.

# Built-ins shipped from ``lazybridge`` (core).
CallbackExporter(fn: Callable[[dict], None])
ConsoleExporter(*, stream=sys.stdout)            # pretty stdout
FilteredExporter(inner: EventExporter, *, event_types: set[str])
JsonFileExporter(path: str)                       # JSONL append
StructuredLogExporter(logger_name: str = "lazybridge")

# Built-in shipped from ``lazybridge.ext.otel`` (alpha extension).
from lazybridge.ext.otel import OTelExporter
OTelExporter(endpoint: str | None = None, *, exporter: Any | None = None)

Usage:
  Session(exporters=[
      ConsoleExporter(),
      JsonFileExporter(path="events.jsonl"),
      OTelExporter(endpoint="http://otelcol:4318"),
  ])

**rules**

- Each event is a ``dict`` with at minimum ``event_type``,
  ``session_id``, ``run_id`` (possibly ``None``). Engine-specific
  fields are merged in by the emitter.
- Exporters fire in registration order. An exception in one exporter
  does NOT block others; LazyBridge warns once per exporter instance
  and suppresses subsequent failures from the same exporter.
- ``FilteredExporter`` is a combinator — pass an inner exporter and a
  set of event_type strings to forward.
- ``OTelExporter`` requires ``pip install lazybridge[otel]``.  It emits
  spans conforming to the OpenTelemetry GenAI Semantic Conventions
  (``gen_ai.system``, ``gen_ai.usage.input_tokens``,
  ``gen_ai.tool.call.id``, …) so dashboards built for the standard
  render LazyBridge traces without translation.
- OTel span hierarchy mirrors the run:
    * ``invoke_agent <name>`` (root per ``Agent.run``)
    * ├ ``chat <model>`` (one per LLM round-trip)
    * └ ``execute_tool <tool>`` (one per tool invocation, correlated
      by ``tool_use_id``)
  Cross-agent parenting works automatically through OTel contextvars
  — an inner Agent invoked through a tool becomes a descendant of the
  outer tool span, no run-id chaining required.
- For high-throughput emit paths, pair ``Session(batched=True,
  on_full="hybrid")`` with the slower exporters (OTel, JSON-file).

**example**

```python
from lazybridge import (
    Agent, Session,
    ConsoleExporter, JsonFileExporter, FilteredExporter,
    CallbackExporter, EventType,
)
from lazybridge.ext.otel import OTelExporter

def on_alert(event):
    if event["event_type"] == EventType.TOOL_ERROR:
        alert_pagerduty(event)

sess = Session(
    db="events.sqlite",
    batched=True,                          # non-blocking emit
    exporters=[
        JsonFileExporter(path="run.jsonl"),
        FilteredExporter(
            CallbackExporter(fn=on_alert),
            event_types={EventType.TOOL_ERROR, EventType.AGENT_FINISH},
        ),
        OTelExporter(endpoint="http://otelcol:4318"),
    ],
)

Agent.chain(researcher, writer, session=sess)("…")
sess.flush()                               # drain the writer before exit
```

**pitfalls**

- Slow exporters block the engine when ``Session(batched=False)``
  (the default). Set ``batched=True`` for any exporter doing network
  I/O.
- Exporter exceptions warn once per instance and are suppressed
  afterwards. If only the first failure shows up, wrap with
  ``CallbackExporter(fn=print)`` while debugging.
- ``OTelExporter`` keeps a per-instance tracer rooted in its own
  ``TracerProvider`` so multiple exporters in one process don't
  fight. The provider is also installed globally as a best-effort
  default; you can supply your own and pass an in-memory exporter
  for tests.
- When ``Session.batched=True``, ``session.events.query(...)`` may
  return stale rows until ``session.flush()`` drains the writer.

## GraphSchema

**signature**

GraphSchema(session_id: str = "") -> GraphSchema

graph.add_agent(agent: Agent) -> None
graph.add_router(router) -> None
graph.add_edge(from_id, to_id, *, label="", kind=EdgeType.TOOL) -> None
graph.nodes() -> list[_BaseNode]
graph.edges() -> list[Edge]
graph.edges_from(node_id) / edges_to(node_id) -> list[Edge]

graph.to_dict() / to_json(indent=2) / to_yaml() -> str | dict
GraphSchema.from_dict / from_json / from_file -> GraphSchema
graph.save(path: str)     # .json or .yaml by extension

NodeType (StrEnum):  AGENT, ROUTER
EdgeType (StrEnum):  TOOL, CONTEXT, ROUTER

Auto-populated: every Agent(session=s) registers into s.graph.
Every as_tool wrapping records an edge with label="as_tool".

**rules**

- Nodes are ``AgentNode`` (provider, model, system) or ``RouterNode``
  (routes, default). ``add_agent`` reads ``agent.id`` / ``name`` /
  ``engine.provider`` / ``engine.model`` (duck-typed).
- ``session.register_tool_edge(outer, inner, label=…)`` adds an
  ``EdgeType.TOOL`` edge manually if you're wiring outside of
  ``as_tool`` (rare).
- Serialisation is descriptor-only: reconstructing a runnable pipeline
  from a saved graph is the caller's job.

**example**

```python
from lazybridge import Agent, Session

sess = Session()
researcher = Agent("claude-opus-4-7", name="researcher", session=sess)
writer     = Agent("claude-opus-4-7", name="writer",     session=sess)

orchestrator = Agent(
    "claude-opus-4-7",
    name="orchestrator",
    tools=[researcher, writer],   # as_tool edges registered automatically
    session=sess,
)

print(sess.graph.to_json(indent=2))
# {
#   "session_id": "...",
#   "nodes": [AgentNode(researcher), AgentNode(writer), AgentNode(orchestrator)],
#   "edges": [
#     Edge(from=orchestrator, to=researcher, label="as_tool", type="tool"),
#     Edge(from=orchestrator, to=writer,     label="as_tool", type="tool"),
#   ]
# }

# Persist + reload.
sess.graph.save("topology.yaml")
from lazybridge import GraphSchema
replay = GraphSchema.from_file("topology.yaml")
assert len(replay.nodes()) == 3
```

**pitfalls**

- An Agent without ``session=`` is not registered anywhere. If you pass
  it as a nested tool to an Agent with a session, the outer Agent
  propagates its session down and registers the nested one for you.
- ``to_yaml`` requires PyYAML (``pip install lazybridge[yaml]``);
  ``to_json`` is stdlib-only.
- ``from_dict`` reconstructs descriptors only — the ``provider`` /
  ``model`` strings on ``AgentNode`` are not live ``LLMEngine``s.

## verify=

**signature**

# Three placements, same judge contract.

# 1. Agent-level (final output gate)
Agent("model", verify=judge_agent, max_verify=3, ...)

# 2. Tool-level (every call through the tool gated — "Option B")
agent.as_tool(name, description, verify=judge_agent, max_verify=3)

# 3. Plan-level (per-step, via agent-as-step with verify=)
Plan(Step(Agent(..., verify=judge_agent), ...))

# Judge contract
# Judge receives the agent's output text (and the original task for
# context) and must respond with a string starting with
# "approved" (case-insensitive) to accept. Anything else is treated
# as a rejection; its text is injected as feedback on the next retry.
# Judges may be Agents or plain callables: `Callable[[str], Any]`.

**rules**

- Retry loop: up to ``max_verify`` attempts. Final attempt is returned
  as-is even if still rejected (no infinite loop).
- Rejection feedback is appended to the task string for the next
  attempt: ``f"{original_task}\n\nFeedback: {judge_verdict}"``.
- Agent-level ``verify=`` gates the Agent's final output, regardless of
  which tool chain the engine chose internally.
- Tool-level ``verify=`` (Option B via ``as_tool``) gates every
  invocation of that specific wrapped agent — useful when one
  sub-agent is the risky one and the rest is fine.
- Plan-level is just a special case of agent-level: wrap the step's
  agent with its own ``verify=``.

**example**

```python
from lazybridge import Agent, Plan, Step

judge = Agent(
    "claude-opus-4-7",   # would typically be a cheaper model
    name="judge",
    system='Respond "approved" or "rejected: <short reason>".',
)

# Agent-level: final output gated.
writer = Agent("claude-opus-4-7", verify=judge, max_verify=2)
writer("write a haiku about bees")

# Tool-level (Option B): every call of synthesizer is gated.
synthesizer = Agent("claude-opus-4-7", name="synthesizer")
orchestrator = Agent(
    "claude-opus-4-7",
    tools=[synthesizer.as_tool("synth", verify=judge, max_verify=2)],
)

# Plan-level: one step gated, rest unchecked.
plan = Plan(
    Step(fetcher, name="fetch"),
    Step(Agent("claude-opus-4-7", verify=judge, name="summarise"),
         name="summarise"),
    Step(publisher, name="publish"),
)
```

**pitfalls**

- A strict judge + small ``max_verify`` silently returns poor output.
  Log the retry feedback during development so you know when you're
  hitting the cap.
- Judges as *callables* returning booleans don't produce feedback;
  retries reuse the same task. Return a string verdict if you want the
  feedback loop.
- Nested verify (Agent-level + tool-level + Plan-level all on the
  same path) is allowed but expensive. Pick one per agent unless
  you're intentionally stacking.
- Keep judges cheap (a smaller/faster model) and specific (one
  criterion per judge). Multi-criteria judges conflate failure modes
  and produce vague feedback.
