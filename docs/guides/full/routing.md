# Routing

Conditional flow inside a `Plan`: when one step's output decides
which step runs next. Two forms — `routes={...}` for code-decided
branches, `routes_by="field"` for LLM-decided branches — with the
same call-site visibility and the same compile-time validation.

## Signature

```python
from lazybridge import Step, when

# Form A — predicate map. Your code decides.
Step(
    target,
    name="...",
    routes={
        "target_step_name": predicate,        # Callable[[Envelope], bool]
        "another_target":   another_predicate,
    },
    after_branches=None,                       # str — optional rejoin point
)

# Form B — Literal field. The LLM decides.
Step(
    target,
    name="...",
    output=SomeModel,                          # must declare a Literal[...] field
    routes_by="field_name",                    # name of that Literal field
    after_branches=None,
)
```

`routes` and `routes_by` are mutually exclusive — use one (or
neither) per step. Both place the routing decision **at the call
site** so reviewers can see the branch table without reading any
Pydantic class.

### The `when` DSL

`routes` predicates are callables `(Envelope) -> bool`. Writing them
as raw lambdas works but is dense; the `when` DSL makes the common
shapes declarative.

```python
from lazybridge import when

# Field-level checks (the workhorse)
when.field("items").empty()              # field is None or zero-length container
when.field("items").not_empty()
when.field("severity").equals("urgent")  # ==
when.field("severity").not_equals("spam")
when.field("approved").is_(True)         # `is` — for True / False / None
when.field("kind").in_({"a", "b"})       # membership
when.field("kind").not_in_({"a", "b"})
when.field("score").greater_than(0.5)
when.field("score").less_than(0.5)
when.field("text").matches(r"^urgent")   # re.search

# Strict mode — typos in field names raise instead of silently routing wrong
when.field("items", strict=True).empty()

# Escape hatches
when.payload(callable)                   # callable(payload) -> bool
when.envelope(callable)                  # callable(envelope) -> bool
when.errored()                           # True iff envelope carries an error
```

## Synopsis

Routing in `Plan` is **declared on the `Step`**, never hidden
inside the output model. Two forms:

**Form A — `routes={...}` (predicate map).** Your code decides the
branch. The framework calls each predicate in declared order with
the step's output envelope; the **first one returning `True`**
makes Plan jump to that target step. If none returns `True`,
linear progression continues to the next declared step.

**Form B — `routes_by="field"` (Literal field).** The LLM decides
the branch. You declare a `Literal[...]`-typed field on the step's
`output=Model`; the framework reads `env.payload.<field>` after the
step runs. If the value is a string matching a step name, Plan
jumps there; if `None` or unmatched, linear progression continues.

Both forms are validated at construction. Unknown route targets,
malformed Literal types, predicates that aren't callable — all
caught by `PlanCompileError` before any LLM call.

### Detour vs. exclusive branch

By default, routing is a **detour**: when step A routes to step X,
Plan runs X and then **resumes linear progression from X's declared
position**. There is no implicit "no fall-through" mode.

Set `after_branches="step_name"` alongside `routes` or `routes_by`
to make routing **exclusive**: only the matched branch runs; all
declared steps between the routing step and the rejoin point are
skipped; execution continues at `step_name` after the chosen branch
completes.

| Pattern | When to use |
|---|---|
| **Terminal early-out** | Put the routed-to step *last* in the declared list; after it runs, linear progression has nowhere left to go → Plan ends. |
| **Mid-pipeline detour** | Route to a step in the middle; after it runs, the pipeline continues from there. |
| **Loop / self-correction** | Route back to an earlier step; the loop exits once the predicate stops firing. `max_iterations` is the safety net. |
| **Exclusive branch with rejoin** | Set `after_branches="<rejoin>"` so siblings are skipped; execution always resumes at the named rejoin step. |

## When to use which form

- **`routes={...}`** when *your code* decides — programmatic checks
  on the typed payload (empty list, score below threshold, regex
  match, multi-field combinator). Cheap, deterministic, no extra
  LLM call.
- **`routes_by="field"`** when *the LLM* decides — classification,
  triage, intent detection. The model emits a Literal value as
  part of its structured output; you don't write the dispatch
  logic.

## When NOT to use routing

- **Linear pipelines.** Just stack `Step`s in declared order; no
  routing fields needed.
- **Parallel fan-out where every branch should run.** Use
  `Step(parallel=True)` on each branch; no routing primitives are
  involved.
- **Crash recovery.** That's `resume=True` + `checkpoint_key=`,
  not routing. Routing is *control flow within a single run*;
  resume is *durability across runs*.
- **Loop counters.** Routing back to an earlier step works for
  bounded retries via `max_iterations`, but if you need an
  explicit attempt counter, persist it via `writes=` and read
  it via a sentinel rather than relying on routing alone.

## Example

```python
from typing import Literal

from pydantic import BaseModel

from lazybridge import Agent, LLMEngine, Plan, Step, Store, from_prev, from_start, from_step, when


# 1) Form A (predicate map) — empty-search early-out via the when DSL.
class Hits(BaseModel):
    items: list[str]


searcher = Agent(engine=LLMEngine("claude-opus-4-7"), name="search")
ranker = Agent(engine=LLMEngine("claude-opus-4-7"), name="rank")
writer = Agent(engine=LLMEngine("claude-opus-4-7"), name="write")
apology_agent = Agent(engine=LLMEngine("claude-opus-4-7"), name="apology")


plan = Agent(
    engine=Plan(
        Step("search",
             output=Hits,
             # Branch table: when items is empty, route to "apology"
             # instead of falling through to "rank".
             routes={"apology": when.field("items").empty()}),
        Step("rank",        task="Rank these search hits."),
        Step("write",       task="Write a 200-word brief.", context=from_step("rank")),
        Step("apology",     task="Apologise; suggest broader terms."),  # last → terminal
    ),
    tools=[searcher, ranker, writer, apology_agent],
)


# 2) Form B (routes_by) — LLM-decided triage with exclusive branching.
class Triage(BaseModel):
    summary: str
    severity: Literal["urgent", "normal", "spam"] | None = None


classifier = Agent(engine=LLMEngine("claude-opus-4-7"), name="classify", output=Triage)
escalator  = Agent(engine=LLMEngine("claude-opus-4-7"), name="urgent")
triager    = Agent(engine=LLMEngine("claude-opus-4-7"), name="normal")
closer     = Agent(engine=LLMEngine("claude-opus-4-7"), name="spam")
archiver   = Agent(engine=LLMEngine("claude-opus-4-7"), name="archive")


plan = Agent(
    engine=Plan(
        Step("classify",
             output=Triage,
             routes_by="severity",            # reads env.payload.severity
             after_branches="archive"),       # skip siblings; rejoin at "archive"
        Step("urgent",  task="Page on-call; open P0."),
        Step("normal",  task="Add to support backlog."),
        Step("spam",    task="Close as spam."),
        Step("archive", task="Log to the audit archive."),    # always runs
    ),
    tools=[classifier, escalator, triager, closer, archiver],
)


# 3) Self-correction loop — route back when the reviewer rejects.
class Verdict(BaseModel):
    feedback: str
    approved: bool


reviewer  = Agent(engine=LLMEngine("claude-opus-4-7"), name="review")
publisher = Agent(engine=LLMEngine("claude-opus-4-7"), name="publish")


store = Store()
plan = Agent(
    engine=Plan(
        Step("write",
             task="Draft a 200-word answer; if a 'verdict' is in the store, "
                  "rewrite addressing the feedback.",
             context=from_start,
             sources=[store],
             writes="draft"),
        Step("review",
             task="Score the draft; approved=True only if accuracy + tone + length all pass.",
             context=from_prev,
             output=Verdict,
             writes="verdict",
             # Loop back to the writer when rejected.
             routes={"write": when.field("approved").is_(False)}),
        Step("publish", task="Final-format and publish.", context=from_step("write")),
        # Without max_iterations, an infinite-rejection bug would loop
        # forever. 8 attempts is a defensible upper bound for most
        # policies; tune to your SLA.
        max_iterations=8,
    ),
    tools=[writer, reviewer, publisher],
    store=store,
)


# 4) Lambda escape hatch for one-off predicates.
classify_hits = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    output=Hits,
    name="classify",
)
plan = Agent(
    engine=Plan(
        Step("classify",
             routes={"apology": lambda env: not env.payload.items}),
        Step("rank"),
        Step("apology"),
    ),
    tools=[classify_hits, ranker, apology_agent],
)


# 5) Custom predicate function for multi-field combinators.
class Score(BaseModel):
    score: float
    topic: str


def needs_review(env) -> bool:
    """Route to review when the score is low AND the topic is sensitive."""
    return env.payload.score < 0.5 and env.payload.topic in {"medical", "legal"}


score_classifier = Agent(engine=LLMEngine("claude-opus-4-7"), output=Score, name="classify")
auto_agent       = Agent(engine=LLMEngine("claude-opus-4-7"), name="auto")
review_agent     = Agent(engine=LLMEngine("claude-opus-4-7"), name="review")

plan = Agent(
    engine=Plan(
        Step("classify",
             routes={"review": needs_review}),
        Step("auto"),
        Step("review"),
    ),
    tools=[score_classifier, auto_agent, review_agent],
)
```

## Pitfalls

- **`routes_by` requires `output=` to be a Pydantic model with the
  named field as `Literal[...]` (or `Literal[...] | None`).**
  Anything else fails at construction. The compiler also verifies
  every Literal value matches a declared step name.
- **Routing cycles are not a compile error.** `A → B → A` may be
  intentional (self-correction loops). They surface at runtime as
  `MaxIterationsExceeded` once `Plan(max_iterations=...)` fires.
  Always pair a loop with a counter or termination predicate.
- **Predicate evaluation order matters.** `routes={...}` evaluates
  in declared order; the first `True` wins. If multiple
  predicates can match, put the more specific one first.
- **`routes_by` and `routes` are mutually exclusive.** Setting
  both on the same step fails at construction.
- **`routes_by="field"` returning `None`** (or any value not
  matching a step name) means "don't route" — linear progression
  continues. This is *not* an error; design your Literal type
  accordingly.
- **`when.field(name)` is non-strict by default.** A typo in the
  field name silently returns `None`, which can mask routing
  bugs (e.g. a payload that always routes to the empty branch
  because the predicate sees nothing). Use
  `when.field(name, strict=True)` to make typos raise.
- **`when` chains return predicates, not bools.** `when.field("x").empty()`
  is a callable; `Step` invokes it later. Don't accidentally call
  it eagerly (`when.field("x").empty()(env)` works but is rarely
  what you mean to write).
- **Routing primitives are ignored on parallel branches.** A
  `parallel=True` step's `routes=` / `routes_by=` is silently
  dropped — parallel bands have their own control flow. Set
  routing on the step *after* the band.
- **A predicate that raises is wrapped as `PlanRuntimeError`.**
  The engine catches the underlying exception and re-raises it
  as `PlanRuntimeError` (a `RuntimeError` subclass) with the
  offending step name, target, and underlying error class in
  the message. Distinct from `PlanCompileError` (build-time
  DAG validation) so caught-at-runtime predicate bugs don't
  conflate with caught-at-construction DAG bugs.
- **`after_branches` must come AFTER the routing step in declared
  order.** A typo or a backward reference fails fast at
  construction with a `PlanCompileError` message that names both
  positions. The rejoin point is also validated for existence.

## See also

- [Plan](plan.md) — the engine that interprets routing decisions.
- [Step](step.md) — the surface that carries `routes=`,
  `routes_by=`, and `after_branches=`.
- [Sentinels](sentinels.md) — sentinels are about *data flow*;
  routing is about *control flow*. They don't overlap.
- [verify=](../mid/verify.md) — judge-and-retry around an output;
  complementary to routing for "wrong → try again" semantics.
- *Guides → Full → Parallel plan steps* (Phase 3b) — concurrent
  bands have their own control flow; routing primitives don't
  apply.
