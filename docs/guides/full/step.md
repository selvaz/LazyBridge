# Step

The unit a `Plan` is built from. Each `Step` declares a target (an
agent, a callable, or a tool name), the task it runs, where its
input comes from, what types it expects, and whether to persist its
payload. Routing is also declared on the step — see
[Routing](routing.md) for the full surface.

## Signature

```python
from lazybridge import Step, from_prev

Step(
    target,                        # str (tool name) | Callable | Agent
    task=from_prev,                # Sentinel or literal string — the prompt for this step
    context=None,                  # Sentinel | str | list[Sentinel | str] — side context
    sources=(),                    # iterable of objects with .text() (live-view injection)
    writes=None,                   # str — Store key the payload is persisted under
    input=Any,                     # type annotation for the step's input (informational)
    output=str,                    # type for Envelope.payload (Pydantic class enables validation)
    parallel=False,                # mark as a member of a parallel band
    name=None,                     # unique within the Plan; defaults to target's name
    # Routing — see Routing guide for the full surface.
    routes=None,                   # dict[str, Callable[[Envelope], bool]]
    routes_by=None,                # str — name of a Literal field on `output` model
    after_branches=None,           # str — exclusive-branch rejoin point
)
```

`task=from_prev` is the default and means "feed me whatever the
previous step produced". A literal string is used verbatim — useful
for hard-coded prompts at intermediate steps where the data flows
through `context=` instead.

## Synopsis

A `Step` is the smallest declarative unit `Plan` orchestrates. Its
fields fall into three groups:

**Data flow.** `target` is *what runs*; `task=` is *the prompt*;
`context=` is *the side data*; `sources=` is *the live-view objects*
(e.g. a `Store` or `Memory` whose current state should be appended
verbatim). All four flow into the step's `Envelope` before the
target runs.

**Typing and persistence.** `input=` and `output=` declare the
step's expected input and output types; `output=PydanticModel`
enables validation and unlocks `routes_by="field"`. `writes="key"`
persists the step's payload to `store["key"]` after a successful
run — required for checkpoint resume and for downstream agents
reading via `sources=[store]`.

**Concurrency and naming.** `parallel=True` marks the step as a
member of a concurrent band (consecutive `parallel=True` steps are
dispatched together via `asyncio.gather`). `name=` is the
authoritative key the rest of the plan references — duplicates and
typos surface at construction time.

**Routing fields** (`routes`, `routes_by`, `after_branches`) get
their own dedicated guide: [Routing](routing.md).

## When to use specific fields

- **`task=` literal string** — for *specialised* steps where the
  prompt is fixed and the data flows through `context=`. Example:
  `Step("rank", task="Rank by relevance.", context=from_prev)`.
- **`task=` sentinel** — for *delegating* steps where the previous
  step's output **is** the prompt. Default `from_prev` is
  appropriate most of the time.
- **`context=` single sentinel** — pull data from one upstream
  step.
- **`context=[...]` list** — synthesise from multiple upstream
  steps without an intermediate combiner. Items resolve
  independently and join with blank-line separators (same shape
  as `sources`); literal strings can ride along to inject fixed
  boilerplate.
- **`sources=`** — for *live-view* state that should reflect the
  most recent value at step execution time (`Store`, `Memory`,
  any object with `.text()`). Sentinels resolve once at the start
  of the step; sources re-materialise on every read.
- **`output=Model`** — when the next step's `context=` will read
  a typed payload, or when you want compile-time validation of a
  `routes_by="field"` reference.
- **`writes="key"`** — for *crash recovery* (`resume=True`
  reconstructs from store writes) and for *cross-agent reads*
  (a downstream agent with `sources=[store]` sees the live key).
- **`parallel=True`** — when the step has no data dependency on
  its declared neighbours and concurrent execution is safe.

## When NOT to use specific fields

- **Don't set `output=Model` purely for "type docs".** It activates
  Pydantic validation, structured-output retry, and `routes_by`
  semantics. If you want documentation, use `description=` or a
  comment.
- **Don't use `writes=` for in-Plan-only data.** A downstream step
  can read upstream output via `from_step("name")` directly from
  the in-memory history — no `Store` needed unless the value must
  also survive a crash or be visible to other agents.
- **Don't set `name=` to a string that collides with a tool.**
  When the target is a string (`Step("research")`), the framework
  resolves it against the wrapping agent's `tools=[...]` map. The
  step's `name=` and the resolved tool's name are the same key —
  there's no separate "step name" to disambiguate.

## Example

```python
from pydantic import BaseModel

from lazybridge import Agent, LLMEngine, Plan, Step, Store, from_prev, from_step


class Hits(BaseModel):
    items: list[str]


class Ranked(BaseModel):
    top: list[str]


def normalise(text: str) -> str:
    """Strip and lowercase — pure Python, no LLM."""
    return text.strip().lower()


searcher = Agent(
    engine=LLMEngine("deepseek-v4-flash"),
    name="search",
)
ranker = Agent(
    engine=LLMEngine("deepseek-v4-flash"),
    name="rank",
)
writer = Agent(
    engine=LLMEngine("gpt-5.4-mini"),
    name="write",
)


# 1) Mixed step targets — agents, plain callables, tool names by string.
plan = Agent(
    engine=Plan(
        Step(searcher, name="search"),                 # Agent target
        Step(normalise, name="clean", task=from_prev), # plain callable target
        Step("score",   name="score", task=from_prev), # tool-name string target
        Step(writer,    name="write",
             task="Write a 150-word brief.",
             context=from_step("clean")),
    ),
    tools=[score_tool],                                # the "score" tool name resolves here
)


# 2) Multi-source synthesis with context=[...].
class Brief(BaseModel):
    title: str
    body: str


synth = Agent(
    engine=LLMEngine("deepseek-v4-flash"),
    name="synth",
    output=Brief,
)
policy_loader = Agent(
    engine=LLMEngine("deepseek-v4-flash"),
    name="policy",
)
competitor = Agent(
    engine=LLMEngine("deepseek-v4-flash"),
    name="bench",
)


plan = Agent(
    engine=Plan(
        Step(searcher,      name="search",  writes="hits",    output=Hits),
        Step(policy_loader, name="policy",  task="Load the 2026 acceptable-use policy."),
        Step(competitor,    name="bench",   task="Find three relevant prior posts."),
        Step(synth,         name="synth",
             task="Draft a 300-word brief; cite each source explicitly.",
             context=[
                 from_step("search"),
                 from_step("policy"),
                 from_step("bench"),
                 "Style: neutral, third-person, no superlatives.",
             ],
             output=Brief),
    ),
)


# 3) Live-view sources — a writer reads any prior reviewer verdict
#    on every loop iteration.
class Verdict(BaseModel):
    feedback: str
    approved: bool


store = Store()
plan = Agent(
    engine=Plan(
        Step(writer, name="write",
             task="Draft a 200-word answer. If a 'verdict' is in the store, "
                  "rewrite the previous draft addressing the feedback.",
             sources=[store],                           # live read on every run
             writes="draft"),
        Step(reviewer, name="review",
             task="Score the draft.",
             context=from_prev,
             output=Verdict,
             writes="verdict"),
    ),
    store=store,
)
```

## Pitfalls

- **`task=str` is a literal, not a sentinel reference.**
  `task="from_prev"` puts the literal string `"from_prev"` into the
  step's task. Use the imported `from_prev` symbol.
- **`output=Model` is for typing the payload, not for routing.**
  A `next` field on the model is just a regular field. To declare
  routing, set `routes={...}` or `routes_by="field"` on the step
  (see [Routing](routing.md)).
- **`writes=` does not deduplicate.** Two steps with
  `writes="result"` overwrite the same key. Pick distinct keys
  or namespace them.
- **`Step(target=callable, name="...")` doesn't get an LLM.** The
  callable runs once with the step's task as its argument. Useful
  for normalisation, validation, or any deterministic
  transformation between LLM steps.
- **`Step("name")` requires the name to resolve.** If the wrapping
  agent has no `tools=[...]` matching `"name"`, the framework
  raises `PlanCompileError` at construction. Either pass the agent
  in `tools=[...]`, or change the step to `Step(target=agent)`
  directly.
- **`parallel=True` is bundled with consecutive parallel steps.**
  The engine groups every adjacent `parallel=True` step into one
  band; a non-parallel step in between starts a new band. Keep
  parallel siblings contiguous in the declaration.
- **A failed parallel branch wipes the band's writes.** No `writes=`
  from the band are applied on error — `resume=True` re-runs the
  whole band cleanly. Don't write side effects inside parallel
  steps that aren't crash-safe to repeat.

## See also

- [Plan](plan.md) — the engine that interprets a list of `Step`s.
- [Sentinels](sentinels.md) — `from_prev` / `from_start` /
  `from_step` / `from_parallel` / `from_parallel_all` /
  `from_memory` / `from_agent` semantics in full.
- [Routing](routing.md) — `routes={...}` predicates,
  `routes_by="field"` Literal dispatch, `after_branches=`
  rejoin points, and the `when` DSL.
- [Store](../mid/store.md) — `writes=` lands here; `sources=[store]`
  reads it back live.
- *Guides → Full → Parallel plan steps* (Phase 3b) — concurrent
  bands and `from_parallel_all` aggregation.
