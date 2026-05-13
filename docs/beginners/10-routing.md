# Step 10: Routing — conditional branching

The four composition primitives so far — sub-agent as a tool, chain,
parallel, Plan — all run **every** step they're declared with. There's no
"if the output looks like X, do Y instead". That's what **routing** adds.

Routing is a feature of `Plan`: you mark some steps with a `routes=` (or
`routes_by=`) parameter, and after that step runs the Plan **conditionally
jumps** to a target step. With it you can express triage, retry-on-bad-
output, empty-result handling, and any pattern where the workflow shape
depends on the data flowing through it.

---

## When chain / parallel / Plan aren't enough

A concrete case: you're processing inbound support tickets. A classifier
agent reads the ticket and decides if it's `urgent`, `normal`, or `spam`.
Each category needs a *different* handler. They all end in the same archive
step:

```
ticket ──► classifier ──┬──► urgent_handler ──┐
                        ├──► normal_handler ──┼──► archive
                        └──► spam_handler  ───┘
```

Three things `chain` can't do:

1. **Pick one branch out of N** based on the classifier's output
2. **Skip the other two** (otherwise you'd be paying for every branch every time)
3. **Rejoin** at a common downstream step

`Plan` with `routes=` (or `routes_by=`) handles all three.

---

## The simplest route — `routes={...}` with a predicate

The general form:

```python
Step("step_name",
     routes={"target_step": predicate},        # branch table
     after_branches="rejoin_step")             # ← WHERE the Plan resumes after the branch
```

Two parameters, one rule each:

- **`routes={...}`** — a mapping `{step_name: predicate}`. After
  `step_name` runs, predicates are evaluated in declared order; the
  first one returning `True` makes the Plan **jump** to that target
  step.
- **`after_branches="..."`** — the step the Plan jumps to once the
  routed-to branch completes. Without this, control falls through
  linearly from the routed-to step's position, which is almost never
  what you want (we'll show why in the next section).

**Always specify where the Plan resumes.** A predicate that doesn't fire
also has to land somewhere — for predicate-based routing this means
covering every case explicitly, so the routing decision is exhaustive.

A minimal example — branch on whether the search found anything, and
always finish on a logging step:

```python
from pydantic import BaseModel
from lazybridge import Agent, LLMEngine, Plan, Step, when


class SearchResult(BaseModel):
    found: bool
    items: list[str] = []


def web_search(query: str) -> SearchResult:
    """Stub — pretend the search returned nothing."""
    return SearchResult(found=False, items=[])


searcher = Agent(
    engine=LLMEngine("claude-haiku-4-5", system="..."),
    tools=[web_search],
    output=SearchResult,
    name="searcher",
)
reporter = Agent(engine=LLMEngine("claude-haiku-4-5",
                                  system="Summarise the search hits."),
                 name="reporter")
apology  = Agent(engine=LLMEngine("claude-haiku-4-5",
                                  system="Politely say nothing was found."),
                 name="apology")
log_outcome = Agent(engine=LLMEngine("claude-haiku-4-5",
                                     system="Emit one line of metrics."),
                    name="log_outcome")


pipeline = Agent(
    engine=Plan(
        Step("searcher",
             routes={
                 "reporter": when.field("found").is_(True),       # success branch
                 "apology":  when.field("found").is_(False),      # failure branch
             },
             after_branches="log_outcome"),     # every branch lands here
        Step("reporter"),
        Step("apology"),
        Step("log_outcome"),                    # rejoin terminal — always runs
    ),
    tools=[searcher, reporter, apology, log_outcome],
    name="pipeline",
)

print(pipeline("rare-historical-event-from-2026").text())
```

Read aloud — for each outcome, *what runs and in what order*:

| `searcher` result | Path |
|---|---|
| `found=True`  | `searcher` → `reporter` → `log_outcome` → end |
| `found=False` | `searcher` → `apology`  → `log_outcome` → end |

Both paths run exactly **two steps after `searcher`**, in the same shape:
the routed-to branch, then the rejoin terminal. No surprise extras.

!!! warning "Don't write 'half' routing"
    The temptation is to write just `routes={"apology": when.field("found").is_(False)}`
    and rely on linear fall-through for the success case. **That's
    broken.** When the predicate doesn't fire, the Plan walks linearly
    through every step between the routing step and the end — including
    your `apology` step, which then *also* runs on success. The next
    section spells out exactly why.

    Rule of thumb: every routing step needs (1) routes that cover all
    cases, AND (2) `after_branches=` pointing at a step *after* every
    branch in the declared order.

---

## The `when` DSL — predicates without lambdas

Raw lambdas are fine for one-offs. For anything reusable, LazyBridge ships a
small predicate DSL via the `when` helper:

```python
from lazybridge import when

# Same predicate, two ways:
routes={"apology": lambda env: not env.payload}                # raw lambda
routes={"apology": when.field("items").empty()}                # when DSL
```

A handful of verbs cover ~95% of real-world cases:

| Builder | Predicate behaviour |
|---|---|
| `when.field("x").empty()` | `env.payload.x` is empty/falsy |
| `when.field("x").equals(value)` | `env.payload.x == value` |
| `when.field("x").in_(["a", "b"])` | `env.payload.x` is in the set |
| `when.field("x").matches(regex)` | regex match on a string field |
| `when.field("x").gt(n)` / `.lt(n)` | numeric comparisons |
| `when.text().contains("...")` | `env.text()` contains a substring |
| `when.error()` | the step produced an error envelope |

For genuinely complex logic, fall back to `when.payload(callable)` or a
plain lambda — the DSL doesn't force itself on you.

---

## The detour trap — and `after_branches`

This is the **single most counter-intuitive thing about routing**. Read it
twice.

!!! danger "Routing is a *detour*, not a *replacement*"
    When a route fires, the Plan jumps to the target step and runs it.
    After the target finishes, the Plan walks forward through whatever
    comes **after the *target*** in the declared step list — *not* after
    the routing step. The branches you "didn't take" that happen to sit
    later in the list will still run.

Concrete example. Naive routing:

```python
# WRONG for triage — extra branches still run depending on which one was picked
Plan(
    Step("classifier", routes_by="severity"),    # picks "urgent"
    Step("urgent"),     # ← runs (routed here)
    Step("normal"),     # ← runs after urgent (linear from urgent's position)
    Step("spam"),       # ← runs after normal
)
```

What happens at runtime when `classifier` picks `"urgent"`:

1. `classifier` runs, `routes_by="severity"` reads `"urgent"` off the
   typed output, Plan returns `"urgent"` as the next step
2. Plan jumps to `urgent`, runs it
3. After `urgent` finishes, Plan asks "what's next?". `urgent` itself
   has no `routes=` / `routes_by=`, and there's no rejoin marker, so
   the Plan **falls through to linear progression from `urgent`'s
   declared position** — and the next declared step after `urgent` is
   `normal`
4. `normal` runs (you paid for it)
5. `spam` runs (linear after `normal`)

You wanted *one* branch; you got *three*. Costs 3× what you expected,
and the final output is `spam`'s, not `urgent`'s.

!!! note "The trap is asymmetric — it depends on which branch fires"
    Same plan, same wiring, different outcomes purely based on the
    classifier's choice:

    | `classifier` picks | What actually runs |
    |---|---|
    | `"urgent"` | `classifier` → `urgent` → `normal` → `spam` (3 branches!) |
    | `"normal"` | `classifier` → `normal` → `spam` (2 branches) |
    | `"spam"`   | `classifier` → `spam` (correct — 1 branch) |

    The plan happens to do the right thing **only when the model picks
    the *last* declared branch**. Anything else over-runs. Don't try to
    paper over this by reordering branches by priority — use
    `after_branches=`.

The fix is one parameter — `after_branches=` — which converts routing
from a detour into an **exclusive branch with a guaranteed rejoin point**.

The triage example, done right:

```python
from typing import Literal
from pydantic import BaseModel
from lazybridge import Agent, LLMEngine, Plan, Step


class Triage(BaseModel):
    severity: Literal["urgent", "normal", "spam"]
    reason: str


classifier = Agent(
    engine=LLMEngine("claude-haiku-4-5",
                     system="Classify the ticket. Output the typed Triage object."),
    output=Triage,
    name="classifier",
)

urgent_handler = Agent(engine=LLMEngine("claude-opus-4-7", system="..."), name="urgent")
normal_handler = Agent(engine=LLMEngine("claude-haiku-4-5", system="..."), name="normal")
spam_handler   = Agent(engine=LLMEngine("claude-haiku-4-5", system="..."), name="spam")
archive        = Agent(engine=LLMEngine("claude-haiku-4-5", system="..."), name="archive")


triage_pipeline = Agent(
    engine=Plan(
        Step("classifier",
             routes_by="severity",                  # ← LLM-decided routing
             after_branches="archive"),             # ← guaranteed rejoin point
        Step("urgent"),
        Step("normal"),
        Step("spam"),
        Step("archive"),                            # always runs (the rejoin)
    ),
    tools=[classifier, urgent_handler, normal_handler, spam_handler, archive],
    name="triage_pipeline",
)
```

What `after_branches="archive"` guarantees:

- Exactly **one** of `urgent` / `normal` / `spam` runs (the one the
  classifier picked via `severity`)
- The other two are **skipped** — you don't pay for them
- After the chosen branch completes, execution **jumps to `archive`**
  unconditionally

This is the canonical triage shape. Memorise it.

---

## `routes_by` — LLM-decided routing, type-safe

`routes_by=` is the more powerful sibling of `routes=`. Instead of writing
predicates yourself, you let the **model** pick the next step by populating
a field on a structured output.

The rules:

1. The step's `output=` must be a Pydantic model
2. One field on that model is declared `Literal["a", "b", "c"]` (or
   `Literal[...] | None`)
3. You pass that field's name as `routes_by="field_name"`
4. The Plan compiler **validates** at construction time that every literal
   value matches a declared step name

What you get for free:

- **Compile-time safety.** Typo in the Literal? `PlanCompileError` at
  construction, before any LLM call. Refactor a step name? The compiler
  catches the dangling literal too.
- **The model's choice is visible.** `env.payload.severity == "urgent"`
  isn't a routing artifact — it's structured data your downstream code
  can inspect.
- **No predicate to maintain.** The "rules" live in the Literal type
  itself; the routing is implicit.

`routes_by=` is the recommended path when the *model* should pick. Use
`routes=` when the routing is a **mechanical** decision (e.g., "is the
result list empty?") — predicates are simpler than forcing the model to
emit a triage label for that.

---

## Loops — when the route goes backwards

A route target can be an **earlier** step. That's how you build loops:
write → critic → (revise back to write if rejected, or continue).

```python
class CriticVerdict(BaseModel):
    decision: Literal["accept", "revise"]
    notes: str


pipeline = Agent(
    engine=Plan(
        Step("writer"),
        Step("critic",
             output=CriticVerdict,
             routes={"writer": when.field("decision").equals("revise")}),
        # If decision == "accept", flows through to publisher
        Step("publisher"),
        max_iterations=5,                     # ← cap the loop; default 25
    ),
    tools=[writer, critic, publisher],
    name="pipeline",
)
```

There's no special "loop" primitive — a loop is *just a route back*.

The `max_iterations=` parameter on `Plan` is the safety net: it caps how
many step executions can run in total before the Plan gives up with a
`MaxIterationsExceeded` error envelope. Default is 25; **always set it
explicitly lower during development** (e.g. `max_iterations=5`) so a
runaway critic loop fails in seconds, not minutes.

`verify=` (Step 6) covers the *most common* "write → critique → revise"
case with less ceremony; reach for routing loops when the critic needs to
be more sophisticated than the `verify=` contract supports (e.g.,
multi-criteria evaluation that affects routing in different ways).

---

## Tracing — see the routing decision

`verbose=True` shows the routing verdict explicitly:

```text
[plan ▶ triage_pipeline  steps=5]
  [step 1/5: classifier  model=claude-haiku-4-5]
    task: <ticket body>
    output: Triage(severity="urgent", reason="server down for paying customer")
    ◆ routes_by("severity") → jump to "urgent"  (after_branches="archive")
  [step 2/5: urgent  model=claude-opus-4-7]
    task: <ticket body>
    assistant: Paging on-call engineer ...
    ◆ branch complete → jump to "archive"   (skipping normal, spam)
  [step 5/5: archive  model=claude-haiku-4-5]
    assistant: Ticket archived as urgent ...
[done] steps_run=3/5  total_cost=$0.0021
```

Two things to notice:

- The trace shows **which** route fired and **why** (the structured field
  value)
- Skipped steps are reported in the summary (`steps_run=3/5`) — useful for
  audit and cost analysis

---

## When to use routes / routes_by / after_branches

| Symptom | Use |
|---|---|
| Retry on bad output (write → critique → maybe rewrite) | Often `verify=` (Step 6) is enough; reach for routes only when verify's contract is too small |
| Empty result → fallback handler | `routes={"fallback": when.field("items").empty()}` |
| Mechanical condition (length, presence, regex) | `routes={...}` with `when` |
| Pick branch from N labels the LLM produces | `routes_by="<field>"` + `output=PydanticModel` |
| Triage shape (one branch out of N, then rejoin) | `routes_by=...` + `after_branches="..."` |
| Loop back to earlier step | A route whose target is an earlier step + `max_iterations` |

If you find yourself writing a long predicate, the Plan is too clever — the
routing decision probably belongs in an agent's structured output (use
`routes_by=`) rather than in glue code.

---

## How other frameworks express conditional flow

??? example "LangGraph (`add_conditional_edges`)"

    LangGraph's equivalent of `routes_by=` is a conditional edges function:

    ```python
    from typing import Literal
    from typing_extensions import TypedDict
    from langgraph.graph import StateGraph, START, END

    class State(TypedDict):
        ticket: str
        severity: Literal["urgent", "normal", "spam"] | None

    def classifier_node(state: State):
        # ... call the model, parse severity ...
        return {"severity": "urgent"}   # set the field

    def route(state: State) -> Literal["urgent", "normal", "spam"]:
        return state["severity"]

    builder = StateGraph(State)
    builder.add_node("classifier", classifier_node)
    builder.add_node("urgent", urgent_node)
    builder.add_node("normal", normal_node)
    builder.add_node("spam", spam_node)
    builder.add_node("archive", archive_node)
    builder.add_edge(START, "classifier")
    builder.add_conditional_edges("classifier", route,
                                  ["urgent", "normal", "spam"])
    builder.add_edge("urgent", "archive")
    builder.add_edge("normal", "archive")
    builder.add_edge("spam",   "archive")
    builder.add_edge("archive", END)
    graph = builder.compile()
    ```

    Same outcome, more explicit graph wiring. Both compile at construction
    time. LangGraph has finer control (multiple conditional edges, complex
    routers); LazyBridge's `routes_by=` plus `after_branches=` covers the
    triage case in two lines.

??? example "CrewAI"

    CrewAI's sequential and hierarchical processes don't express
    conditional branching directly. To get the same shape you'd typically
    write a custom Python wrapper that runs the classifier, inspects the
    result, and selects which sub-crew to dispatch:

    ```python
    classifier_crew = Crew(agents=[classifier], tasks=[triage_task],
                           process=Process.sequential)
    severity = classifier_crew.kickoff(inputs={"ticket": text})

    if severity == "urgent":
        out = urgent_crew.kickoff(inputs={"ticket": text})
    elif severity == "normal":
        out = normal_crew.kickoff(inputs={"ticket": text})
    else:
        out = spam_crew.kickoff(inputs={"ticket": text})

    archive_crew.kickoff(inputs={"summary": out})
    ```

    You've left the framework's abstractions. There's no DAG to inspect,
    no compile-time check, no built-in rejoin. Functional, but not
    declarative.

---

## Summary

| Concept | Syntax | What it does |
|---|---|---|
| Predicate routing | `Step("a", routes={"b": predicate})` | Jump to "b" if predicate(env) is truthy |
| LLM-decided routing | `Step("a", output=Model, routes_by="field")` | Jump to step named by `env.payload.field` |
| Exclusive branch + rejoin | `Step(..., after_branches="rejoin_step")` | Only one branch runs; jumps to rejoin afterwards |
| Declarative predicates | `when.field("x").empty()` etc. | Predicate DSL — no lambdas |
| Detour vs replacement | Default is *detour*; use `after_branches=` for branch+rejoin | Subtle but critical |
| Loop | A route target earlier in the declaration order | Use `Plan(max_iterations=N)` to bound |
| Compile-time validation | `PlanCompileError` on bad targets / Literal values | Catch typos before billing |
| Tracing | `verbose=True` shows the route decision and skipped steps | Audit-friendly |

You now have **every** structural composition primitive LazyBridge offers.
The next step adds the one piece that isn't structural at all: a human in
the loop, as an engine.

---

[**Step 11: Human in the loop with `HumanEngine` →**](11-human-engine.md){ .md-button .md-button--primary }

[← Step 9: Explicit DAGs](09-plan.md){ .md-button }
