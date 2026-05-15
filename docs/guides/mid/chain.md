# Chain

The simplest multi-agent shape: run agents one after another, each
agent's output becoming the next agent's task. A chain is a `Plan` of
sequential `Step`s — `Agent.chain(...)` is the one-line sugar over the
canonical form.

## Signature

```python
# Canonical — explicit Plan with one Step per agent
from lazybridge import Agent, LLMEngine, Plan, Step

pipeline = Agent(
    engine=Plan(
        Step(target=researcher, name=researcher.name),
        Step(target=editor,     name=editor.name),
        Step(target=writer,     name=writer.name),
    ),
    name="pipeline",
)


# Sugar — Agent.chain wraps the same Plan for you
pipeline = Agent.chain(researcher, editor, writer, name="pipeline")
```

The two forms produce structurally identical agents. Sugar saves the
`Plan(Step(...))` boilerplate when there is no router, no parallel
band, and no checkpoint. See
[Canonical vs sugar](../../concepts/canonical-vs-sugar.md) for the
full comparison.

## Synopsis

A chain runs `N` agents sequentially. The first agent receives the
input task; every subsequent agent receives the previous agent's
`Envelope.text()` as its task (the default `from_prev` sentinel). The
result is the last agent's `Envelope`.

Internally, both `Agent.chain` and the canonical form build the same
`Plan` of `Step(target=..., name=...)` entries. `Plan` dispatches
`Agent` targets via `target.run()` directly — no `tools=[...]` is
needed and no extra `Tool` wrapping happens.

State on the outer agent (memory, session, guards, timeout, fallback)
applies at the chain boundary. Each inner agent keeps its own.

## When to use it

- **Linear multi-agent pipelines** with text hand-offs:
  research → edit → write, extract → classify → summarise, plan →
  draft → revise.
- **Quick CrewAI-style sequential crews** where ordering is fixed
  and each agent's output is consumed verbatim by the next.
- **Adding session / memory / guards once at the top** of a small
  pipeline — they apply at the boundary; you don't have to plumb
  them into every step.

## When NOT to use it

- **Typed hand-offs.** `Agent.chain` carries the previous step's
  `text()` (a string) into the next step's task — Pydantic models
  flatten to JSON. If step *N* must consume a typed payload from
  step *N-1*, use a `Plan` with `Step(output=Model)` and a
  sentinel like `context=from_step("name").payload`.
- **Conditional routing.** Chains have no branches. If a step's
  output decides which agent runs next, use `Plan` with `Step(routes=...)`
  predicates or `Step(routes_by="field")`.
- **Crash resume.** Chains have no `checkpoint_key=`. For
  resumable pipelines use `Plan` directly with `store=` +
  `checkpoint_key=`.
- **Fan-out on the same task.** That's `Agent.parallel(...)` — see
  [Parallel](parallel.md).
- **LLM-directed dispatch.** When you want the *model* to choose
  which sub-agent to call, put the agents in `tools=[...]` of an
  outer `Agent`, not in a chain.

## Example

```python
from lazybridge import Agent, LLMEngine, Memory, Plan, Step


def search(query: str) -> str:
    """Search the web for ``query`` and return the top three hits."""
    return "..."


researcher = Agent(
    engine=LLMEngine("gpt-5.4-mini"),
    tools=[search],
    name="researcher",
)
editor = Agent(
    engine=LLMEngine("gpt-5.4-mini"),
    name="editor",
)
writer = Agent(
    engine=LLMEngine("gpt-5.4-mini"),
    name="writer",
)


# 1) Canonical — explicit Plan; what Agent.chain produces internally.
pipeline = Agent(
    engine=Plan(
        Step(target=researcher, name=researcher.name),
        Step(target=editor,     name=editor.name),
        Step(target=writer,     name=writer.name),
    ),
    name="pipeline",
    memory=Memory(strategy="auto"),
)
result = pipeline("AI trends April 2026")
print(result.text())


# 2) Sugar — same agent, two characters less.
pipeline_sugar = Agent.chain(
    researcher,
    editor,
    writer,
    name="pipeline",
    memory=Memory(strategy="auto"),
)
result = pipeline_sugar("AI trends April 2026")
print(result.text())
```

## Pitfalls

- **Typed outputs flatten to text.** A chain step that produces a
  Pydantic model passes its JSON serialisation to the next step's
  task — the downstream agent sees a string, not a typed payload.
  Use `Plan` with explicit sentinels (`context=from_step("name")`)
  to preserve types.
- **The outer agent's name is `"chain"` by default.** Set
  `name="…"` if you want it to appear distinctly in
  `Session.graph` or in cost rollup tables.
- **Memory on the chain wraps the whole pipeline**, not individual
  steps. If you want each inner agent to keep its own conversation
  history, give each its own `memory=Memory(...)` and pass nothing
  on the chain.
- **`Agent.chain(...).chain(...)` does not exist.** `Agent.chain`
  returns a regular `Agent` whose engine is a `Plan`; you can wrap
  it again only by passing it as a target to another `Plan`. The
  same agent shape composes recursively.
- **Errors propagate.** A failed step short-circuits the chain and
  the chain's envelope carries the error. Use a `fallback=` agent
  on the chain (or on individual steps via `verify=` semantics) if
  you want graceful degradation.

## See also

- [Parallel](parallel.md) — fan-out instead of sequence.
- [As tool](as-tool.md) — wrapping an agent (or a chain) as a tool
  for an outer agent that decides when to invoke it.
- [Nested pipelines](../full/nested-pipelines.md) — when a chain
  isn't enough: Plan-of-Plans, parallel bands of sub-pipelines,
  and LLM-decided dispatch over sub-pipelines (horizontal
  composition beyond the linear case).
- [Canonical vs sugar](../../concepts/canonical-vs-sugar.md) — the
  full table mapping `Agent.chain` and friends to their canonical
  `Plan(...)` equivalents.
- *Guides → Full → Plan* (Phase 3) — typed hand-offs, routing,
  checkpoints, and the `Step` surface in full.
