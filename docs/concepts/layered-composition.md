---
title: Layered Composition
---

# Layered Composition

## The claim: one concept

LazyBridge has one composable unit: **Tool**. `Tool.wrap()` accepts a
plain function, an `Agent`, a `Plan`-backed `Agent`, or an MCP server —
all through the same contract. Because `Plan` is an engine (not a special
container), an `Agent(engine=Plan(...))` is itself a valid tool. This
means pipelines compose recursively with no nesting syntax at any depth.

```
function  ──►  Tool
Agent     ──►  Tool
Agent(engine=Plan)  ──►  Tool  ──►  Step.target in an outer Plan
```

There is no "pipeline type". A pipeline is an agent whose engine is a
Plan; that agent wraps as a tool exactly like any other agent.

---

## The composition hierarchy

```
              function
                 │ Tool.wrap()
                 ▼
               Tool
                 │ tools=[...]
                 ▼
              Agent
                 │ agent.as_tool()  or  tools=[agent]
                 ▼
          Agent-as-tool
                 │ Step.target = agent
                 ▼
     Agent(engine=Plan)          ← sub-pipeline
                 │ tools=[sub_pipeline]
                 ▼
     Step(target="sub_pipeline")  in outer Plan
                 │
                 ▼
     Agent(engine=Plan([..., sub_pipeline_step, ...]))
```

Every level is just `Tool`. The nesting is structural, not syntactic.

---

## Canonical 13-line example

```python
from lazybridge import Agent, LLMEngine, Plan, Step, Session, from_step

search    = Agent(engine=LLMEngine("gpt-5.4-mini"),      name="search")
summarise = Agent(engine=LLMEngine("gemini-2.5-pro"),    name="summarise")
writer    = Agent(engine=LLMEngine("claude-sonnet-4-6"), name="write")

research = Agent(
    engine=Plan(Step("search"), Step("summarise")),        # a sub-pipeline
    tools=[search, summarise], name="research",
)
article = Agent(
    engine=Plan(Step("research"),                           # research is one tool
                Step("write", context=from_step("research"))),
    tools=[research, writer], session=Session(),
)
print(article("AI agents in 2026").text())
```

`research` is a Plan-backed agent. The outer Plan treats it as a single
tool named `"research"`. No glue, no special sub-pipeline type.

---

## Three composition dimensions

### Vertical — sequence

Steps in a Plan execute in declaration order. The simplest nested case:

```python
Agent(engine=Plan(Step("a"), Step("b")), tools=[a, b])
```

`Agent.chain(a, b)` is sugar for the same thing when you don't need
named steps or sentinels.

### Parallel — concurrent bands

Mark steps as `parallel=True` to run them in the same band:

```python
Plan(
    Step("fetch_news",    parallel=True),
    Step("fetch_papers",  parallel=True),
    Step("synthesise",    context=from_parallel("fetch_news", "fetch_papers")),
)
```

`Agent.parallel(a, b)` is sugar for a two-step parallel band followed
by a collect step.

### Nested — Plan inside Plan

Give a `Step` a target that is itself a Plan-backed agent:

```python
inner = Agent(engine=Plan(Step("a"), Step("b")), tools=[a, b], name="inner")
outer = Agent(engine=Plan(Step("inner"), Step("c")), tools=[inner, c])
```

The outer Plan sees `inner` as one tool. Nesting can go arbitrarily
deep — there is no nesting limit in the framework.

---

## What nesting gives you for free

| Feature | Notes |
|---|---|
| **Cost rollup** | `Envelope.metadata.nested_cost` aggregates token spend across all levels automatically. |
| **Per-level pre-launch validation** | Each Plan runs `PlanCompiler.validate()` at construction. A broken inner plan fails immediately — before any LLM call in any level. |
| **OTel `nesting_level`** | Every span emitted by an inner plan carries `nesting_level` so you can filter by depth in your observability backend. |
| **`verify=` at any level** | A `verify=judge` on a Step inside an inner Plan works exactly like one on a top-level Step. |
| **Per-level checkpoint** | Each Plan can have its own `checkpoint_key=`; SQLite-backed `Store` namespaces them. A resume restarts from the deepest incomplete step. |

---

## What it does NOT give you

**Sentinels don't cross Plan boundaries.** A `from_step("x")` inside an
inner plan refers to a step in *that* plan, not the outer one. To pass
outer results in, use `Step(context=from_step("outer_step"))` at the
point where you call the sub-pipeline.

**Checkpoint key collisions are silent.** If two Plans at different
levels use the same `checkpoint_key=` string and share a `Store`, the
inner one will overwrite the outer one's checkpoint. Namespace them:
`checkpoint_key="outer.research"`, `checkpoint_key="inner.summarise"`.

**`max_iterations` is per-plan, not transitive.** An outer Plan with
`max_iterations=3` does not limit how many iterations an inner Plan
can run.

---

## When NOT to nest

Nesting adds a Plan compilation step and an extra agent boundary for
every level. For simple linear flows, `Agent.chain` or a flat Plan is
clearer and faster:

```python
# Prefer this for a simple sequence:
result = Agent.chain(search, summarise, writer)("topic")

# Reserve Plan-of-Plans for when the inner pipeline:
#   - runs in parallel with other steps, or
#   - needs its own verify=/checkpoint, or
#   - is reused in multiple outer plans
```

A good signal that you need nesting: you find yourself wanting `verify=`
on a group of steps as a unit, or you want to checkpoint a multi-step
research phase independently of the writing phase.

---

## See also

- [Composition patterns](../guides/full/composition-patterns.md) — the three
  concrete shapes with pitfalls and worked examples
- [Decisions: Composition](../decisions/composition.md) — when to use chain
  vs parallel vs Plan vs nested Plan
- [Everything is a tool](everything-is-a-tool.md) — the single-contract
  model that makes this possible
