# Orchestration tools — let an outer agent compose work over sub-agents

**Use these tools when** you have a registry of specialist sub-agents
(research, math, writer, …) and you want a *generalist* outer agent to
decide which one(s) to call and how to compose them — sequential
pipeline, parallel fan-out, or a mixed DAG — without you hard-wiring the
orchestration ahead of time.

The pattern lives in [`examples/patterns/plan_tool.py`][plan_tool] and
exposes three reusable [`Tool`](../guides/tool.md) factories plus a
prepared system-prompt addendum. The outer agent's LLM picks the
simplest tool that fits each query.

[plan_tool]: https://github.com/selvaz/LazyBridge/blob/main/examples/patterns/plan_tool.py

## Quickstart

```python
from lazybridge import Agent, LLMEngine
from examples.patterns.plan_tool import (
    make_orchestration_tools,
    ORCHESTRATOR_GUIDANCE,
)

def web_search(q: str) -> str:
    """Look up current facts."""
    return "..."

def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

research = Agent("claude-opus-4-7", tools=[web_search],
                 name="research", description="Web lookups. No math.")
math     = Agent("claude-opus-4-7", tools=[add],
                 name="math",     description="Arithmetic only.")
writer   = Agent("claude-opus-4-7",
                 name="writer",   description="Prose synthesis.")

REGISTRY = {"research": research, "math": math, "writer": writer}

orchestrator = Agent(
    engine=LLMEngine(
        "claude-opus-4-7",
        system="You are a generalist assistant.\n\n" + ORCHESTRATOR_GUIDANCE,
    ),
    tools=make_orchestration_tools(REGISTRY),
)

orchestrator(
    "Get the 2024 headcounts of Apple, Google, and Meta in parallel, "
    "sum them, and write a short comment."
)
```

## What the three tools do

| Tool                                        | Shape                              | When to use                                                  |
|---------------------------------------------|------------------------------------|--------------------------------------------------------------|
| `execute_chain`                             | `a → b → c` (sequential)           | Strict pipeline; each step builds on the previous output.    |
| `execute_parallel`                          | `[a, b, c]` (raw)                  | Independent legs; raw labelled outputs are enough.           |
| `execute_parallel(synthesize_with="agent")` | `[a, b, c] → synth`                | Fan-out + a single coherent answer drawing on all legs.      |
| `execute_plan`                              | Linear DAG with optional branches  | Linear pipelines that pass *one* branch's output forward.    |

> **Important — `from_parallel` reads ONE branch.** Plan's `from_parallel("name")`
> is an alias for `from_step("name")`: it forwards a single envelope, not a
> list. For "fan out N + synthesise all of them", use
> `execute_parallel(synthesize_with=...)`. Don't try to express that pattern
> through `execute_plan`; the join step would only see one branch.

All three return `str`. On a bad spec they return a structured error
prefix the LLM can parse and self-correct from on its next attempt:

- `CHAIN_REJECTED: …`
- `PARALLEL_REJECTED: …`
- `PLAN_REJECTED: …`
- `*_RUNTIME_ERROR: …`

`execute_plan` invokes `Agent.from_engine(plan)`, which triggers
[`PlanCompiler`](../guides/plan.md) — forward `from_step` references,
unknown step names, and duplicates are caught **before any inner LLM
call runs**.

## ORCHESTRATOR_GUIDANCE

Composing plans is a non-trivial task and the LLM benefits from
concrete patterns. The module exports `ORCHESTRATOR_GUIDANCE` — a
~11k-character system-prompt addendum covering:

- **Decision rules** — pick the simplest tool; skip orchestration for trivia.
- **Tool reference** — every field of `StepSpec`, every `task_kind`.
- **Eleven worked examples** — from "answer directly" up to nested
  parallel-of-pipelines DAGs the orchestrator builds itself.
- **Pitfalls** — forward refs, duplicate names, single-agent batching,
  trivia avoidance, error-recovery loop.

Drop it into your outer agent's system prompt verbatim. The decision
table at the top is what most queries hit; the examples disambiguate
the rest.

## Registry conventions

- Every entry needs `name=` and `description=`. The default tool
  description templates inject the descriptions so the LLM can read
  what each agent does without you re-stating it in the system prompt.
- Entries can be **leaf agents** (single LLM call) **or composed**
  ([`Agent.chain(...)`](../guides/chain.md), or an Agent backed by a
  custom `Plan` via `Agent.from_engine`). To the outer LLM they are
  indistinguishable — a registry entry is "one named capability".
- `Agent.parallel(...)` returns a `_ParallelAgent` whose `.run()` yields
  `list[Envelope]`. That's incompatible with `Step.target` (Plan steps
  expect one envelope per call), so wrap a parallel inside a leaf
  Agent that joins the outputs into a single text envelope before
  registering it. The demo at the bottom of `plan_tool.py` shows the
  pattern.

## Two layers of nesting

**Layer 1 — pre-built registry entries.** Whoever owns the registry can
put `Agent.chain(...)` or an Agent-from-Plan in there. The outer LLM
calls them by name like any leaf.

**Layer 2 — runtime composition.** The outer LLM composes work itself by
issuing one or more orchestration tool calls. Useful patterns:

- *Fan out + synthesise* — one call: `execute_parallel(jobs=[…], synthesize_with="writer")`.
- *Linear pipeline* — one call: `execute_chain(agents=[…], task=…)`.
- *Lead-in + parallel-with-synth + tail* — three calls in sequence
  (chain, parallel-with-synth, chain). Each shape is correct on its own;
  `execute_plan` cannot express the middle stage as a single step because
  `from_parallel` only reads one branch.
- *Linear pipelines that read one specific branch from a parallel band* —
  one `execute_plan` call.

There is no "DAG join that delivers a list of parallel branches to one
follow-up step". Compose via multiple tool calls instead.

## Error-recovery loop

The orchestrator's system prompt should make the LLM treat
`*_REJECTED` as a self-correctable signal:

```
The tool result starting with PLAN_REJECTED / CHAIN_REJECTED /
PARALLEL_REJECTED: read the message, fix the spec, re-emit the tool
call. Don't apologise to the user; fix and retry.
```

Common rejection messages and their fixes:

| Message contains | Cause | Fix |
|---|---|---|
| `references a step that is not earlier in the plan` | Forward `from_step`/`from_parallel` ref | Reorder steps so the referenced step appears first |
| `duplicate step name` | Two steps share `name` | Rename collisions |
| `unknown agent name(s)` | Used a name not in the registry | Use one of the listed agents |
| `task_kind='literal' requires task_text` | Missing `task_text` for a literal step | Provide a `task_text` string |

## Pitfalls

- **Don't plan for trivia.** A one-line factual question doesn't need
  any tool. The outer system prompt should say so explicitly.
- **Don't fan out single-agent work.** If three "parallel" jobs would
  all hit the research agent with similar tasks the agent could batch,
  emit one batched task instead.
- **`from_parallel` is single-branch.** It's just `from_step` under a
  parallel-flavoured name. It forwards exactly one branch's output. For
  fan-out + multi-branch synthesis, use
  `execute_parallel(synthesize_with=...)`.
- **`task_kind="from_prev"` on the first step** receives the original
  user task verbatim. That's usually what you want; if not, use
  `task_kind="literal"` with a hand-crafted `task_text`.

!!! note "API reference"

    ```python
    from examples.patterns.plan_tool import (
        make_execute_chain_tool,    # → Tool: execute_chain(agents, task)
        make_execute_parallel_tool, # → Tool: execute_parallel(jobs)
        make_execute_plan_tool,     # → Tool: execute_plan(steps, task)
        make_orchestration_tools,   # → list[Tool] of all three
        ORCHESTRATOR_GUIDANCE,      # str — drop into the outer system prompt
        StepSpec,                   # Pydantic model for plan steps
        ParallelJob,                # Pydantic model for parallel jobs
        PlanSpec,                   # Pydantic model: task + steps
    )
    ```

    Each `make_*_tool(registry, *, name=..., description=...)`:

    - `registry: dict[str, Agent]` — the named sub-agents the LLM may dispatch to.
    - `name`: tool name visible to the LLM (defaults: `execute_chain`,
      `execute_parallel`, `execute_plan`).
    - `description`: override the LLM-facing description. The default
      enumerates the registry's agents and their `description=` strings.

!!! warning "Rules & invariants"

    - Tool results never raise across the LLM boundary. Bad specs and
      runtime errors are returned as `*_REJECTED` / `*_RUNTIME_ERROR`
      strings the LLM can read and retry.
    - `execute_plan`'s validation runs at `Agent.from_engine(plan)` —
      the moment the materialised plan is wrapped — *before* any inner
      LLM call. Compile errors never burn provider tokens.
    - The registry is read-only at tool-construction time. Mutating
      `registry` after `make_*_tool(registry)` returns has undefined
      effects; rebuild the tool to add/remove agents.
