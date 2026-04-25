# Planner with execute_plan

**Use this when** you have specialist sub-agents and you want a generalist
outer agent to decide whether to call one of them, several in sequence, or
compose them into a small DAG — all without you hard-wiring the
orchestration up front.

The pattern lives in [`examples/patterns/plan_tool.py`][plan_tool] and
exposes a single factory:

```python
make_planner(agents: list[Agent]) -> Agent
```

The returned planner has each sub-agent as a direct tool *and* an
`execute_plan` tool. The LLM picks the simplest fit:

- trivial query → answer directly;
- one sub-agent suffices → call that sub-agent as a tool;
- multi-step or coordinated → compose a `Plan` via `execute_plan`.

[plan_tool]: https://github.com/selvaz/LazyBridge/blob/main/examples/patterns/plan_tool.py

## Quickstart

```python
from lazybridge import Agent, LLMEngine
from examples.patterns.plan_tool import make_planner

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

planner = make_planner([research, math, writer])
planner("Research quantum networking and write a one-paragraph brief.")
```

## What `execute_plan` does

The LLM emits a typed `PlanSpec` (an ordered list of `StepSpec`).
`_materialize` builds a real [`Plan`](../guides/plan.md);
`Agent.from_engine(plan)` triggers
[`PlanCompiler`](../guides/plan.md#compile-time-validation) so forward
`from_step` references, unknown step names, and duplicates are rejected
**before any inner LLM call runs**. On rejection the tool returns
`PLAN_REJECTED: <reason>` so the planner LLM can read it and self-correct.

### `StepSpec` fields

| Field | Notes |
|---|---|
| `name` | Unique snake_case identifier. |
| `agent` | Must match one of the sub-agents passed to `make_planner`. |
| `task_kind` | `"literal"` (use `task_text`) / `"from_prev"` (default) / `"from_step"` (use `task_step`) / `"from_parallel"` (alias of `from_step` for readability). |
| `task_text` | Required when `task_kind="literal"`. |
| `task_step` | Required when `task_kind` is `from_step` or `from_parallel`. |
| `context_kind` / `context_step` | Optional; pull a SECOND step's output into context. Lets the join step combine two parallel branches. |
| `parallel` | `true` to run concurrently with adjacent `parallel=true` siblings. |

## What `from_parallel` actually does

`from_parallel("name")` is an alias of `from_step` — it forwards a single
specific branch's envelope, not a list. Two ways to make this useful:

- **Read one branch in the join step.** Set `task_kind="from_parallel"`
  and `task_step="<branch_name>"` to feed that one branch as the join
  step's task.
- **Combine two branches.** Set `task_kind="from_parallel"` for branch A
  and `context_kind="from_parallel"` + `context_step` for branch B. The
  join step sees A as task and B as context.

For three or more parallel branches that all need to flow into one
synthesis step, `execute_plan` is not the right tool — call the
sub-agents directly (one tool call each), or have a single agent do
the lookups internally.

## Worked examples

### Sequential pipeline
```python
execute_plan(
    task="Quantum networking",
    steps=[
        {"name": "r", "agent": "research"},
        {"name": "w", "agent": "writer"},
    ],
)
```

### Parallel band + one branch read
```python
execute_plan(
    task="...",
    steps=[
        {"name": "hc_apple",  "agent": "research", "task_kind": "literal",
         "task_text": "headcount of Apple in 2024",  "parallel": True},
        {"name": "hc_google", "agent": "research", "task_kind": "literal",
         "task_text": "headcount of Google in 2024", "parallel": True},
        {"name": "report",    "agent": "writer",
         "task_kind": "from_parallel", "task_step": "hc_apple"},
    ],
)
```

### Parallel band + combine two branches in the join
```python
execute_plan(
    task="...",
    steps=[
        {"name": "hc_apple",  "agent": "research", "task_kind": "literal",
         "task_text": "headcount of Apple in 2024",  "parallel": True},
        {"name": "hc_google", "agent": "research", "task_kind": "literal",
         "task_text": "headcount of Google in 2024", "parallel": True},
        {"name": "report",    "agent": "writer",
         "task_kind": "from_parallel", "task_step": "hc_apple",
         "context_kind": "from_parallel", "context_step": "hc_google"},
    ],
)
```

## Pitfalls

- **Forward references.** `from_step` / `from_parallel` must point at a
  step defined earlier in the list. Caught at compile time.
- **Duplicate step names.** Rejected at compile time.
- **Three+ branch synthesis.** `execute_plan` doesn't deliver a list of
  parallel branches to a join step. Call sub-agents directly instead.
- **Don't plan for trivia.** A one-line factual question doesn't need
  any tool call at all. The default system prompt says so.

!!! note "API reference"

    ```python
    from examples.patterns.plan_tool import (
        make_planner,            # Agent factory — the single entry point.
        make_execute_plan_tool,  # Lower-level Tool factory if you need it.
        PLANNER_GUIDANCE,        # System-prompt addendum (decision rules + examples).
        StepSpec,                # Pydantic model for plan steps.
        PlanSpec,                # Pydantic model: task + steps.
    )

    make_planner(
        agents: list[Agent],
        *,
        model: str = "claude-opus-4-7",
        system: str | None = None,    # defaults to PLANNER_GUIDANCE
        name: str = "planner",
        verbose: bool = False,
    ) -> Agent
    ```

!!! warning "Rules & invariants"

    - Tool results never raise across the LLM boundary. Bad specs and
      runtime errors come back as `PLAN_REJECTED` / `PLAN_RUNTIME_ERROR`
      strings the LLM can read and retry.
    - Validation runs at `Agent.from_engine(plan)` — *before* any inner
      LLM call burns tokens.
    - `agents` must have unique `.name` values; `make_planner` raises
      otherwise.
