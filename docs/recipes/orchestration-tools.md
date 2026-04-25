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

## Five principles baked into the planner

These are in `PLANNER_GUIDANCE` and exist because they're what reliably
makes plan-driven agents work:

1. **Think first, then structure.** `execute_plan` requires a `reasoning`
   argument. The LLM must justify the plan shape in prose before
   committing to JSON; empty / boilerplate reasoning is rejected.
2. **Coarse steps, not micro-steps.** Prefer 2-4 step plans. Six+ steps
   usually means the work belongs inside one sub-agent.
3. **Re-plan, don't perfect-plan.** Two simple `execute_plan` calls in
   sequence beat one complex one. Big tasks → short scouting plan, then
   re-plan with what you learned.
4. **Verify the answer addresses the question.** Optional `verify=` on
   `make_planner` wraps the planner with a judge agent (extra LLM call;
   off by default).
5. **Prefer the simpler shape.** Direct sub-agent call > linear plan >
   plan with parallel band > plan with combined branches.

## What `execute_plan` does

The LLM emits a typed `PlanSpec`: a `reasoning` string, the user `task`, and
an ordered list of `StepSpec`. `_materialize` builds a real
[`Plan`](../guides/plan.md); `Agent.from_engine(plan)` triggers
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
    reasoning="Two-step pipeline: research gathers facts, writer turns "
              "them into prose. Linear, no branching. Smallest shape.",
    task="Quantum networking",
    steps=[
        {"name": "gather", "agent": "research"},
        {"name": "draft",  "agent": "writer"},
    ],
)
```

### Parallel band + one branch read
```python
execute_plan(
    reasoning="Two parallel lookups; report uses Apple only per user "
              "instruction. Google is collected but not consumed.",
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
    reasoning="Two parallel lookups feed a comparison writer. Apple as "
              "task, Google as context — Plan only forwards two branches "
              "per step, which is enough for a comparison.",
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

### Big task — short scout, then re-plan
For uncertain or open-ended work, don't try to fully plan ahead. Issue a
short scouting `execute_plan`, read the result, and call again with what
you've learned. Two simple plans beat one speculative big one.

## Optional: verify= for high-stakes outputs

```python
from lazybridge import Agent, LLMEngine
from examples.patterns.plan_tool import make_planner, PLANNER_VERIFY_PROMPT

judge = Agent(
    engine=LLMEngine("claude-opus-4-7", system=PLANNER_VERIFY_PROMPT),
    name="judge",
)
planner = make_planner([research, math, writer], verify=judge, max_verify=3)
```

When `verify=` is set, the planner's final output runs through the judge
(LazyBridge's built-in verify-with-retry loop). The judge replies
`approved` or `rejected: <reason>`; on rejection the planner retries up
to `max_verify` times with the judge's feedback in context. Costs one
extra LLM call per attempt — turn it on when wrong answers are expensive.

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
        PLANNER_GUIDANCE,        # System-prompt addendum (5 principles + examples).
        PLANNER_VERIFY_PROMPT,   # Suggested system prompt for the verify= judge.
        StepSpec,                # Pydantic model for plan steps.
        PlanSpec,                # Pydantic model: reasoning + task + steps.
    )

    make_planner(
        agents: list[Agent],
        *,
        model: str = "claude-opus-4-7",
        system: str | None = None,    # defaults to PLANNER_GUIDANCE
        name: str = "planner",
        verbose: bool = False,
        verify: Agent | None = None,  # optional judge; off by default
        max_verify: int = 3,
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
