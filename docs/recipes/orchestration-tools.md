# Planner over sub-agents

**Use these patterns when** you have specialist sub-agents and you want a
generalist outer agent to decide whether to call one of them, several in
sequence, or compose them — without you hard-wiring the orchestration up
front.

LazyBridge ships **two planner factories**, both in
[`examples/patterns/`][examples-dir], that take the same input
(`agents: list[Agent]`) and return a configured `Agent`. Pick by trade-off:

| Factory | Style | Pros | Cons |
|---|---|---|---|
| `make_planner` | Plan **builder** (DAG) | Compile-time validation; native parallel; precise | More tool calls; LLM has to learn the DAG shape |
| `make_blackboard_planner` | **Blackboard** (todo list) | Trivial to prompt; flexible re-planning | No native parallel; no structural validation |

Start with `make_planner` for tasks that benefit from parallelism or
validation; use `make_blackboard_planner` for exploratory work where the
shape emerges as the LLM goes.

[examples-dir]: https://github.com/selvaz/LazyBridge/blob/main/examples/patterns/

## Quickstart — `make_planner` (DAG builder)

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

The returned planner has each sub-agent as a direct tool **and** five
**builder tools** that compose a `Plan` one step at a time:

| Tool | What it does |
|---|---|
| `create_plan(reasoning)` | Start an empty plan; returns a `plan_id`. |
| `add_step(plan_id, name, agent, …, parallel)` | Append one step; validated immediately. |
| `inspect_plan(plan_id)` | Show the current shape. |
| `run_plan(plan_id, task)` | Materialise + execute; returns final text. |
| `discard_plan(plan_id)` | Drop without running. |

Each `add_step` validates **locally**: unknown agent, duplicate name,
forward `from_step` reference, missing `task_text` — all caught with a
pointed `REJECTED: <hint>` so the LLM corrects that single step rather
than re-emitting a whole DAG.

### `add_step` field reference

| Field | Notes |
|---|---|
| `name` | Unique snake_case identifier within the plan. |
| `agent` | Must match one of the agents passed to `make_planner`. |
| `task_kind` | `"literal"` (use `task_text`) / `"from_prev"` (default) / `"from_step"` (use `task_step`) / `"from_parallel"` (alias of `from_step` for readability). |
| `task_text` | Required when `task_kind="literal"`. |
| `task_step` | Required when `task_kind` is `from_step` or `from_parallel`. |
| `context_kind` / `context_step` | Optional; pull a SECOND step's output into context. Lets the join step combine two parallel branches. |
| `parallel` | `true` to run concurrently with adjacent `parallel=true` siblings. |

### What `from_parallel` actually does

`from_parallel("name")` is an alias of `from_step` — it forwards a single
specific branch's envelope, not a list. Two ways to make this useful:

- **Read one branch.** `task_kind="from_parallel"`, `task_step="<branch_name>"`.
- **Combine two branches.** Set `task_kind="from_parallel"` for branch A
  and `context_kind="from_parallel"` + `context_step` for branch B. The
  join step sees A as task and B as context.

For three or more parallel branches that all need to flow into one
synthesis step, neither planner is the right tool — call sub-agents
directly, or have a single sub-agent batch the lookups internally.

### Worked examples (builder)

**Sequential pipeline**
```python
pid = create_plan(reasoning="Two-step pipeline: research gathers facts, writer turns them into prose.")
add_step(pid, name="gather", agent="research")
add_step(pid, name="draft",  agent="writer")
run_plan(pid, task="Quantum networking")
```

**Parallel band + read one branch**
```python
pid = create_plan(reasoning="Two parallel lookups, writer reads only Apple per user instruction.")
add_step(pid, name="hc_apple",  agent="research", task_kind="literal",
         task_text="headcount of Apple in 2024", parallel=True)
add_step(pid, name="hc_google", agent="research", task_kind="literal",
         task_text="headcount of Google in 2024", parallel=True)
add_step(pid, name="report",    agent="writer",   task_kind="from_parallel",
         task_step="hc_apple")
run_plan(pid, task="...")
```

**Parallel band + combine two branches**
```python
pid = create_plan(reasoning="Two parallel lookups feed a comparison writer.")
add_step(pid, name="hc_apple",  agent="research", task_kind="literal",
         task_text="headcount of Apple in 2024", parallel=True)
add_step(pid, name="hc_google", agent="research", task_kind="literal",
         task_text="headcount of Google in 2024", parallel=True)
add_step(pid, name="report",    agent="writer",
         task_kind="from_parallel", task_step="hc_apple",
         context_kind="from_parallel", context_step="hc_google")
run_plan(pid, task="...")
```

**Big task — short scout, then re-plan**

For uncertain or open-ended work, don't try to fully plan ahead. Build a
short scouting plan, run it, then `create_plan` again with what you
learned. Two simple plans beat one speculative big one.

### Optional `verify=` for high-stakes outputs

```python
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

## Alternative — `make_blackboard_planner` (todo list)

Less precise but simpler to prompt. Instead of a typed DAG, the LLM
manages a flat to-do list via three blackboard tools:

| Tool | What it does |
|---|---|
| `set_plan(reasoning, tasks)` | Initialise / reset the plan with a new list. |
| `get_plan()` | Read current state with checkmarks + recorded results. |
| `mark_done(task_index, result_summary)` | Tick a task; record a 1-3 sentence summary. |

```python
from examples.patterns.blackboard_planner import make_blackboard_planner

planner = make_blackboard_planner([research, math, writer])
planner("Research recent agent frameworks and write a one-paragraph summary.")
```

### Workflow (what the LLM does turn-by-turn)

1. `set_plan(reasoning="…", tasks=[…])` — flat list, 3-6 coarse items.
2. Loop: pick the next `[ ]` task → call the right sub-agent → `mark_done`.
3. Once everything is `[x]`, synthesise the final answer for the user.

`set_plan` and `mark_done` both return the formatted state with a
`next: N` footer, so the LLM rarely needs to call `get_plan` explicitly.

### Worked examples

**Simple two-step task.** User: *"Research recent agent frameworks and
write a one-paragraph summary."*

```text
[turn 1] set_plan(
    reasoning="Two-step pipeline: research finds frameworks, writer turns
               them into prose.",
    tasks=[
        "Research the most discussed AI agent frameworks of 2026.",
        "Write a one-paragraph summary based on the research findings.",
    ],
)
→ reasoning: Two-step pipeline...
    0. [ ] Research the most discussed AI agent frameworks of 2026.
    1. [ ] Write a one-paragraph summary based on the research findings.
  next: 0

[turn 2] research("Research the most discussed AI agent frameworks of 2026.")
→ "Top 5 in 2026: LazyBridge, LangGraph, CrewAI, AutoGen, Smol-Agents..."

[turn 3] mark_done(
    task_index=0,
    result_summary="Top 5 in 2026: LazyBridge, LangGraph, CrewAI, AutoGen, Smol-Agents.",
)
→ ...
    0. [x] Research the most discussed AI agent frameworks of 2026.
       → Top 5 in 2026: LazyBridge, LangGraph, ...
    1. [ ] Write a one-paragraph summary based on the research findings.
  next: 1

[turn 4] writer("Write a one-paragraph summary based on the research findings.")
→ "Five frameworks dominated agent tooling in 2026..."

[turn 5] mark_done(task_index=1, result_summary="Wrote 4-sentence summary.")
→ ... all tasks done — reply to user

[turn 6] reply to user with the writer's paragraph.
```

**Mid-flow revision.** The user asks a follow-up while the planner is
mid-task; the planner abandons the rest of the current list and resets:

```text
[turn 1] set_plan(reasoning="...", tasks=["A", "B", "C", "D"])
[turn 2] research("A") → ...
[turn 3] mark_done(0, "...")
# new info changes the plan
[turn 4] set_plan(
    reasoning="A's result made B/C/D irrelevant; replacing with E and F.",
    tasks=["E", "F"],
)
→ reasoning: A's result made B/C/D irrelevant...
    0. [ ] E
    1. [ ] F
  next: 0
```
Calling `set_plan` again drops all prior `[x]` state — by design.

**Sub-agent fails / returns error.** The planner inspects the result, then
either retries or revises the plan:

```text
[turn N] research("look up X") → "stub returned no results"
[turn N+1] mark_done(2, "research returned no results — retry with broader query")
[turn N+2] research("look up X with broader scope") → "..."
[turn N+3] (note: same task index is already [x]; planner adds a new task)
           set_plan(
               reasoning="Adding a retry slot since research needed a broader query.",
               tasks=[...prior 3 tasks..., "Re-do research with broader scope"],
           )
```

The blackboard has no structural validation — the LLM is responsible for
picking the right sub-agent per task and for keeping `mark_done` summaries
faithful. The trade-off is fewer guardrails in exchange for more freedom
to reshape the plan as understanding evolves.

## Pitfalls (both planners)

- **Don't plan for trivia.** A one-line factual question doesn't need
  any tool call. The default system prompts say so.
- **`from_parallel` reads one branch.** Three+ branch synthesis: not the
  builder's job. Call sub-agents directly.
- **Build steps in dependency order.** `add_step` rejects forward
  `from_step` references — just add the dependency first.
- **`run_plan` consumes the plan.** Build a new one if you need to run
  again. (`discard_plan` for explicit cleanup of abandoned plans.)
- **`mark_done` summaries are brief.** 1-3 sentences — not the full
  sub-agent output. The summary is your future-self's hint.

!!! note "API reference"

    ```python
    # plan_tool.py
    from examples.patterns.plan_tool import (
        make_planner,                # Agent factory (DAG builder).
        make_plan_builder_tools,     # Lower-level: returns the 5 builder Tools.
        PLANNER_GUIDANCE,            # System-prompt addendum (5 principles + workflow + examples).
        PLANNER_VERIFY_PROMPT,       # Suggested judge prompt for verify=.
        StepSpec, PlanSpec,          # Pydantic models used internally.
    )

    make_planner(
        agents: list[Agent],
        *,
        model: str = "claude-opus-4-7",
        system: str | None = None,    # defaults to PLANNER_GUIDANCE
        name: str = "planner",
        verbose: bool = False,
        verify: Agent | None = None,
        max_verify: int = 3,
    ) -> Agent

    # blackboard_planner.py
    from examples.patterns.blackboard_planner import (
        make_blackboard_planner,     # Agent factory (todo list).
        BLACKBOARD_PLANNER_GUIDANCE, # System-prompt addendum.
    )

    make_blackboard_planner(
        agents: list[Agent],
        *,
        model: str = "claude-opus-4-7",
        system: str | None = None,    # defaults to BLACKBOARD_PLANNER_GUIDANCE
        name: str = "blackboard_planner",
        verbose: bool = False,
        verify: Agent | None = None,
        max_verify: int = 3,
    ) -> Agent
    ```

!!! warning "Rules & invariants"

    - Tool results never raise across the LLM boundary. Bad inputs and
      runtime errors come back as `REJECTED: …` / `PLAN_RUNTIME_ERROR: …`
      strings the LLM can read and retry.
    - Builder validation runs at `add_step` time (local) and again at
      `run_plan → Agent.from_engine` time (PlanCompiler defence-in-depth)
      — *before* any inner LLM call burns tokens.
    - `agents` must have unique `.name` values; both factories raise
      otherwise.
    - In-progress plans are capped at 50 per builder factory; the oldest
      is evicted on overflow so a misbehaving planner can't leak memory.
