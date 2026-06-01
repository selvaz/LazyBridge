# ReplanEngine

The adaptive-orchestration engine — the dynamic counterpart to
[`Plan`](plan.md). Where `Plan` compiles a fixed DAG at construction
time, `ReplanEngine` lets a **planner agent** decide the shape of the
work at runtime: it is called every round, its output drives which
tools run, and a Store-backed checkpoint is written after every round
so a restart resumes from the correct round without re-executing
completed work.

Pass it to an `Agent` like any other engine.

## Signature

```python
from lazybridge import Agent, LLMEngine, ReplanEngine, Store
from lazybridge.engines.replan import PlanRound, Task

ReplanEngine(
    planner_name="planner",   # name of the planner tool in the parent Agent's tool_map
    store=None,               # Store for checkpoint/resume
    checkpoint_key=None,      # str — required to enable persistence
    resume=False,             # continue from the last checkpoint on the next call
    max_rounds=20,            # safety cap on replan rounds; guards against bad termination
)
```

`ReplanEngine` has no constructor injection of the planner or workers —
it follows LazyBridge's **"everything is a tool"** principle. Everything
it dispatches is resolved from the parent `Agent`'s `tool_map` at run
time:

- The **planner** is a `Tool` in `tools=[]`, built with
  `output=PlanRound`, and located by `planner_name`.
- The **workers** (agents, plain functions, pool routes) are also in
  `tools=[]`. Each task is dispatched verbatim via
  `tool.run(**task.kwargs)` — no special-casing for pools or agents.

## The two output types

The planner emits a `PlanRound` each turn; `ReplanEngine` deserialises
it and dispatches its tasks.

```python
class Task(BaseModel):
    tool: str                 # name of a tool in the tool_map
    kwargs: dict[str, Any]    # forwarded verbatim to tool.run(**kwargs)
    parallel: bool = True     # True → run concurrently with adjacent parallel siblings

class PlanRound(BaseModel):
    reasoning: str            # why this set of tasks was chosen
    tasks: list[Task]         # tasks to execute this round
    done: bool = False        # True → stop; final_answer required
    final_answer: str | None  # the user-facing answer (required when done=True)
```

Tasks within the same round flagged `parallel=True` run concurrently via
`asyncio.gather`; `parallel=False` tasks run sequentially after the
parallel group. **Dependent tasks belong in the next round** — after the
planner has seen the outputs from this one.

## When to use it

| Use… | when… |
|---|---|
| [`LLMEngine`](../../reference/engines.md#llm-engine) | a single agent calls tools in a loop and you need no persistence — the built-in tool-calling loop already does ReAct. |
| **`ReplanEngine`** | the *shape* of the work depends on the query and intermediate results — structured replan rounds, explicit parallelism, and checkpoint/resume on the loop. |
| [`Plan`](plan.md) | the step topology is **fixed and known up front** (DAG compiled at construction). |

`ReplanEngine` is "ReAct on tasks": the planning unit is a *batch* of
tasks rather than a single tool call.

## Minimal example — planner + plain functions

You do **not** need a hierarchy of sub-agents. The workers can be plain
Python functions; the only required `Agent` is the planner.

```python
from lazybridge import Agent, LLMEngine, ReplanEngine
from lazybridge.engines.replan import PlanRound

def fetch(url: str) -> str:
    """Download a page."""
    return f"[contents of {url}]"

def word_count(text: str) -> int:
    """Count words."""
    return len(text.split())

planner = Agent(
    engine=LLMEngine("claude-opus-4-8", system="You are a task planner. Emit one PlanRound per round."),
    output=PlanRound,
    name="planner",                      # ← ReplanEngine finds it by this name
)

agent = Agent(
    engine=ReplanEngine(max_rounds=5),
    tools=[planner, fetch, word_count],  # workers are just functions
    name="agent",
)

print(agent("Download example.com and tell me how many words it has").text())
```

!!! note "Why not just `LLMEngine`?"
    For a single agent that reasons and calls tools in a loop, `LLMEngine`
    already does ReAct — you don't need `ReplanEngine`. Reach for
    `ReplanEngine` when you want **structured replan rounds**, **explicit
    parallelism**, or **checkpoint/resume** on the loop.

## Parallel fan-out across workers

The planner can emit several independent tasks in one round; they run
concurrently. Dependent work goes in the next round.

```python
research = Agent(
    engine=LLMEngine("claude-sonnet-4-6", system="You look up facts via web_search. No math."),
    tools=[web_search], name="research", description="Web lookups. Cannot do math.",
)
math = Agent(
    engine=LLMEngine("claude-sonnet-4-6", system="You do arithmetic with add/multiply."),
    tools=[add, multiply], name="math", description="Arithmetic only.",
)
writer = Agent(
    engine=LLMEngine("claude-sonnet-4-6", system="You synthesise prior results into prose."),
    name="writer", description="Final synthesis. Adds no new facts.",
)

guardian = Agent(
    engine=ReplanEngine(max_rounds=10),
    tools=[planner, research, math, writer],
    name="guardian",
)

env = guardian(
    "Combined headcount of Apple and Google in 2024, then write a paragraph "
    "on what those numbers say about their staffing strategies."
)
print(env.text())
```

A round the planner might emit (the `PlanRound` schema):

```python
PlanRound(
    reasoning="The two headcounts are independent → run them in parallel.",
    tasks=[
        Task(tool="research", kwargs={"task": "Apple headcount 2024"},  parallel=True),
        Task(tool="research", kwargs={"task": "Google headcount 2024"}, parallel=True),
    ],
    done=False,
)
# next round: Task(tool="math", ...) to sum them, then Task(tool="writer", ...)
```

The planner's system prompt does **not** hardcode worker names —
`ReplanEngine` injects the available tool schemas and the accumulated
history into every planner call dynamically.

## Checkpoint & resume

For long or expensive pipelines, pass `store=` **and** `checkpoint_key=`
to persist round state after every round. Pass `resume=True` to continue
from the last checkpoint on the next call.

```python
from lazybridge import Agent, ReplanEngine, Store

store = Store(db="project.sqlite")

guardian = Agent(
    engine=ReplanEngine(
        store=store,
        checkpoint_key="report-apple-google",   # unique key per run
        resume=True,                             # continue from the last checkpoint
        max_rounds=20,
    ),
    tools=[planner, research, math, writer],
    name="guardian",
)

guardian("…the long query…")     # first session — checkpoints each round
guardian("continue")             # resumes from the last completed round
```

Semantics match [`Plan`](plan.md):

- The `store` alone does nothing — persistence is keyed on
  `checkpoint_key`. Without it, every run is in-memory.
- The first call **claims** the key via compare-and-swap.
- With **`resume=False`**, a second run against a key already held by
  another run raises `ConcurrentPlanRunError` — fail-fast, single-writer.
  Use a unique `checkpoint_key` for a fresh concurrent run.
- With **`resume=True`**, a second call **adopts** the existing
  checkpoint instead of raising (it stamps its own `run_uid`). This is
  what lets *you* resume your own crashed or paused run — but it is not a
  concurrency guard: do **not** point two `resume=True` workers at the
  same key, or the adopter will preempt the still-running one, which then
  loses its next checkpoint CAS. Give each concurrent run its own
  `checkpoint_key`.
- A completed run (`status="done"`) short-circuits on the next
  `resume=True` call and returns the cached `final_answer` immediately.

## Termination & safety

- **`max_rounds`** is the safety net for bad termination logic. If the
  planner keeps emitting `done=False`, the loop bails after this many
  rounds. Set it defensively.
- **`done=True` requires `final_answer`.** `ReplanEngine` rejects a
  `done` round with a `None` answer *before* writing a permanent `done`
  checkpoint — otherwise every future `resume=True` call would
  short-circuit with an empty payload.
- **Pathological case**: a planner that emits `done=False` with an empty
  task list spins until `max_rounds`. Mitigate by steering the planner
  to set `done=True` with a `final_answer` when no tasks remain.

## See also

- [Plan](plan.md) — the static alternative when the topology is known up
  front.
- [Dynamic re-planning recipe](../../recipes/dynamic-replanning.md) —
  the runnable end-to-end example this guide is drawn from.
- [Parallel](../mid/parallel.md) — application-layer fan-out used inside
  a round.
- [Engines reference](../../reference/engines.md) — the auto-generated
  `ReplanEngine`, `PlanRound`, and `Task` API.
