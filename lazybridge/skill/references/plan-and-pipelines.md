# Plans and pipelines

Read for any multi-stage job: pipelines, scheduled work, branching,
parallel bands, checkpoint/resume, batch runs, planner-driven loops.
Source of truth: `lazybridge/engines/plan/`, `lazybridge/engines/replan/`,
`lazybridge/sentinels.py`, `lazybridge/predicates.py`; docs
`docs/guides/project-patterns.md` (every claim below was executed there),
`docs/guides/full/{plan,step,sentinels,routing,checkpoint,parallel-plan-steps,replan-engine}.md`.

## Project architecture preflight

When the request is a *project* (several stages, a scheduled job, anything
with persistence), run this checklist before writing code:

1. **Stages, phases, scheduled work or resume** → one root
   `Agent(engine=Plan(...))`. A plain Python function is a legal `Step`
   target. Never hand-roll `for step in steps: step()` over a shared `state`
   dict: that re-implements `Plan` without checkpointing, resume, cost roll-up
   or session events.
2. **Name every `Step`.** The default name is `target.__name__`; two unnamed
   lambdas collide at compile time, and `routes=`, `from_step()` and
   checkpoints all reference the name.
3. **Independent stages over the same input** → a parallel band
   (`Step(..., parallel=True)`) or `Agent.parallel`; never `asyncio.gather` /
   `ThreadPoolExecutor` glue. **N inputs through the same pipeline** →
   `plan.run_many(tasks, concurrency=...)` (keeps input order, isolates a
   failing task in its own slot).
4. **Branching** → `routes=` with a predicate for **every** branch, plus
   `after_branches=` for the rejoin. An approval gate with only an
   `"execute"` predicate lets a rejection fall through into execute
   (verified); a later `halt` step does not fix it.
5. **Scheduled + resumable** → `store=`, `resume=True` and a **run-specific**
   `checkpoint_key` (e.g. `f"job:{run_date}"`). A permanent key with
   `resume=True` makes every run after the first a silent no-op (verified:
   3 runs, 1 execution).
6. **Large data** → move a handle (a `Store` key, a file path), never
   `json.dumps` the payload into a prompt. A step's typed payload collapses to
   `str` for any target that is not an `Agent`. If the plan is resumable the
   depot behind the handle must be durable, not a module-level dict.
7. **State crossing runs** → the shared `Store` with reconstructable keys and
   `agent_id` provenance (`keys()`, `read_entry()`, `compare_and_swap()` are
   enough for a registry). Do not open a new `sqlite3` / `duckdb` file.
8. **Irreversible effects** (send, publish, trade) → last in the plan, with a
   conscious choice: reserve-before-acting via `compare_and_swap`
   (at-most-once; a crash before the effect means it never happens) or a
   receiver-side idempotency key (exactly-once effect only if the receiver
   records the key atomically with the effect). Check-then-act in your own
   process is never safe on its own. Ideally behind a `HumanEngine` gate with
   a safe `default=`.
9. **"Not ready yet"** → `raise PlanPaused(...)`, not a sleep loop or a
   failure. It persists only with `store=` **and** `checkpoint_key=` on the
   Plan; `resume=True` is what the *next* run needs to continue from that step.
10. **Always check the result**: a failed Plan *returns* an error envelope.
    `if not result.ok: raise SystemExit(f"pipeline failed: {result.error}")`.
11. **Emit a composition test** with every new pipeline: build it behind a
    factory that accepts its agents and inject `MockAgent`s (see `testing.md`).
12. **Before finishing, grep the diff** for `for step in`, `state = {}`,
    `json.dumps(` inside a prompt, `try:` around an agent call, new
    `sqlite3.connect(` / `duckdb.connect(`. Replace each with the declarative
    primitive or state why it is domain logic.

## Plan and Step

```python
from lazybridge import Agent, Plan, Step, Store, from_step

pipeline = Agent(
    engine=Plan(
        Step("research"),
        Step("write", context=from_step("research")),
        store=Store(db="runs.sqlite"),
        checkpoint_key=f"report:{run_date}",
        resume=True,
    ),
    tools=[researcher, writer],
    name="pipeline",
)
```

- `Step(target, task=, context=, sources=, writes=, input=, output=,
  parallel=, name=, routes=, routes_by=, after_branches=)`. A string target
  references a tool/agent by `name=` in the Agent's `tools=`; an `Agent` or a
  callable can be the target directly.
- `Plan(*steps, max_iterations=100, store=, checkpoint_key=, resume=False,
  on_concurrent="fail" | "fork", stream_buffer=64)`. Plan kwargs live on
  `Plan`; `tools=`, `session=`, `name=` live on the `Agent`.
- Plans are validated at construction: forward references, duplicate names
  and unknown targets raise `PlanCompileError` before any model call.
- `writes="key"` persists a step's payload in the Store for crash recovery;
  do not use it for data only consumed inside the same plan.
- Agents used directly as a `Step` target do **not** inherit the parent's
  `Session`; pass `session=` explicitly (see `sessions-and-memory.md`).

## Sentinels

| Sentinel | Resolves to |
|---|---|
| `from_prev` | Previous step's payload (default `task`) |
| `from_start` | The plan's initial input |
| `from_step("name")` | That step's output |
| `from_parallel("name")` | One branch's output |
| `from_parallel_all("name")` | All consecutive parallel siblings, labelled text |
| `from_memory("name")` | Live `Memory` of agent `name` (agent needs `memory=`) |
| `from_agent("name")` | Last output of agent `name` from the shared `Store` (needs `store=`) |

`context=` accepts one sentinel/string or a list. Since 1.4.0 a referenced
step contributes its output whatever its type (rendered via `Envelope.text()`).

## Routing

```python
from lazybridge import Step, when

Step(
    "triage",
    routes={
        "legal": when.field("category").equals("legal"),
        "technical": when.field("category").not_equals("legal"),
    },
    after_branches="summarise",
)
```

- Predicates: `when.field(name)` with `.equals/.not_equals/.is_/.in_/
  .not_in_/.greater_than/.less_than/.matches/.empty/.not_empty`;
  `when.payload(fn)`, `when.envelope(fn)`, `when.errored()`.
- `routes_by="field"` lets the model choose via a `Literal` field on the
  step's `output=` model. `routes` and `routes_by` are mutually exclusive.
- There is no implicit "no fall-through": cover every outcome. A backwards
  route builds a loop; `max_iterations` is the safety net.

## Checkpoint, resume, concurrency

- `store=` + `checkpoint_key=` saves progress; `resume=True` continues from
  the last completed step.
- `on_concurrent="fail"` (default): a second run on the same key raises
  `ConcurrentPlanRunError` (compare-and-swap). `"fork"`: each run claims an
  isolated keyspace (good for independent batches sharing a Store).
  `"fork"` + `resume=True` is a `PlanCompileError`.
- `PlanRuntimeError` reports runtime faults with the offending step.

## Parallel work

- Parallel band: consecutive `Step(..., parallel=True)` run concurrently;
  aggregate with `from_parallel_all("first_of_band")`.
- `Agent.parallel(a, b)` for scripted fan-out of whole agents (one joined
  `Envelope`); `tools=[a, b]` when the model should decide.
- `plan.run_many(tasks, concurrency=N, tools=[...])` on the `Plan` object for
  a batch (returns `list[Envelope]` in input order; async twin `arun_many`).

## ReplanEngine (planner decides each round)

```python
from lazybridge import Agent, LLMEngine, PlanRound, ReplanEngine, Store

planner = Agent(
    engine=LLMEngine("claude-opus-5-5", system="You are a task planner."),
    output=PlanRound,
    name="planner",
)
guardian = Agent(
    engine=ReplanEngine(store=Store(db="project.sqlite"), checkpoint_key="proj", resume=True),
    tools=[planner, analyst, coder],
    name="guardian",
)
```

`ReplanEngine(planner_name="planner", store=, checkpoint_key=, resume=,
max_rounds=20)`. Each `ReplanTask` names a `tool` plus `kwargs`; dispatch is
uniform for agents, pools and functions. Use `Plan` for fixed DAGs and
`ReplanEngine` when each round depends on the previous one. For a durable
to-do list worked by several processes use `DurableBlackboard`
(`extensions.md`). Serialising a plan: `docs/guides/advanced/plan-serialize.md`;
`GraphSchema` is the topology descriptor used by observability and the
visualizer (`docs/guides/full/graph-schema.md`).
