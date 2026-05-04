# Examples

Side-by-side ports of canonical examples from other agent frameworks,
re-implemented with the LazyBridge primitives in `lazybridge/`.

Each file shows the **original framework code** in its module docstring,
followed by the LazyBridge equivalent. The point is to make the conceptual
mapping concrete: what's a `@CrewBase` class, a `create_react_agent`, a
`create_supervisor`, in LazyBridge terms.

## Layout

```
examples/
├── langgraph/
│   ├── 01_react_agent_weather.py        # create_react_agent → Agent(tools=[...])
│   └── 02_supervisor_research_math.py   # create_supervisor   → Agent(tools=[agent, agent])
├── crewai/
│   ├── 01_research_crew_single_agent.py # @CrewBase + Flow    → Agent(tools=[...])
│   └── 02_research_and_report.py        # 2-agent sequential  → Agent.chain(a, b)
└── patterns/
    ├── dynamic_planner.py                # (legacy) planner → typed rounds → asyncio.gather
    ├── agent_builds_plan.py              # (legacy) planner emits PlanSpec → materialise into Plan
    ├── plan_tool.py                      # demo of lazybridge.ext.planners.make_planner
    └── blackboard_planner.py             # demo of lazybridge.ext.planners.make_blackboard_planner
```

## Patterns

The two main planner factories now live in-box at
[`lazybridge.ext.planners`](../lazybridge/ext/planners/) — the example files in
`patterns/plan_tool.py` and `patterns/blackboard_planner.py` are thin
runnable demos that import them. Full guide:
[docs/recipes/orchestration-tools.md](../docs/recipes/orchestration-tools.md).

```python
from lazybridge.ext.planners import make_planner, make_blackboard_planner

planner_dag        = make_planner([research, math, writer])         # DAG builder
planner_blackboard = make_blackboard_planner([research, math, writer]) # todo list
```

**DAG builder (`make_planner`)** — sub-agents as direct tools *plus* five
**builder tools** that compose a `Plan` one step at a time:
`create_plan` → `add_step` (×N) → `inspect_plan` (optional) → `run_plan`.
Each `add_step` validates locally — unknown agent, duplicate name, forward
`from_step` ref — so the LLM corrects one step at a time instead of
re-emitting a whole DAG. Native parallel via `parallel=True` and
`task_kind="from_parallel_all"` for N-branch synthesis. Optional `verify=`
for high-stakes outputs.

**Blackboard (`make_blackboard_planner`)** — sub-agents as direct tools
*plus* three blackboard tools for managing a flat todo list:
`set_plan` → loop(`mark_done` after each sub-agent call) → reply. No DAG,
no structural validation; the LLM revises the plan freely by calling
`set_plan` again. Use for exploratory work where the shape emerges as
you go.

**`patterns/dynamic_planner.py`** *(legacy)* — earlier exploration: planner
uses `Agent(output=PlanRound)` to emit a typed task list per round, an
external loop dispatches with `asyncio.gather`. Superseded by the two
factories above.

**`patterns/agent_builds_plan.py`** *(legacy)* — minimal walkthrough of the
"planner emits PlanSpec → materialize into Plan" mechanism that
`plan_tool.py` productionises with the builder API.

## Concept map

| LangGraph                              | CrewAI                              | LazyBridge                                              |
|----------------------------------------|-------------------------------------|---------------------------------------------------------|
| `create_react_agent(model, tools=[])`  | `Agent(role, goal, backstory)`      | `Agent("model", tools=[...])`                           |
| `@tool` decorator                      | `crewai_tools.SerperDevTool()`      | Plain Python function (signature + docstring)           |
| `prompt="..."`                         | `role` / `goal` / `backstory` YAML  | `LLMEngine(model, system="...")`                        |
| `graph.stream(stream_mode="updates")`  | `verbose=True`                      | `Agent(..., verbose=True)`                              |
| `create_supervisor([a, b], model)`     | `Crew(agents=[...], hierarchical)`  | `Agent("model", tools=[agent_a, agent_b])`              |
| Sequential `StateGraph` edges          | `Crew(process=Process.sequential)`  | `Agent.chain(a, b, c)`                                  |
| Parallel branches in graph             | `Process.hierarchical` fan-out      | `Agent.parallel(a, b, c)` or `Step(parallel=True)`      |
| `MemorySaver` / `checkpointer`         | Crew memory + cache                 | `Memory(strategy="auto")` + `Plan(resume=True)`         |
| Custom `interrupt_before` node         | Manual user input task              | `SupervisorEngine` (REPL) or `HumanEngine` (gate)       |
| Channel/state schema                   | Task `context=[other_task]`         | `Store` (k/v blackboard) + sentinels (`from_step`)      |
| `with_structured_output(Model)`        | `output_pydantic=Model`             | `Agent(..., output=Model)` → `env.payload.field`        |
| LangSmith tracing                      | CrewAI telemetry hooks              | `Session(exporters=[OTelExporter(...), JsonFileExporter(...)])` (OTel from `lazybridge.ext.otel`) |

## Why these are shorter

Three things shrink the LazyBridge ports vs. the originals:

1. **Tool-is-tool**: agents and functions share the `tools=[...]` slot, so
   "supervisor" and "crew sequential" stop being separate primitives — they're
   just `tools=[agent_a, agent_b]` and `Agent.chain(a, b)`.
2. **No graph schema**: LazyBridge doesn't ask you to declare nodes/edges/state
   upfront. `Agent.chain` and `Plan` cover the deterministic cases; the LLM's
   tool-calling loop covers the dynamic ones.
3. **No YAML and no `@CrewBase` metaclass**: `role` / `goal` / `backstory` is
   a system prompt. Everything stays in Python where you can refactor it.

## Running

The examples need provider credentials (Anthropic / OpenAI) in the standard
env vars (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`). The `serper_search` and
`web_search` functions are stubs — wire them to your real search API for
non-toy outputs.

```bash
python examples/langgraph/01_react_agent_weather.py
python examples/langgraph/02_supervisor_research_math.py
python examples/crewai/01_research_crew_single_agent.py
python examples/crewai/02_research_and_report.py
```
