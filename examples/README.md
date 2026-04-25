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
└── crewai/
    ├── 01_research_crew_single_agent.py # @CrewBase + Flow    → Agent(tools=[...])
    └── 02_research_and_report.py        # 2-agent sequential  → Agent.chain(a, b)
```

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
| LangSmith tracing                      | CrewAI telemetry hooks              | `Session(exporters=[OTelExporter(...), JsonFileExporter(...)])` |

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
