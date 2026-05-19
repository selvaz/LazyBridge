# Recipes

Each recipe is a runnable example from the `examples/` directory in
the repository, embedded verbatim and walked through. Run any of
them directly with `python examples/<file>.py` once you have the
relevant provider key in your environment.

The recipes are roughly ordered by the
[progressive complexity ladder](../concepts/progressive-complexity.md):
single agent → tools → composition → planning → production.

## Single agent + tools

- [React agent](react-agent.md) — `examples/langgraph/01_react_agent_weather.py`
- [Researcher (single agent)](researcher-single.md) — `examples/crewai/01_research_crew_single_agent.py`

## Sequential composition

- [Researcher → reporter](researcher-reporter.md) — `examples/crewai/02_research_and_report.py`

## Hierarchical / supervisor

- [Supervisor pattern](supervisor-pattern.md) — `examples/langgraph/02_supervisor_research_math.py`

## Planning patterns

- [Plan tool](plan-tool.md) — `examples/patterns/plan_tool.py`
- [Agent builds a plan](agent-builds-plan.md) — `examples/patterns/agent_builds_plan.py`
- [Blackboard planner](blackboard-planner.md) — `examples/patterns/blackboard_planner.py`
- [Dynamic re-planning](dynamic-replanning.md) — `examples/patterns/dynamic_planner.py`

## Human-in-the-loop

The same `HumanEngine` primitive composes into four structurally
different roles — a leaf clarifier, a pipeline entrypoint, a cyclic
chat loop, and a fully custom UI surface — without any chat-specific
framework code. See `examples/hil_app/` for runnable versions.

- [HIL as a clarifier](hil-clarify.md) — `examples/hil_app/01_clarify.py`
- [HIL as an entrypoint](hil-entrypoint.md) — `examples/hil_app/02_entrypoint.py`
- [HIL as a chat loop](hil-chat-loop.md) — `examples/hil_app/03_chat_loop.py` (web UI with persistent server)
- [HIL with a custom UI](hil-custom-ui.md) — `examples/hil_app/04_custom_ui.py` (file-watched `_UIProtocol`)

## Composition shapes

- [Nested pipelines (horizontal)](../guides/full/nested-pipelines.md)
  — Plan-of-Plans, parallel bands of sub-pipelines, and LLM-decided
  dispatch over sub-pipelines.  Companion to the vertical
  `chain` / `Plan` recipes above.

## Reporting

The HTML/PDF reporting recipes moved to the sibling
[LazyReport](https://github.com/selvaz/LazyReport) repository in 0.7.9.
Install with `pip install lazybridge-reports` and follow the recipes
there — the rest of LazyBridge stays installable without WeasyPrint /
Quarto / Pandoc.

## Observability

- [Live visualization](live-visualization.md) — `examples/viz_demo.py`
- [Visualization mock](visualization-mock.md) — `examples/viz_mock_demo.py`
