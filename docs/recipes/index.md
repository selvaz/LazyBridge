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

## Production pipelines

- [Parallel report](parallel-report.md) — `examples/parallel_report_pipeline.py`
- [Daily news](daily-news.md) — `examples/daily_news_report.py`

## Observability

- [Live visualization](live-visualization.md) — `examples/viz_demo.py`
- [Visualization mock](visualization-mock.md) — `examples/viz_mock_demo.py`
