# Researcher (single agent)

A single research agent with one stub web-search tool — the
LazyBridge equivalent of CrewAI's "single-agent crew" quickstart.
Demonstrates that you don't need a multi-agent crew, decorators, or
YAML configs to get a researcher up and running.

## Source

```python
--8<-- "examples/crewai/01_research_crew_single_agent.py"
```

## Walkthrough

- **No `@CrewBase`, no YAML.** The agent is a plain `Agent(engine=
  LLMEngine(...), tools=[...])` constructor call. The role / goal /
  backstory live in the `system` prompt directly.
- **`serper_search`** is a stub function — swap for Serper, Tavily,
  or any web-search API in production. The signature stays the same
  because the agent doesn't care about implementation details.
- **`output=Path(...)`-style file write** is plain Python around the
  `result.text()` call — there's no framework hook for "write the
  output to a file"; that's just code you put after the agent runs.

## Variations

- Add a structured `output=PydanticModel` to validate the report
  shape (title, bullets, sources) instead of free text.
- Wrap with `verify=judge_agent` for a judge-and-retry loop on the
  output (see [verify=](../guides/mid/verify.md)).
- Move to a two-agent pipeline by chaining a reporter agent — see
  [Researcher → reporter](researcher-reporter.md).

## See also

- [Researcher → reporter](researcher-reporter.md) — the next rung:
  add a reporter agent in sequence via `Agent.chain`.
- [Tool](../guides/basic/tool.md) — schema-from-signature semantics
  for the search function.
- [Agent](../guides/basic/agent.md) — `Agent(engine=...)` canonical
  constructor.
