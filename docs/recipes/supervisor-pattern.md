# Supervisor pattern

A supervisor agent dispatches to specialist sub-agents (research,
math) by including them in `tools=[...]`. There's no separate
`create_supervisor` primitive — agents-as-tools is the same
mechanism. The supervisor's LLM picks which specialist to call (and
may call them sequentially or in parallel — that decision is the
model's, not the framework's).

The LazyBridge equivalent of LangGraph's `create_supervisor`
recipe.

## Source

```python
--8<-- "examples/langgraph/02_supervisor_research_math.py"
```

## Walkthrough

- **`tools=[research_agent, math_agent]`** on the supervisor —
  agents are passed directly. Each specialist's `name=` becomes the
  surface tool name the supervisor's LLM sees. No `as_tool()` call
  required.
- **Per-agent `description=`** is what the supervisor's model reads
  when deciding which specialist fits the request. Distinct,
  precise descriptions are the lever — vague ones cause routing
  mistakes.
- **Each specialist has its own `system` prompt** scoped to its
  job ("You are a math expert. Always use one tool at a time.").
  The supervisor's prompt focuses on dispatch, not domain.
- **The model decides parallelism.** When the supervisor's LLM
  emits two tool calls in one turn, the engine dispatches them
  concurrently via `asyncio.gather` — no config knob.

## Variations

- Add a `verify=judge` on a specialist via
  `agent.as_tool(name="research", verify=judge)` to gate every
  research invocation through a judge-and-retry loop.
- Promote the supervisor to a `Plan` if you want explicit ordering
  or a parallel band — `tools=[]` is for LLM-decided dispatch;
  `Plan(Step(...))` is for declared dispatch.
- Cap the supervisor's loop with `LLMEngine(max_turns=N)` if a
  bad model decision risks unbounded re-dispatch.

## See also

- [As tool](../guides/mid/as-tool.md) — implicit `tools=[agent]`
  vs explicit `agent.as_tool(...)`, and when each is right.
- [Researcher → reporter](researcher-reporter.md) — the previous
  rung: fixed-order chain instead of LLM-directed dispatch.
- [Plan](../guides/full/plan.md) — the alternative when you want
  declared ordering instead of LLM choice.
