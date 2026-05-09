# React agent

A single agent that uses one tool. The agent's `LLMEngine` runs a
ReAct loop natively: read the task, decide whether to call a tool,
observe the result, decide again, return the final answer.

This is the LazyBridge equivalent of LangGraph's `create_react_agent` —
no graph DSL, no `@tool` decorator. Tools are plain functions; the
schema comes from type hints + docstring.

## Source

```python
--8<-- "examples/langgraph/01_react_agent_weather.py"
```

## Walkthrough

- **`LLMEngine("claude-opus-4-7", system=...)`** is the engine. The
  `system` prompt sets the persona; everything else (tool dispatch,
  observation, retry on bad tool call) happens inside the engine's
  loop.
- **`tools=[check_weather]`** — the function is passed by reference;
  LazyBridge auto-wraps it as a `Tool` and infers the JSON schema
  from the type hints and docstring. No `@tool` decorator.
- **`verbose=True`** prints turn-by-turn updates to stdout — the
  equivalent of LangGraph's `graph.stream(stream_mode="updates")`.
- **`agent("what is the weather in sf")`** runs the loop synchronously
  and returns an `Envelope`. `.text()` extracts the final answer.

## Variations

- Add more tools by extending `tools=[...]` — the engine emits
  parallel tool calls automatically when the model asks for several
  in one turn.
- Swap the model for a different provider (`gpt-4o`, `gemini-3-flash-preview`)
  with no other changes; LazyBridge infers the provider from the
  model string.
- For typed structured output, pass `output=PydanticModel` — the
  payload becomes a model instance and the framework re-prompts on
  validation errors.

## See also

- [Agent](../guides/basic/agent.md) — the constructor surface this
  recipe uses.
- [Tool](../guides/basic/tool.md) — how plain functions become
  tools, and when to construct `Tool(...)` explicitly.
- [Researcher → reporter](researcher-reporter.md) — the next rung:
  two agents in sequence.
