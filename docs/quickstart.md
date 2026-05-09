# Quickstart

Five minutes from `pip install` to a running agent that calls a tool.

## 1. Install

```bash
pip install "lazybridge[anthropic]"
```

LazyBridge has no required provider — pick the SDK matching the model you
want to call. Available extras: `anthropic`, `openai`, `google`, `deepseek`,
`litellm` (for 100+ providers via LiteLLM). You can install several at once.

Set the matching API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## 2. Your first agent

```python
from lazybridge import Agent, LLMEngine

agent = Agent(
    engine=LLMEngine("claude-opus-4-7"),
)
result = agent("Explain LazyBridge in one sentence.")
print(result.text())
```

That's a complete, runnable program — no `asyncio.run`, no event loop,
no `@tool` decorators. The canonical shape is `Agent(engine=...)` with
each argument on its own line: it's what every example in `examples/`
uses, and it's what you'll extend the moment you need to configure the
engine (`system=`, `max_turns=`, `thinking=`, …).

Shorter forms exist (`Agent.from_model("claude-opus-4-7")` and the
string-positional shortcut `Agent("claude-opus-4-7")`) but they're
sugar — they save a line at the cost of hiding which engine the agent
actually runs. Stick to the canonical form while you're learning;
reach for sugar only after you can write the canonical version from
memory.

Calling `agent(task)` returns an `Envelope` — LazyBridge's typed
result wrapper that carries the payload, token / cost metadata, error
info, and any nested sub-agent rollup. Call `.text()` to get a string.

If you'd rather drive things asynchronously, every agent also exposes
`await agent.run(task)` and an `async for chunk in agent.stream(task)`
streaming form. The sync call shown above is the canonical entry point.

## 3. Add a tool

A tool is just a normal Python function. LazyBridge inspects the
signature, type hints, and docstring to build the schema the LLM sees —
no second JSON definition required.

```python
from lazybridge import Agent, LLMEngine

def get_weather(city: str) -> str:
    """Return the current weather for ``city``."""
    # Replace with a real API call. This stub keeps the example offline.
    return f"It's 18°C and sunny in {city}."

agent = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    tools=[get_weather],
)
result = agent("What's the weather like in Paris right now?")
print(result.text())
```

The agent will decide on its own to call `get_weather("Paris")`,
observe the result, and produce the final answer. You did not define a
JSON schema, you did not write any orchestration code, and you did not
lock yourself to a specific provider.

To watch the loop turn-by-turn, pass `verbose=True` — it's the
equivalent of LangGraph's `graph.stream(stream_mode="updates")`.

## 4. What to read next

You've now seen the whole core surface: an agent, a model, and a function
exposed as a tool. From here:

- [**Concepts → Mental model**](concepts/mental-model.md) —
  `Agent = Engine + Tools + State` and the composition rules you'll
  lean on.
- [**Concepts → Progressive complexity**](concepts/progressive-complexity.md) —
  the twelve rungs from this quickstart to a checkpointed production
  pipeline.
- **Guides → Basic → Tool** *(coming next)* — the three schema modes
  (`signature` / `llm` / `hybrid`) and when to pick each.
- **Recipes → React agent** *(coming next)* — the same pattern as above,
  end-to-end with a real tool and a verbose run.
- **For LLM assistants** *(coming next)* — let Claude or ChatGPT generate
  more LazyBridge code for you.
