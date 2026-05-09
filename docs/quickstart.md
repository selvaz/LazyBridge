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
import asyncio
from lazybridge import Agent

async def main():
    agent = Agent.from_model("claude-sonnet-4-6")
    result = await agent.run("Explain LazyBridge in one sentence.")
    print(result.text())

asyncio.run(main())
```

`Agent.from_model(...)` infers the provider from the model name (Anthropic,
OpenAI, Google, …). If you want to be explicit, use
`Agent.from_provider("anthropic", "claude-sonnet-4-6")`.

`agent.run(...)` returns an `Envelope` — LazyBridge's typed result wrapper
that carries the payload, token/cost metadata, error info, and any nested
sub-agent rollup. Call `.text()` to get a string.

## 3. Add a tool

A tool is just a normal Python function. LazyBridge inspects the signature,
type hints, and docstring to build the schema the LLM sees — no second
JSON definition required.

```python
import asyncio
from lazybridge import Agent

def get_weather(city: str) -> str:
    """Return the current weather for a city.

    Args:
        city: A city name, e.g. "Paris" or "Tokyo".
    """
    # Replace with a real API call. This stub keeps the example offline.
    return f"It's 18°C and sunny in {city}."

async def main():
    agent = Agent.from_model("claude-sonnet-4-6", tools=[get_weather])
    result = await agent.run("What's the weather like in Paris right now?")
    print(result.text())

asyncio.run(main())
```

The agent will decide on its own to call `get_weather("Paris")`, observe the
result, and produce the final answer. You did not define a JSON schema, you
did not write any orchestration code, and you did not lock yourself to a
specific provider.

## 4. What to read next

You've now seen the whole core surface: an agent, a model, and a function
exposed as a tool. From here:

- **Concepts** *(coming next)* — `Agent = Engine + Tools + State` and the
  composition rules you'll lean on.
- **Guides → Basic → Tool** *(coming next)* — the three schema modes
  (`signature` / `llm` / `hybrid`) and when to pick each.
- **Recipes → React agent** *(coming next)* — the same pattern as above,
  end-to-end with a real tool and a verbose run.
- **For LLM assistants** *(coming next)* — let Claude or ChatGPT generate
  more LazyBridge code for you.
