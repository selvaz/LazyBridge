# LazyBridge — Quick Start

Get from zero to a running pipeline in 5 minutes.

---

## Install

```bash
pip install lazybridge
```

---

## Set your API key

LazyBridge reads standard environment variables:

```bash
# Pick the provider you want to use:
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
export DEEPSEEK_API_KEY="..."
```

---

## Example 1 — Single call (one line)

```python
from lazybridge import LazyAgent

ai = LazyAgent("anthropic")

# Simple text response
answer = ai.text("What is the capital of France?")
print(answer)  # "Paris"

# Full response object (tokens, stop reason, etc.)
resp = ai.chat("Explain quantum entanglement in one sentence.")
print(resp.content)
print(f"Tokens used: {resp.usage.input_tokens} in / {resp.usage.output_tokens} out")
```

Switch provider by changing one word:

```python
ai_openai   = LazyAgent("openai")
ai_google   = LazyAgent("google")
ai_deepseek = LazyAgent("deepseek")
```

---

## Example 2 — Tool loop (function calling)

Give the agent a Python function. It will call it automatically when needed.

```python
from lazybridge import LazyAgent, LazyTool

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # In a real app, call a weather API here
    return f"Weather in {city}: 22°C, partly cloudy"

weather_tool = LazyTool.from_function(get_weather)

ai = LazyAgent("anthropic")
result = ai.loop("What's the weather like in Rome and Paris?", tools=[weather_tool])
print(result.content)
```

That's it. `loop()` handles the full tool-call cycle automatically:
- Sends your message to the LLM
- Detects when the LLM wants to call a tool
- Runs your function with the LLM's arguments
- Feeds the result back to the LLM
- Repeats until the LLM is done

---

## Example 3 — Multi-agent pipeline

Two agents, one session, shared state and tracking.

```python
from lazybridge import LazyAgent, LazySession, LazyContext

# Shared container (tracking, store, graph)
sess = LazySession()

# Two agents connected to the same session
researcher = LazyAgent("anthropic", name="researcher", session=sess)
writer     = LazyAgent("openai",    name="writer",     session=sess)

# Step 1: researcher does its work
researcher.loop("Find the top 3 developments in AI this week.")

# Step 2: writer reads researcher's output via LazyContext
writer.chat(
    "Write a short newsletter section from this research.",
    context=LazyContext.from_agent(researcher),
)

# Read the writer's result
print(writer.result)          # plain text output

# Inspect what happened
print(sess.events.get())      # full event log
print(sess.store.read_all())  # shared state
```

---

## Next steps

| Goal | Read |
|------|------|
| All LazyAgent options | [agents.md](agents.md) |
| Multi-agent sessions | [sessions.md](sessions.md) |
| Tools and delegation | [tools.md](tools.md) |
| Context injection | [context.md](context.md) |
| Conditional routing | [routing.md](routing.md) |
| Full pipeline examples | [pipelines.md](pipelines.md) |
| LazyBridge vs raw SDK | [comparison.md](comparison.md) |
