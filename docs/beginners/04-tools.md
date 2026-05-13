# Step 4: Giving your agent tools

The agent in Step 3 was already useful — but it could only produce text. Ask it
"What's the weather in Rome right now?" and it will either guess or politely
admit it doesn't know.

A **tool** is the missing piece. With tools, the LLM can call functions in *your*
code: look things up, run computations, hit APIs, write to a database. The LLM
stops being just a generator and becomes an *actor*.

This is the step that turns LazyBridge from "a nice SDK wrapper" into something
the older frameworks (LangChain, CrewAI) built whole products around.

---

## A tool is just a Python function

This is the entire definition:

```python
def get_weather(city: str) -> str:
    """Return the current weather forecast for a city."""
    # In real code you'd call a weather API. For a tutorial, return a stub:
    return f"In {city} the weather is 22°C and sunny."
```

That's it. No decorators required, no JSON schema, no special base class.

Give it to the agent via `tools=[...]`:

```python
from lazybridge import Agent, LLMEngine

agent = Agent(
    engine=LLMEngine(
        "claude-haiku-4-5",
        system="You are a helpful assistant. Answer concisely.",
    ),
    tools=[get_weather],
)

answer = agent("What's the weather in Rome and Tokyo right now?").text()
print(answer)
```

Run it (with `ANTHROPIC_API_KEY` set) and you'll see something like:

```text
In Rome it's 22°C and sunny; in Tokyo it's 22°C and sunny.
```

The LLM **decided** to call `get_weather` twice — once for each city — and
LazyBridge ran the calls, fed the results back, and let the model write the
final answer. You wrote zero loop logic.

---

## How does LazyBridge know what the tool does?

When you pass `tools=[get_weather]`, LazyBridge introspects the function and
generates a JSON schema for the model automatically:

| Source in your code | Becomes |
|---|---|
| Function name (`get_weather`) | Tool name the model calls |
| Docstring first line | Tool description (helps the model decide *when* to call) |
| Parameter type hints (`city: str`) | JSON schema property types |
| Default values / `Optional` | Required vs optional fields |
| Return annotation (`-> str`) | Documentation only — the LLM gets `str(return_value)` |

This is why type hints and a one-line docstring are not optional: they're how the
model decides whether to use your tool and what to pass it. Bad signature, bad
calls.

!!! tip "Write tools like you'd write helper functions"
    `def get_weather(city: str) -> str:` is good.
    `def gw(c) -> str:` is bad — the LLM has no idea what `c` is.

    The same rule applies in raw SDKs, except there you'd write the JSON schema
    by hand. LazyBridge just reads what you already wrote.

---

## Seeing the loop — `verbose=True`

Turn on `verbose=True` and you'll see the exact decisions the model made:

```python
agent = Agent(
    engine=LLMEngine("claude-haiku-4-5", system="You are a helpful assistant."),
    tools=[get_weather],
    verbose=True,
)

agent("What's the weather in Rome and Tokyo right now?")
```

Output (abbreviated):

```text
[agent ▶ engine=LLMEngine model=claude-haiku-4-5 tools=[get_weather]]
  user: What's the weather in Rome and Tokyo right now?
  assistant: ◆ tool_call get_weather(city="Rome")
             ◆ tool_call get_weather(city="Tokyo")
  tool[get_weather]: In Rome the weather is 22°C and sunny.
  tool[get_weather]: In Tokyo the weather is 22°C and sunny.
  assistant: In Rome it's 22°C and sunny; in Tokyo it's 22°C and sunny.
[done] turns=2  tokens=314/87  cost=$0.0002
```

What just happened:

1. **Turn 1** — model received the prompt, decided to call `get_weather` twice
   (in parallel — most modern models can request multiple tools per turn)
2. **LazyBridge ran the calls** locally, captured the return values
3. **Turn 2** — model received the tool results and produced the final answer
4. **Loop terminated** because the model stopped requesting tools

You did not write any of this loop. LazyBridge handles it.

---

## Multiple tools — the model picks

You usually give the agent more than one tool and let it choose:

```python
from lazybridge import Agent, LLMEngine


def get_weather(city: str) -> str:
    """Return the current weather forecast for a city."""
    return f"In {city} the weather is 22°C and sunny."


def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount from one currency to another at today's rate."""
    # Stub — in real code call an exchange-rate API
    return f"{amount} {from_currency} = {amount * 1.08} {to_currency}"


def calculator(expression: str) -> float:
    """Evaluate a simple arithmetic expression and return the result."""
    return eval(expression, {"__builtins__": {}}, {})  # demo only — don't eval user input in prod


agent = Agent(
    engine=LLMEngine("claude-haiku-4-5", system="You are a concise assistant."),
    tools=[get_weather, convert_currency, calculator],
    verbose=True,
)

print(agent("What's 18% of 240, and what's that in EUR if it's USD?").text())
```

The model figures out the tool sequence by itself: `calculator("0.18 * 240")` →
`convert_currency(amount=43.2, from_currency="USD", to_currency="EUR")` → final answer.

You never wrote a routing rule.

---

## Explicit control — `Tool.wrap`

The raw-function form is the fast path. When you need control — a clearer
description, a different name, custom validation — wrap the function explicitly:

```python
from lazybridge import Agent, LLMEngine, Tool


def fetch_weather(city: str) -> str:
    """Internal name we don't want exposed to the model."""
    return f"In {city} the weather is 22°C and sunny."


weather_tool = Tool.wrap(
    fetch_weather,
    name="get_weather",
    description="Look up the current weather forecast for any city worldwide.",
)

agent = Agent(
    engine=LLMEngine("claude-haiku-4-5", system="Helpful assistant."),
    tools=[weather_tool],
)
```

The function still does the work; the model only sees the `name` and
`description` you set. Use this when:

- The Python function name is generic (`fetch`, `get_data`) but the model needs
  a specific cue
- You want to override the docstring's first line for the model without
  rewriting the function
- You're building a library of reusable tools and want clean external names

---

## What "the loop" really is

This is the simplest mental model. When you call `agent(prompt)`:

```text
                        ┌──────────────────────┐
                        │   LLM (provider)     │
                        └──────────┬───────────┘
                                   │
   prompt + history + tool defs ──→│            (you send messages)
                                   │
                                   │←── reply: "call get_weather('Rome')"
                                   │
                ┌──────────────────┴──────────────────┐
                │ LazyBridge: parses reply,            │
                │ runs get_weather('Rome'),            │
                │ formats result as a tool message     │
                └──────────────────┬──────────────────┘
                                   │
                       tool result │
                                   ↓
                        ┌──────────────────────┐
                        │   LLM (next turn)    │
                        └──────────┬───────────┘
                                   │
                                   │←── reply: "It's 22°C in Rome."
                                   │       (no more tool calls)
                                   ↓
                          [loop terminates]
                          envelope returned
```

You can cap the loop with `LLMEngine(..., max_turns=N)` — default is 8 — which
prevents runaway tool-calling if the model gets confused.

---

## What you skip vs raw SDKs

Every raw SDK supports tool calling — but it's *manual*. Here's what the same
two-line "give the agent a tool" effort looks like elsewhere:

| Step | OpenAI / Anthropic / Gemini raw SDK | LazyBridge |
|---|---|---|
| Describe the tool to the model | Write a JSON schema dict by hand | Type hints + docstring |
| Pass it to the call | `tools=[{...schema dict...}]` | `tools=[my_function]` |
| Detect the tool call in the reply | Iterate `response.output` / `.content`, branch on `type == "tool_use"` / `"function"` | Built-in |
| Run the function | Match name, parse JSON args, call, capture result | Built-in |
| Feed the result back | Append a tool result message in the SDK's specific format | Built-in |
| Re-invoke the model | Build a second `messages.create()` / `responses.create()` call with the appended history | Built-in |
| Terminate when no more calls | Check `stop_reason` / `finish_reason`, write a `while` loop | Built-in (`max_turns` cap) |
| Parallel tool calls | Read array, run sequentially or wire up your own concurrency | Built-in (concurrent by default) |

The pattern shows up in every framework. LangChain wraps it as `AgentExecutor`;
CrewAI wraps it as `Agent(tools=...)` with their own runtime. LazyBridge is the
same idea, with a tiny surface area and no inheritance hierarchy to learn.

---

## Sub-agents as tools (preview of Step 5)

Here's the trick that makes multi-agent systems trivial in LazyBridge: an
**agent is also a tool**. You can pass another agent into `tools=[...]` and the
parent agent can call it like any function.

```python
researcher = Agent(
    engine=LLMEngine("claude-haiku-4-5", system="You look up facts via web search."),
    tools=[web_search],
    name="researcher",
)

writer = Agent(
    engine=LLMEngine("claude-haiku-4-5", system="You write concise prose."),
    tools=[researcher],            # ← sub-agent passed as a tool
    name="writer",
)

print(writer("Write a one-paragraph note on the Voynich Manuscript.").text())
```

The writer decides when to delegate to the researcher. We'll unpack this pattern
properly in Step 5.

---

## Summary

| Concept | Syntax | What it gives you |
|---|---|---|
| Define a tool | A normal Python function with type hints + docstring | Auto-generated schema |
| Wire it up | `Agent(..., tools=[my_function])` | The LLM can call it |
| Multiple tools | `tools=[a, b, c]` | Model picks which to call |
| Explicit metadata | `Tool.wrap(fn, name=..., description=...)` | Override for clarity |
| Trace the loop | `Agent(..., verbose=True)` | See each tool call live |
| Cap the loop | `LLMEngine(..., max_turns=N)` | Prevent runaway calls |
| Sub-agent as a tool | `tools=[other_agent]` | Compose agents (Step 5) |

You now have one agent with a real ability to act. The next step is making
several agents work together — a single agent has limits, and routing work
through specialised sub-agents is how serious LLM applications scale.

---

[**Step 5: Multiple agents working together →**](05-multi-agent.md){ .md-button .md-button--primary }

[← Step 3: Your first agent](03-first-agent.md){ .md-button }
