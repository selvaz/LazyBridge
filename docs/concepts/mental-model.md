# Mental model

> An agent in LazyBridge is the composition of three elements — and only
> these three. **Engine + Tools + State.**

```text
                ┌──────────────────────┐
    Engine ───► │                      │
                │        Agent         │
    Tools  ───► │                      │ ──► Envelope (payload + metadata)
                │                      │
    State  ───► │                      │
                └──────────────────────┘
```

That decomposition is the only thing you need to internalise before reading
anything else. Whether the agent is a single model call or a checkpointed
multi-region pipeline with human approvals, it is always the same three
pieces.

## Engine — what decides

The engine decides **what happens next**. LazyBridge ships four:

| Engine | What it is | When |
|---|---|---|
| `LLMEngine` | An LLM dynamically picks tools and arguments | Most exploratory or open-ended work |
| `Plan` | A deterministic, validated DAG of steps | When you need auditability, repeatability, or cost control |
| `HumanEngine` | Pauses for human approval / redirection | Compliance gates, high-stakes outputs |
| `SupervisorEngine` | REPL-style human supervision with retry / inspect / rerun | Interactive debugging, demos, sensitive automation |

You can also implement the `BaseEngine` protocol and supply your own. The
agent does not care: from the outside, every engine satisfies the same
contract — take an input `Envelope`, produce an output `Envelope`.

This is why LazyBridge does not equate "agent" with "LLM". **Determinism is
a first-class engine choice**, not a workaround. The same `Agent` object can
swap an `LLMEngine` for a `Plan` without touching the rest of your code.

## Tools — what the agent can do

Tools are the agent's capabilities. In LazyBridge, anything that exposes a
useful capability is a tool: a plain Python function, another agent, a
`Plan`, an MCP server, a provider-native server-side tool (web search, code
execution).

```python
from lazybridge import Agent, LLMEngine, Tool

def get_weather(city: str) -> str:
    """Return the current weather for ``city``."""
    ...

agent = Agent(
    engine=LLMEngine("gpt-5.4-mini"),
    tools=[Tool.wrap(get_weather, name="get_weather")],
)
```

There is no second JSON schema to define and no `@tool` decorator to
remember. LazyBridge inspects the signature, type hints, and docstring to
build the schema the LLM sees. The function stays the source of truth.

The same uniformity holds at every level of composition — see
[Everything is a tool](everything-is-a-tool.md) for the full story.

## State — what persists

State is the part of the agent responsible for continuity, traceability,
and shared information.

| Primitive | Purpose | Default |
|---|---|---|
| `Memory` | Conversation history with configurable compression | None — opt in |
| `Session` | Event bus + observability container; lives across runs | None — opt in |
| `Store` | Cross-run key-value blackboard, in-memory or SQLite | None — opt in |
| `Envelope` | The typed payload + metadata that flows between every component | Always present |

Each one is optional. A trivial agent has no `Memory`, no `Session`, no
`Store` — only the `Envelope` that carries its result. State is something
you add when the workflow demands it, not boilerplate you ship from day
one.

The `Store` is especially important when a system grows beyond one agent.
Multiple agents and pipeline steps can read and write to it, exchanging
structured information without relying on fragile free-form text passing.

## A working agent

```python
from lazybridge import Agent, LLMEngine

agent = Agent(
    engine=LLMEngine("gpt-5.4-mini"),
)
result = agent("Explain LazyBridge in one sentence.")
print(result.text())
```

In this example:

- The **engine** is `LLMEngine("gpt-5.4-mini")`.
- There are no **tools**.
- The only **state** is the result `Envelope`.

`Agent(engine=...)` with each argument on its own line is the
canonical shape — every example in `examples/` follows it, and every
rung on the [progressive complexity ladder](progressive-complexity.md)
adds to it without changing it. Shorter forms exist
(`Agent("gpt-5.4-mini")`, `Agent("gpt-5.4-mini")`)
but they're sugar: convenient for one-liners, but they hide the engine
choice that you'll need to configure as soon as the agent does
anything non-trivial. Learn the canonical form first.

Calling `agent(task)` is the canonical sync entry point. An
`await agent.run(task)` async form and an
`async for chunk in agent.stream(task)` streaming form exist when you
need them — start with the sync call and opt into async only where it
matters.

The same `Agent` would be ready for tools, memory, sessions, plans, or
human approvals the moment you needed any of them — without rewriting.

## When this model bends

A few honest caveats to keep the model from being misleading:

- **The agent is not an LLM.** When `engine=Plan(...)`, no model is called
  to drive the loop — the orchestration is purely deterministic. The model
  only enters where a `Step` happens to dispatch to an LLM-driven sub-agent.
- **The agent is not a graph.** LazyBridge does build a `GraphSchema` for
  inspection, but composition is expressed in plain Python. There is no
  separate graph DSL to learn.
- **Tools are not always functions.** When you wrap an `Agent` as a tool
  (`agent.as_tool(...)`), the "function call" is actually a recursive
  agent run. Cost, errors, and tokens roll up transitively.
- **State is not always durable.** A `Store` with no `db=` argument is
  in-memory and disappears when the process exits. Persistence is opt-in.

## See also

- [Everything is a tool](everything-is-a-tool.md) — the composition rule
  that lets agents, plans, MCP servers, and pipelines all behave the same
  way.
- [Progressive complexity](progressive-complexity.md) — how to grow from
  the three-line agent above to a checkpointed production pipeline
  without changing the mental model.
- [Quickstart](../quickstart.md) — the same three lines, but with a tool
  and a real run.
