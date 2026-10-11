# Mental model

An Agent composes **Engine + Tools + State** and returns an Envelope.

```text
Engine ─┐
Tools  ─┼─→ Agent ─→ Envelope (payload, metadata, error)
State  ─┘
```

The same decomposition applies to a single call and a nested checkpointed
pipeline. Add tools and state when the workflow needs them.

## Engine: what decides next

| Engine | Role |
| --- | --- |
| ClaudeCodeEngine | Local Claude Code runtime through the Agent SDK |
| CodexEngine | Local Codex App Server through JSON-RPC |
| LLMEngine | Provider API model/tool loop |
| Plan | Deterministic validated DAG |
| ReplanEngine | Adaptive rounds proposed by a planner |
| HumanEngine / SupervisorEngine | Human decisions and interactive supervision |

Coding engines, provider APIs and Plans share the
[Engine protocol](../guides/advanced/engine-protocol.md), rather than a
BaseEngine class. Configure model, system instructions and native policy on
the chosen engine. Each runtime has its own tools/attachment/schema limits.

## Tools: what the agent can invoke

Functions, Tool objects, ToolProvider catalogues, named child agents and
Plan-backed pipelines enter tools. Type hints and docstrings derive schemas;
Tool.wrap can pin a stable name. Provider-native capabilities enter
native_tools on LLMEngine and execute at the provider; they are not Python
Tool objects. Claude built-ins and Codex native sandbox tools instead use
their own engine policy.

MCP/gateway connectors come from
[LazyTools](https://tools.lazybridge.com/), the integration layer. Claude's
internal MCP adapter and Codex's dynamic tools expose the same Python tools
without requiring a separately registered MCP service.

The function stays the source of truth: its signature, type hints and docstring
derive the schema. No duplicate decorated function is required. Explicit naming
keeps Plan targets stable when a Python function is renamed:

```python
from lazybridge import Agent, ClaudeCodeEngine, Tool

def get_weather(city: str) -> str:
    """Return a demonstration forecast; this is not a live weather service."""
    return f"The demonstration forecast for {city} is sunny."

weather = Tool.wrap(get_weather, name="weather")
assistant = Agent(name="weather-assistant", engine=ClaudeCodeEngine(), tools=[weather])
print(assistant("Describe the demonstration weather in Paris.").text())
```

## State: continuity, records and observability

| Primitive | Purpose | Default |
| --- | --- | --- |
| Memory | Application conversation history | None; opt in |
| Store | In-memory or SQLite shared key/value records | None; persistence needs db |
| Session | Event log, exporters and graph; in-process approval/runtime slots | None; verbose=True creates one |
| Envelope | Typed task/result/context/error/usage | Every run returns one |
| SessionRegistry | Durable native alias-to-ID bindings | Used when coding engines have session_alias |

Claude native sessions and Codex threads are distinct from the observability
Session. persist_session/persist_thread and native IDs retain runtime history
across processes. Aliases name those IDs. Native resume omits repeated Memory
priming, but explicit context remains available.

Memory, Store and Session are optional. A one-shot Agent can use none of them,
while every call still returns an Envelope. A Store lets several agents exchange
structured records without relying on free-form conversation history. An
in-memory Store disappears with the process; Store(db=...) opts into persistence.

## A working agent

```python
from lazybridge import Agent, LLMEngine

assistant = Agent(engine=LLMEngine("gpt-5.4-mini"), name="explainer")
result = assistant("Explain Engine + Tools + State in one sentence.")
print(result.text())
```

Here the engine is a provider model loop, tools are absent, and only the result
Envelope is present as state. Agent(engine=...) makes the engine choice visible.
Agent(model_id) is sugar for LLMEngine; see
[canonical vs sugar](canonical-vs-sugar.md) before using shortcuts.

Call agent(task) synchronously, await agent.run(task) in async applications, or
iterate agent.stream(task) for text. A sync call blocks an existing event loop;
use the async entry point in request handlers.

## Swap engines while keeping state

```python
from lazybridge import Agent, ClaudeCodeEngine, CodexEngine, Memory, Session, Store

def project_status() -> str:
    """Return a deterministic status."""
    return "Ready for review."

state = dict(tools=[project_status], memory=Memory(), store=Store(), session=Session())
claude = Agent(name="claude", engine=ClaudeCodeEngine(), **state)
codex = Agent(name="codex", engine=CodexEngine(), **state)
assert claude.memory is codex.memory
assert claude.store is codex.store
assert claude.session is codex.session
```

Python state is shared by reference. Native history is not migrated between
engines. Code computes results and checks deterministic acceptance; the LLM
reads those results and proposes the next action.

## Where the shorthand needs care

- An Agent is not necessarily an LLM. Plan executes declared control flow;
  models run only in steps that invoke model-driven workers.
- A Plan is a deterministic graph, while Agent composition uses Python Tools
  and helpers. GraphSchema provides inspection rather than replacing that API.
- A Tool may trigger a nested Agent run. Nested usage metadata rolls up;
  failures still need explicit application handling.
- Application state is not necessarily durable. Store persistence, native
  history and SessionRegistry bindings are separate choices.
- Native tools are not Python functions. Provider capabilities and coding
  runtime built-ins have their own permission and execution boundaries.

See [coding agents](../guides/full/coding-agents.md),
[one writes, the other reviews](../guides/full/claude-writes-codex-reviews.md),
[everything is a tool](everything-is-a-tool.md) and the
[progressive complexity ladder](progressive-complexity.md).
