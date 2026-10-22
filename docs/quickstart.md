# Quickstart

Start with Claude Code, swap to Codex with the same tools and state, then add
independent review. Python 3.11 or later is required.

## Install and sign in

```sh
python -m pip install "lazybridge[claude-code]==1.8.3"
```

Follow [Claude Code setup](guides/full/claude-code-engine.md#setup) and
[Codex setup](guides/full/codex-engine.md#setup) to install and authenticate the
local runtimes. Codex needs no Python SDK extra. The engines reuse runtime
authentication: a subscription login uses plan quota, while another configured
login may use API billing.

## Call a tool

```python
from lazybridge import Agent, ClaudeCodeEngine, Memory, Session, Store

def project_status() -> str:
    """Return a deterministic status for this example."""
    return "The draft is ready for review."

state = dict(tools=[project_status], memory=Memory(), store=Store(), session=Session())
agent = Agent(name="assistant", engine=ClaudeCodeEngine(system="Be concise."), **state)
result = agent("Read project_status and report the next step.")
print(result.text())
```

Type hints and the docstring become the tool schema. For a stable public tool
name use `Tool.wrap(project_status, name="project_status")`. The callable form
above is supported. `result` is an `Envelope`; `.payload` gives typed output
when `output=` is set, and `.text()` gives a string.

## Swap the engine

Continue in the same Python process after the previous block:

```python
from lazybridge import Agent, CodexEngine

agent = Agent(name="assistant", engine=CodexEngine(system="Be concise."), **state)
result = agent("Using the shared context, report the next step.")
print(result.text())
```

The tool list, Memory, Store and observability Session stay in Python. Native
permissions and attachment support differ. Native sessions/threads are
engine-specific; they are not transferred by changing `engine=`. See
[durable aliases](guides/full/session-registry.md) for cross-process continuity.

## Add review

```python
from lazybridge import Agent, ClaudeCodeEngine, CodexEngine

reviewer = Agent(name="reviewer", engine=CodexEngine(
    system='Start with "approved" or "rejected: <reason>". Check boundary cases.',
))
writer = Agent(name="writer", engine=ClaudeCodeEngine(), verify=reviewer, max_verify=3)
print(writer("Draft a function that blocks weekend deployment.").text())
```

Three attempts means at most three writer runs, each reviewed. The final draft
is returned even if all reviews reject it. The
[complete collaboration guide](guides/full/claude-writes-codex-reviews.md) explains
feedback, deterministic acceptance and repository work through LazyTools.

## Run offline or use a provider API

For a credential-free example:

```python
from lazybridge.testing import MockAgent

agent = MockAgent(["The draft is ready for review."], name="offline")
assert agent("Report status").text() == "The draft is ready for review."
```

For provider APIs, install the matching extra (`anthropic`, `openai`, `google`,
`deepseek`, `litellm`), set the provider's credentials, and construct
`Agent.from_provider("anthropic", tier="cheap")` or `Agent(engine=LLMEngine(model))`.
Tier aliases are catalogue choices; they are distinct from LazyTools quota routing.

A provider-only installation can be explicit too:

```sh
python -m pip install "lazybridge[anthropic]==1.8.3"
```

Set ANTHROPIC_API_KEY in the environment running Python for Anthropic API calls
(use OPENAI_API_KEY for OpenAI, GOOGLE_API_KEY for Google, or DEEPSEEK_API_KEY
for DeepSeek). Supply your own key through your shell or secret manager, never
commit it. Coding-runtime login and these provider API credentials are separate.

```python
from lazybridge import Agent, LLMEngine, Tool

def get_weather(city: str) -> str:
    """Return a deterministic demonstration forecast, not live weather."""
    return f"The demonstration weather in {city} is sunny."

agent = Agent.from_provider("anthropic", tier="cheap")
print(agent("Explain LazyBridge in one sentence.").text())

# Pin a catalogue ID when a particular model behavior is required.
weather_agent = Agent(
    name="weather", engine=LLMEngine("claude-haiku-4-5"),
    tools=[Tool.wrap(get_weather, name="get_weather")],
)
print(weather_agent("Explain the demonstration weather in Paris.").text())
```

The model chooses whether to call a tool. verbose=True enables a console Session
to inspect the loop; it does not force a tool call or guarantee an answer's wording.
Canonical Agent(engine=...) exposes the options you will configure as the system
grows. Agent(model_id) is shorthand for a provider engine; a bare provider name
is ambiguous and should be replaced with Agent.from_provider(..., tier=...).

Use `await agent.run(task)` inside async applications. `agent(task)` blocks until
completion, including when called from an existing event loop. Engines also
expose `agent.stream(task)` for incremental text.

Continue with the [mental model](concepts/mental-model.md),
[coding agents](guides/full/coding-agents.md), [tool guide](guides/basic/tool.md),
[recipes](recipes/index.md), or [code generation contract](for-llms/codegen-contract.md).

## What to read next

| Goal | Guide |
| --- | --- |
| Understand the composition contract | [Mental model](concepts/mental-model.md), [everything is a tool](concepts/everything-is-a-tool.md) |
| Grow to a pipeline | [Progressive complexity](concepts/progressive-complexity.md), [Plan](guides/full/plan.md) |
| Choose tool schemas | [Tool](guides/basic/tool.md), [tool wrapping](decisions/tool-wrapping.md) |
| Run an end-to-end tool loop | [ReAct agent recipe](recipes/react-agent.md) |
| Add continuity or tracing | [Memory](guides/mid/memory.md), [Store](guides/mid/store.md), [Session](guides/mid/session.md) |
| Ask an assistant to generate code | [For LLM assistants](for-llms/index.md) |

In an async web handler, use await agent.run directly. The synchronous helper
waits for a worker thread when a loop is already running, which blocks the
handler and allocates a thread per call.
