# Quickstart

Start with Claude Code, swap to Codex with the same tools and state, then add
independent review. Python 3.11 or later is required.

## Install and sign in

```sh
python -m pip install "lazybridge[claude-code]==1.8.0"
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

Use `await agent.run(task)` inside async applications. `agent(task)` blocks until
completion, including when called from an existing event loop. Engines also
expose `agent.stream(task)` for incremental text.

Continue with the [mental model](concepts/mental-model.md),
[coding agents](guides/full/coding-agents.md), [tool guide](guides/basic/tool.md),
[recipes](recipes/index.md), or [code generation contract](for-llms/codegen-contract.md).
