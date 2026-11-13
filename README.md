# LazyBridge

[![tests](https://github.com/selvaz/LazyBridge/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/selvaz/LazyBridge/actions/workflows/test.yml)
[![docs](https://github.com/selvaz/LazyBridge/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/selvaz/LazyBridge/actions/workflows/docs.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**No glue code. One contract. Agents, pipelines and Plans nest as tools.**

LazyBridge is a Python library for composing work as **Agent = Engine + Tools + State**. A function, another agent, or a Plan-backed pipeline all go in `tools=[...]`. The engine is whatever decides the next step: any supported LLM, a deterministic `Plan`, or a human (`HumanEngine`, Alpha).

## 30-second demo (no API keys)

```sh
python -m pip install lazybridge
```

```python
from lazybridge import Agent, Plan, Step, from_step
from lazybridge.testing import MockAgent

research = MockAgent("Rome: 18°C, clear", name="research")
write = MockAgent(lambda env: f"Pack a light jacket. ({env.context or env.task})", name="write")
brief = Agent(name="brief", engine=Plan(Step("research"), Step("write", context=from_step("research"))), tools=[research, write])
desk = Agent(name="desk", engine=Plan(Step("brief")), tools=[brief])
print(desk("Packing brief for Rome").text())
# Pack a light jacket. (Rome: 18°C, clear)
```

`brief` is a Plan-backed agent used as the only tool of `desk`. `write` is still a normal agent: you can call it alone, then put it back in the same `tools` list.

## Install

Python 3.11 or later. For the snippet above, `pip install lazybridge` is enough (current public release: **1.8.7**).

For live engines, install the matching extra and credentials, then run the bundled demo:

```sh
python -m pip install "lazybridge[claude-code,deepseek]==1.8.7"
python -m lazybridge.demo
```

The demo finds which engines this machine can use, gives each the same Python tool and question, then lets one engine write and another review. Enable any one of them, or all three:

| Engine | Enable it |
|---|---|
| DeepSeek API | set `DEEPSEEK_API_KEY` (key from platform.deepseek.com) |
| Claude Code | `npm install -g @anthropic-ai/claude-code`, then run `claude` once to sign in |
| Codex | `npm install -g @openai/codex`, then run `codex login` |

```text
Engines on this machine:
  [x] deepseek
  [x] claude-code
  [x] codex

1) Same tool, same question, 3 engine(s):
   deepseek     LazyBridge treats an agent as an engine plus tools and state, where engines are swappable peers ...
   claude-code  ...
   codex        LazyBridge defines agents as engines plus tools and state, with swappable engines ...

2) deepseek writes, codex reviews (verify=):
print([i**2 for i in range(1, 6)])
```

The demo is one short file, [`lazybridge/demo.py`](https://github.com/selvaz/LazyBridge/blob/main/lazybridge/demo.py):
read it to see the whole API it uses.

```python
from lazybridge import Agent, ClaudeCodeEngine, CodexEngine, LLMEngine, Memory, Session, Store

def project_status() -> str:
    """Return this example project's status."""
    return "The draft is ready for review."

state = dict(tools=[project_status], memory=Memory(), store=Store(), session=Session())
api = Agent(name="api", engine=LLMEngine("claude-haiku-4-5"), **state)
claude = Agent(name="claude", engine=ClaudeCodeEngine(), **state)
codex = Agent(name="codex", engine=CodexEngine(), **state)
assert api.memory is claude.memory is codex.memory
assert api.store is claude.store is codex.store
assert api.session is claude.session is codex.session

# An agent is a tool too: the lead can delegate to Codex.
lead = Agent(name="lead", engine=ClaudeCodeEngine(), tools=[project_status, codex])
```

`LLMEngine` calls provider APIs (Anthropic, OpenAI, Google, DeepSeek, LiteLLM, and more). `Plan` runs a deterministic DAG. Human engines (`HumanEngine`, `SupervisorEngine`) are Alpha. Every agent returns an `Envelope`.

With a subscription login, coding runtimes use the account's plan quota instead of per-token API billing. The adapters reuse the configured authentication and do not force a billing path. Swap one engine for another and keep tools, memory and state; a native Claude transcript still cannot migrate into a Codex thread.

Install and sign in to the local coding runtimes using the
[Claude Code guide](https://core.lazybridge.com/guides/full/claude-code-engine/)
and [Codex guide](https://core.lazybridge.com/guides/full/codex-engine/).
Codex needs its CLI, with no Python SDK extra. For provider APIs, install the
matching extra (`anthropic`, `openai`, `google`, `deepseek`, `litellm`) and
configure that provider's credentials. Local models such as Ollama go through
the `litellm` extra (`LLMEngine("litellm/...")`); LM Studio can use an
OpenAI-compatible base URL. See the
[providers guide](https://core.lazybridge.com/guides/advanced/providers/).

## One writes, the other reviews

```python
from lazybridge import Agent, ClaudeCodeEngine, CodexEngine

reviewer = Agent(name="codex-reviewer", engine=CodexEngine(
    system='Start with "approved" or "rejected: <reason>". Check boundary cases.',
))
writer = Agent(name="claude-writer", engine=ClaudeCodeEngine(),
               verify=reviewer, max_verify=3)
result = writer("Draft a function that blocks weekend deployment.")
print(result.text())
```

`verify=` is configured on the Agent. Rejection supplies feedback for the next
attempt; `max_verify` counts total attempts. Exhaustion returns the final draft
even when rejected. `result.ok` means the run finished without an execution
error, not that the reviewer accepted the work. Use deterministic tests and
explicit acceptance records before shipping. See the full
[Claude Code writes, Codex reviews guide](https://core.lazybridge.com/guides/full/claude-writes-codex-reviews/).

Native history can survive a process restart: Claude has `persist_session` /
`session_id`, Codex has `persist_thread` / `thread_id`, and both support
`session_alias` through `SessionRegistry`. Python `Memory`, `Store` and `Session`
can be reused when changing engines.

Deterministic `Plan` DAGs also support checkpoint/resume and `run_many` fan-out,
alongside `Agent.parallel` and a durable control plane (atomic claims, fencing,
an append-only ledger) for many workers on one machine.

## LazyTools (Alpha)

[LazyTools](https://tools.lazybridge.com/) is the integration layer. It is
**Alpha**, not part of Core:

- [Code bridge](https://tools.lazybridge.com/code-bridge/): detached repository jobs, durable approvals, results and tier-based opposite-engine review.
- [MCP server](https://tools.lazybridge.com/mcp-server/): code-support tools, projects and connectors.
- [Projects](https://tools.lazybridge.com/projects/): plans, claims, contracts and acceptance evidence in a shared Store.

Its distribution is `lazytoolkit`; its Python namespace is `lazytools`. Install
from the GitHub tag (add the `engines` extra for Claude or tier routing):

```sh
pip install "lazytoolkit[mcp] @ git+https://github.com/selvaz/LazyTools@v0.10.6"
```

Core MCP/gateway shims were removed in 0.9. Follow the LazyTools installation
instructions rather than importing those removed modules.

Code computes results; the LLM reads them. Deterministic gates decide what may
ship. Quota routing chooses from eligible tier rungs; the **quota routing
benchmark is in progress**, with no comparative metrics published here.

## Documentation and status

This README targets **1.8.7**.
The core public API is Stable; coding engines and extensions have their own
boundaries. Claude Code and Codex engines are Shipped, not under the Core
Stable guarantee. `HumanEngine` and `SupervisorEngine` are Alpha. LazyTools
is Alpha. See the
[maturity table](https://core.lazybridge.com/#maturity) and [CHANGELOG](CHANGELOG.md).

- [Quickstart](https://core.lazybridge.com/quickstart/)
- [Why LazyBridge](https://core.lazybridge.com/why/) (composition, Plans, review, observability)
- [Comparison](https://core.lazybridge.com/comparison/) and [framework guide](https://core.lazybridge.com/best-python-ai-agent-frameworks/)
- [Coding agents and engine differences](https://core.lazybridge.com/guides/full/coding-agents/)
- [Recipes](https://core.lazybridge.com/recipes/) and [API reference](https://core.lazybridge.com/reference/)
- [For LLM assistants](https://core.lazybridge.com/for-llms/)
- [Website](https://lazybridge.com/)

For development, install `.[test,docs]`, run `python -m pytest`, and build with
`python -m mkdocs build --strict`. The default suite excludes live model calls
and heavy rendering. [CONTRIBUTING](CONTRIBUTING.md) describes the checks.

The longer design rationale lives in the guides linked above, including the
[documentation map](https://core.lazybridge.com/#design-and-documentation-map).
This keeps the README focused.

Issues and contributions are welcome through the repository. See
[SECURITY.md](SECURITY.md) for disclosure and [LICENSE](LICENSE) for Apache 2.0
terms. LazyBridge is designed by selvaz with Claude Code and Codex used as
implementation partners under the owner's direction.
