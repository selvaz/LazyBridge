# LazyBridge

[![tests](https://github.com/selvaz/LazyBridge/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/selvaz/LazyBridge/actions/workflows/test.yml)
[![docs](https://github.com/selvaz/LazyBridge/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/selvaz/LazyBridge/actions/workflows/docs.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Build Python pipelines with **Claude Code and Codex as engines**. Swap the
engine while keeping tools and application state; let one engine write and
the other review. **Agent = Engine + Tools + State** is the shared contract.

```python
from lazybridge import Agent, ClaudeCodeEngine, CodexEngine, Memory, Session, Store

def project_status() -> str:
    """Return this example project's status."""
    return "The draft is ready for review."

state = dict(tools=[project_status], memory=Memory(), store=Store(), session=Session())
claude = Agent(name="claude", engine=ClaudeCodeEngine(), **state)
codex = Agent(name="codex", engine=CodexEngine(), **state)
assert claude.memory is codex.memory
assert claude.store is codex.store
assert claude.session is codex.session
```

Functions, agents and Plan-backed pipelines enter through `tools=[...]`.
`LLMEngine` calls provider APIs; `Plan` runs a deterministic DAG; human engines
handle approval and supervision. Every agent returns an `Envelope`.

## Install and authenticate

Python 3.11 or later:

```sh
python -m pip install "lazybridge[claude-code]==1.8.4"
```

Install and sign in to the local coding runtimes using the
[Claude Code guide](https://core.lazybridge.com/guides/full/claude-code-engine/)
and [Codex guide](https://core.lazybridge.com/guides/full/codex-engine/).
Codex needs its CLI, with no Python SDK extra. With a subscription login,
coding runtimes use the account's plan quota; the adapters reuse the configured
authentication and do not force a billing path. For provider APIs, install the
matching extra (`anthropic`, `openai`, `google`, `deepseek`, `litellm`) and
configure that provider's credentials.

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
even when rejected. Use deterministic tests and explicit acceptance records
before shipping. See the full
[Claude Code writes, Codex reviews guide](https://core.lazybridge.com/guides/full/claude-writes-codex-reviews/).

Native history can survive a process restart: Claude has `persist_session` /
`session_id`, Codex has `persist_thread` / `thread_id`, and both support
`session_alias` through `SessionRegistry`. Python `Memory`, `Store` and `Session`
can be reused when changing engines; a native Claude transcript cannot migrate
into a Codex thread.

## LazyTools connects the work

[LazyTools](https://tools.lazybridge.com/) is the integration layer:

- [Code bridge](https://tools.lazybridge.com/code-bridge/): detached repository jobs, durable approvals, results and tier-based opposite-engine review.
- [MCP server](https://tools.lazybridge.com/mcp-server/): code-support tools, projects and connectors.
- [Projects](https://tools.lazybridge.com/projects/): plans, claims, contracts and acceptance evidence in a shared Store.

Its distribution is `lazytoolkit`; its Python namespace is `lazytools`.
Core MCP/gateway shims were removed in 0.9. Follow the LazyTools installation
instructions rather than importing those removed modules.

Code computes results; the LLM reads them. Deterministic gates decide what may
ship. Quota routing chooses from eligible tier rungs; the **quota routing
benchmark is in progress**, with no comparative metrics published here.

## Documentation and status

This README targets **1.8.4**.
The core public API is Stable; coding engines and extensions have their own
boundaries. LazyTools 0.10.1 declares Alpha. See the
[maturity table](https://core.lazybridge.com/#maturity) and [CHANGELOG](CHANGELOG.md).

- [Quickstart](https://core.lazybridge.com/quickstart/)
- [Coding agents and engine differences](https://core.lazybridge.com/guides/full/coding-agents/)
- [Recipes](https://core.lazybridge.com/recipes/) and [API reference](https://core.lazybridge.com/reference/)
- [For LLM assistants](https://core.lazybridge.com/for-llms/)
- [Website and working projects](https://lazybridge.com/)

For development, install `.[test,docs]`, run `python -m pytest`, and build with
`python -m mkdocs build --strict`. The default suite excludes live model calls
and heavy rendering. [CONTRIBUTING](CONTRIBUTING.md) describes the checks.

The longer design rationale and worked examples live in
[Why LazyBridge](https://core.lazybridge.com/why/) (recursive pipelines,
validation, review and exporters), the
[comparison](https://core.lazybridge.com/comparison/), and the
[documentation map](https://core.lazybridge.com/#design-and-documentation-map).
This keeps the README focused while preserving those explanations in the guides.

Issues and contributions are welcome through the repository. See
[SECURITY.md](SECURITY.md) for disclosure and [LICENSE](LICENSE) for Apache 2.0
terms. LazyBridge is designed by selvaz with Claude Code and Codex used as
implementation partners under the owner's direction.
