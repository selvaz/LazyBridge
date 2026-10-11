---
title: Why LazyBridge
---

# Why LazyBridge

LazyBridge connects Claude Code and Codex to Python tools and shared application
state. Use one runtime to produce a result and the other to review it, then
check acceptance with deterministic code.

## Interchangeable engines

ClaudeCodeEngine, CodexEngine and LLMEngine share the Agent contract. Keep
tools, Memory, Store and Session references when changing engines. Native
history, permissions and attachments remain runtime-specific.

[Coding agents in practice](guides/full/coding-agents.md) shows the engine
swap and explains those boundaries.

## Independent review and durable continuity

Wire an opposite-engine judge through Agent(verify=..., max_verify=...).
Rejection supplies feedback; the total attempt budget bounds the loop.
It returns the last draft even on rejection, so review must be followed by
explicit acceptance checks.

[Claude Code writes, Codex reviews](guides/full/claude-writes-codex-reviews.md)
shows the full pattern. Native sessions/threads can continue across processes
by ID or [named alias](guides/full/session-registry.md), without repeatedly
priming the same history.

## Recursive composition and deterministic work

Functions, named agents and Plan-backed pipelines enter tools through the same
contract. Agent.chain and Agent.parallel produce Agents too. Plans validate
their structure at construction; tool-map references are validated when bound
to the Agent's tool map. Store checkpoints preserve internal progress, while
external side effects remain the caller's responsibility.

Use code for calculations, schema validation and boundary tests. The model can
read their results; it does not replace the deterministic acceptance gate.
[Layered composition](concepts/layered-composition.md) and the
[Plan guide](guides/full/plan.md) describe that split.

## LazyTools integrates repository and project work

[LazyTools](https://tools.lazybridge.com/) supplies MCP/connectors, synchronous
code support and the detached repository code bridge. Bridge jobs keep approval
tickets, results and workspace evidence; tier-based review excludes the
producer engine. Projects hold plans, claims, contracts and verification
records in a configured Store.

See [code bridge](https://tools.lazybridge.com/code-bridge/),
[MCP server](https://tools.lazybridge.com/mcp-server/) and
[projects](https://tools.lazybridge.com/projects/) rather than duplicating their
configuration here. Quota routing applies a deterministic eligibility ladder;
the quota routing benchmark is **in progress**, with no comparative results.

## Observability and limits

Attach Session explicitly (or use verbose=True) to collect events and usage.
Session defaults to an in-memory event log; a configured database persists it.
Exporters, including OpenTelemetry, are opt-in. Codex's zero recorded dollar
cost means no reported price, not free execution.

The core public API is Stable. Coding engines have runtime-specific boundaries;
LazyTools declares Alpha. See [maturity](index.md#maturity), the
[changelog](changelog.md) and [comparison](comparison.md).

[Get started](quickstart.md)
