---
title: Compare workflow approaches
---

# Compare workflow approaches

Choose a workflow contract by testing it against your own repository, approval
policy and restart requirements. This page describes LazyBridge's current
surface; it does not claim that another framework lacks an equivalent feature.

## Choices within LazyBridge

| Need | Mechanism | Boundary |
| --- | --- | --- |
| Claude Code / Codex behind one Python agent | ClaudeCodeEngine / CodexEngine | Shares tools and application state; native transcripts stay engine-specific. |
| Provider API calls | LLMEngine | Provider credentials and capability differences still apply. |
| One writes, the other reviews | Explicit opposite-engine verify judge | Bounded retries; rejected final output can still be returned. |
| Cross-process native continuity | persist_session / persist_thread, IDs, SessionRegistry aliases | Registry file locks do not lock native transcripts across processes. |
| Declared workflow with resume | Plan, Step, Store checkpoints | Internal progress persists; external effects are not rolled back. |
| Human decisions and inspection | HumanEngine / SupervisorEngine | Workflow decisions differ from per-action native approval gates. |
| Repository delegation and durable tickets | LazyTools code bridge | Tier-based review excludes producer; detached jobs still have an OS process lifetime. |
| Shared project acceptance | LazyTools projects | Explicit Store and caller authorization/acceptance policy. |
| Quota-based selection | LazyTools routing | Deterministic eligibility policy; benchmark in progress. |

See [coding agents](guides/full/coding-agents.md), the
[collaboration guide](guides/full/claude-writes-codex-reviews.md) and
[LazyTools code bridge](https://tools.lazybridge.com/code-bridge/).

## Evaluate other frameworks from their primary documentation

[LangGraph](https://docs.langchain.com/oss/python/langgraph/overview),
[CrewAI](https://docs.crewai.com/) and
[Pydantic AI](https://ai.pydantic.dev/) offer different workflow abstractions.
Use their current documentation and a versioned test case to compare behavior;
a static matrix cannot establish feature absence, framework quality or a node
count at which one becomes better.

Useful acceptance criteria include:

- Correctness of the final diff and deterministic boundary tests.
- Native session recovery and duplicate-side-effect handling after interruption.
- Scope of runtime sandboxes versus application tool permissions.
- Reviewer identity and acceptance evidence distinct from execution success.
- Observability, dependency requirements and your existing integration costs.

The [framework documentation directory](best-python-ai-agent-frameworks.md)
links to additional projects. No competitive benchmark is published here.

## Migration considerations

Map functions to Tool.wrap, native provider capabilities to native_tools, and
sub-agents to named Agents in tools. Linear stages can use Agent.chain;
branching/checkpointed workflows use Plan/Step. Keep engine instructions on the
engine, including system, reasoning and deadlines. Match each engine's
structured output and attachment limits before transferring a workflow.

For MCP and HTTP gateways use lazytools.connectors.mcp/gateway. They are
integration-layer packages, not core extensions. Validate credentials,
permissions, Store persistence and deterministic acceptance in the target
environment before replacing an existing workflow.

[Quickstart](quickstart.md) · [Why LazyBridge](why.md)
