# LazyBridge

Build Python pipelines with **Claude Code and Codex as engines**. Swap the
engine while keeping tools and application state; let one engine write and the
other review. LazyBridge uses one contract: **Agent = Engine + Tools + State**.

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

With a subscription login, coding runtimes use the account's plan quota.
The adapters reuse runtime authentication; the configured login can also use
another billing path. Provider APIs remain available through `LLMEngine`.
Native Claude sessions and Codex threads keep their own history: switching
engines shares Python state, and does not migrate a native transcript.

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

`verify=` sends the draft and original task to the reviewer, then feeds rejection
back into the next attempt. `max_verify` counts total attempts. Exhaustion
returns the last draft even if rejected; use tests and explicit acceptance
records before shipping. Opposite-engine review is wired by the caller here.

[Claude Code writes, Codex reviews](guides/full/claude-writes-codex-reviews.md)
shows the complete pattern and links to repository delegation.

## Durable conversations and connected work

Use `persist_session` / `session_id` for Claude, `persist_thread` / `thread_id`
for Codex, or `session_alias` with a [SessionRegistry](guides/full/session-registry.md).
Keep native history distinct from `Memory`, `Store` and the observability `Session`.

[LazyTools](https://tools.lazybridge.com/) is the integration layer. Its
[code bridge](https://tools.lazybridge.com/code-bridge/) supports detached jobs,
durable approval tickets, results and tier-based opposite-engine review.
Its [MCP server](https://tools.lazybridge.com/mcp-server/) exposes coding and
project tools; [projects](https://tools.lazybridge.com/projects/) hold plans,
claims and acceptance evidence. Code computes results and renders schedules;
deterministic checks decide whether a result may ship.

Quota routing applies a deterministic tier ladder to available telemetry.
The **quota routing benchmark is in progress**; there are no comparative
quality, speed or savings results to report.

## Start here

- [Quickstart](quickstart.md): install, call a tool, swap engines and add review.
- [Coding agents](guides/full/coding-agents.md): engine differences and composition.
- [Claude Code](guides/full/claude-code-engine.md) and [Codex](guides/full/codex-engine.md): setup and options.
- [Permissions](guides/full/coding-agent-config.md): native controls and approvals.
- [Mental model](concepts/mental-model.md), [recipes](recipes/index.md) and [reference](reference/index.md).
- [For LLM assistants](for-llms/index.md): code generation contract and searchable corpus.
- [Website and working projects](https://lazybridge.com/): examples of the connected workflow.

## Design and documentation map

Engine choice, recursive Tools and optional state keep composition explicit.
Use deterministic Plans when control flow should be declared; use model loops
when the worker should choose. Type hints derive tool schemas, while Envelopes
and opt-in Session/exporters expose results and run evidence.

| Need | Read next |
| --- | --- |
| Composition and convenience forms | [Everything is a tool](concepts/everything-is-a-tool.md), [progressive complexity](concepts/progressive-complexity.md), [canonical vs sugar](concepts/canonical-vs-sugar.md) |
| Basic contracts | [Agent](guides/basic/agent.md), [Tool](guides/basic/tool.md), [Envelope](guides/basic/envelope.md), [native tools](guides/basic/native-tools.md) |
| Application state and quality | [Memory](guides/mid/memory.md), [Store](guides/mid/store.md), [Session](guides/mid/session.md), [guards](guides/mid/guards.md), [verify=](guides/mid/verify.md), [evals](guides/mid/evals.md) |
| Pipelines and human decisions | [Chain](guides/mid/chain.md), [parallel](guides/mid/parallel.md), [as tool](guides/mid/as-tool.md), [HumanEngine](guides/mid/human-engine.md), [SupervisorEngine](guides/full/supervisor.md) |
| Deterministic production work | [Plan](guides/full/plan.md), [Step](guides/full/step.md), [sentinels](guides/full/sentinels.md), [routing](guides/full/routing.md), [parallel bands](guides/full/parallel-plan-steps.md), [checkpoint](guides/full/checkpoint.md) |
| Runtime inspection | [Exporters](guides/full/exporters.md), [GraphSchema](guides/full/graph-schema.md), [OpenTelemetry](guides/advanced/otel.md), [Visualizer](guides/advanced/visualizer.md) |
| Extensions and integration boundaries | [Engine protocol](guides/advanced/engine-protocol.md), [BaseProvider](guides/advanced/base-provider.md), [provider catalogue](guides/advanced/providers.md), [Plan serialization](guides/advanced/plan-serialize.md), [multimodal](guides/mid/multimodal.md), [LazyTools MCP](https://tools.lazybridge.com/mcp/) and [gateway](https://tools.lazybridge.com/gateway/) |
| Choosing and recovering | [Decisions](decisions/index.md), [errors](errors.md), [framework comparison](comparison.md), [why](why.md) |

## Maturity

This documentation targets **LazyBridge 1.8.4**. The core is Stable; this does not establish a Stable
guarantee for every engine or extension. LazyTools 0.10.1 declares Alpha.

| Subsystem | Status | Boundary |
| --- | --- | --- |
| `Agent`, `LLMEngine`, `Tool`, `Envelope`, `Plan`, `Step`, sentinels | Stable | Core public API; breaking changes require a major version. |
| `Memory`, `Store`, `Session`, event log, exporters, `GraphSchema` | Stable | Persistence and exporters are opt-in; credential redaction is enabled by default. |
| Provider adapters | Stable | Model catalogues and runtime capabilities change independently. |
| Checkpoint/resume and guards | Stable | Checkpoints do not roll back external side effects. |
| Claude Code / Codex engines, native persistence and aliases | Shipped | Runtime-specific permissions, attachments and history; no blanket engine maturity promotion. |
| Approval, delegation and knowledge extensions | Shipped | Generic hooks; application policy remains caller-defined. |
| MCP, code support, code bridge, projects and quota routing | LazyTools Alpha | See the integration-layer docs rather than importing removed core connectors. |
| `NativeTool`, `HumanEngine`, `SupervisorEngine` | Alpha | See the corresponding guides for capability boundaries. |
| Evals and Visualizer | Experimental | Extension APIs may consolidate. |
| Provider fallback-chain tables; automatic PII redaction | Planned | Explicit `Agent(fallback=...)` works; default redaction masks credentials, not arbitrary personal data. |

See the [changelog](changelog.md) and [migration history](migrations/1.0-to-0.7.md)
for the withdrawn, incompatible April 2026 release numbered 1.0.0.
