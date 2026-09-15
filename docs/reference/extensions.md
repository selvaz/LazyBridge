# Extension engines & integrations

Framework extensions that live under `lazybridge.ext.*` — `pip install
lazybridge` ships them by default (except `OTelExporter`, which requires the
`[otel]` extra).

> **Connectors moved (0.8).** The MCP connector and the external tool gateway
> are no longer `lazybridge.ext.*` — they moved to the
> [LazyTools](https://tools.lazybridge.com/) package (`lazytools.connectors.{mcp,gateway}`,
> `pip install lazytoolkit`). The old `lazybridge.ext.{mcp,gateway}` deprecation
> shims were removed in 0.9 — import from `lazytools` instead.

For narrative usage see the corresponding guides:
[HumanEngine](../guides/mid/human-engine.md),
[SupervisorEngine](../guides/full/supervisor.md),
[MCP](https://tools.lazybridge.com/mcp/),
[Evals](../guides/mid/evals.md),
[OpenTelemetry](../guides/advanced/otel.md),
[Visualizer](../guides/advanced/visualizer.md).

## Human-in-the-loop

::: lazybridge.ext.hil.HumanEngine

::: lazybridge.ext.hil.SupervisorEngine

::: lazybridge.ext.hil.human_agent

::: lazybridge.ext.hil.supervisor_agent

## MCP integration

Moved to `lazytools.connectors.mcp` — see the [MCP guide](https://tools.lazybridge.com/mcp/)
and the [LazyTools overview](https://tools.lazybridge.com/). Install with
`pip install lazytoolkit[mcp]`.

## Evaluation framework

::: lazybridge.ext.evals.EvalSuite

::: lazybridge.ext.evals.EvalCase

::: lazybridge.ext.evals.EvalReport

::: lazybridge.ext.evals.EvalResult

### Assertion helpers

Ready-made `assertion` callables for `EvalCase` (compose your own too):

::: lazybridge.ext.evals.exact_match

::: lazybridge.ext.evals.contains

::: lazybridge.ext.evals.not_contains

::: lazybridge.ext.evals.min_length

::: lazybridge.ext.evals.max_length

::: lazybridge.ext.evals.llm_judge

## Planners

Multi-step planning agents (`lazybridge.ext.planners`). `orchestrator_agent` /
`blackboard_orchestrator_agent` are the canonical factories; `make_planner` /
`make_blackboard_planner` are backward-compat aliases for the same callables.
See the [Planners guide](../recipes/plan-tool.md).

::: lazybridge.ext.planners.orchestrator_agent

::: lazybridge.ext.planners.blackboard_orchestrator_agent

::: lazybridge.ext.planners.make_plan_builder_tools

::: lazybridge.ext.planners.make_execute_plan_tool

::: lazybridge.ext.planners.PlanSpec

::: lazybridge.ext.planners.StepSpec

## Tiered approval gate

Declarative, per-agent `ApprovalGate` policy for coding engines
(`lazybridge.ext.approval`) — an ordered `Rule` table assigns each tool one of
four tiers (`allow`/`session`/`ask`/`deny`); unmatched calls are denied by
default. `TieredGate` implements `ApprovalGate` directly (no wrapper).
Promoted from the `approval-lab` prototype, with session grants scoped to
`(provider, kind, name, cwd, policy fingerprint)` and every decision recorded
in a structured `AuditRecord`.

::: lazybridge.ext.approval.TieredGate

::: lazybridge.ext.approval.Rule

::: lazybridge.ext.approval.AuditRecord

::: lazybridge.ext.approval.Channel

::: lazybridge.ext.approval.TerminalChannel

## Durable background delegation

Fire-and-forget delegation for long-running agents
(`lazybridge.ext.delegation`) — Store-backed job records kept separate from
the process-local `asyncio` tasks that execute them, process-lifetime
serialized consultant conversations, process-local capped parallel fan-out,
and delegation linked to a [`DurableBlackboard`](#planners) task. Job
records survive restarts only when backed by a persistent `Store`; executing
jobs do not resume after restart, and callers should invoke
`JobRegistry.reclaim_interrupted()` during startup.

Promoted from LazyCEO's generic background-delegation infrastructure after
live production use. Tools produced by the fire-and-forget delegation
factories must be awaited via `Tool.run` from the delegating agent's
persistent event loop. `Tool.run_sync` is unsupported because its normal
short-lived-loop paths cancel pending background tasks when the outer call
returns. The synchronous status/result tools built by `JobRegistry` are not
subject to this restriction.

::: lazybridge.ext.delegation.JobRegistry

::: lazybridge.ext.delegation.make_background_delegate

::: lazybridge.ext.delegation.make_claude_delegate_engine_factory

::: lazybridge.ext.delegation.make_parallel_delegate

::: lazybridge.ext.delegation.make_persistent_consultant

::: lazybridge.ext.delegation.make_plan_delegate

::: lazybridge.ext.delegation.make_claude_writer

::: lazybridge.ext.delegation.make_codex_writer

::: lazybridge.ext.delegation.session_id_key

## Durable knowledge base

Store-backed, cross-session "lessons" a long-running agent can record after
non-obvious success (`lazybridge.ext.knowledge`). With a persistent `Store`,
these reusable notes outlive the plan, task, conversation, and process in
which they were learned. `save_lesson()` is create-only; updates and
retractions are explicit optimistic-CAS operations through
`revise_lesson()`.

Promoted from LazyCEO's `lazyceo.lessons` module after live production use.
Unlike `lazybridge.Memory`, which represents conversation-history context,
the knowledge base holds independently searchable retrospective lessons;
unlike `DurableBlackboard`, it does not represent task state.

::: lazybridge.ext.knowledge.DurableKnowledgeBase

::: lazybridge.ext.knowledge.LessonStatus

::: lazybridge.ext.knowledge.render_lesson_line

::: lazybridge.ext.knowledge.slugify

## OpenTelemetry exporter

::: lazybridge.ext.otel.OTelExporter

## Visualizer

::: lazybridge.ext.viz.Visualizer
