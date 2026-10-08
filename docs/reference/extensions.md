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


### Replaceable delegation lifecycle (1.8)

`make_background_delegate`, `make_parallel_delegate`, and `make_plan_delegate`
accept `job_runner: JobRunner | None = None`. Omitting it preserves the existing
worker. `JobRunner(prepare=..., register=..., execute=..., finalize=..., rollback=...)`
exposes four sync or async phases, each receiving the same `JobContext`.
The default prepare constructs both engine and Agent; register calls
`JobRegistry.begin_execution(job_id)` to CAS a running, unstarted record to
started; execute runs the prepared Agent; finalize stores the result using
`JobRegistry.update(job_id, changes, only_active=True)` while preserving unknown fields.
A custom register returns `False` when its CAS loses. No execution or terminal
rewrite follows that refusal. The rollback callback runs once on refusal, phase
failure, or cancellation, including partial setup. Its context exposes `phase`,
`error`, `started`, `prepared`, `result`, and caller-owned `metadata`; callers can
capture claim ownership and pre-claim attempts there. Reservation settlement is
owned by the runner: refund before execution, release after execution starts.
Callbacks should not also settle that reservation. Exceptions propagate after
cleanup; a failing Store makes terminal recording best effort. No lifecycle can
write a terminal record while its Store is unavailable.


### Background admission without confirmation (1.8)

`make_background_delegate(..., admission_gate=...)` checks once before registration
when `pre_confirm` is absent. A refusal returns `rejection_text()` directly and
creates no job. With confirmation, admission remains after approval. A grant is
owned by one job: setup, scheduling, CAS refusal, and cancellation before execution
refund it; execution completion/failure/cancellation release it. Admission enables
the optional CAS runner by default, so engine and Agent setup happen before the
execution marker. `make_claude_writer` forwards the same gate. Quotas, approval
spending, rejection wording, and reservation storage remain caller policy.
Cancellation before a task's first step schedules retained cleanup on the host's
persistent event loop; await the retained tasks before closing that loop.


### Writer aliases and reusable provider factories (1.8)

Both writers accept `session_alias: str | None`,
`session_registry: SessionRegistry | None`, and
`accept_session_alias_override: bool = False`. A generated `session_alias` tool
parameter is a durable name scoped to the engine cwd. The existing optional
`session` parameter continues to mean a native Codex thread or Claude session id.
Combining a native id and an alias in the same factory call raises ValueError.
The engines resolve and bind aliases through `lazybridge.engines.sessions`;
reconstructing the writer and registry after restart resumes the saved native id.

`make_codex_writer_engine_factory(*, workspace_root, gate, model=None, effort=None,
session_alias=None, session_registry=None, writable_roots=None)` returns a callable
accepting keyword `cwd`, `model`, `effort`, `session`, `session_alias`, and
`writable_roots`. `make_claude_writer_engine_factory` takes the same common
settings, with `model="sonnet"` and `max_turns=60` instead of writable roots;
its callable also accepts a per-call `max_turns`. Every call builds a fresh engine
and keeps the existing writer permission configuration and unbounded request timeout.

Writers also forward optional `engine_factory`, `extra_params`, `guard`, and
`job_runner` to background delegation. For example, declare
`extra_params={"cwd": ExtraParam(required=True)}` and validate allowed repositories
in your own guard; the provider factory receives the chosen cwd. Custom factories
own their build-time settings; per-call arguments are forwarded unchanged. No
repository resolution, model rules, or quota policy is supplied by these adapters.
