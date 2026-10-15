# Extensions (`lazybridge.ext.*`) and the control plane

Read when using human-in-the-loop engines, approval policy and queues,
background delegation, durable to-do boards, a lessons store, evals,
OpenTelemetry, the visualizer, or `lazybridge.control_plane`. Source of truth:
`lazybridge/ext/<name>/` (each has `__all__`); docs
`docs/reference/extensions.md`, `docs/guides/core-vs-ext.md`,
`docs/recipes/durable-blackboard.md`, `docs/guides/advanced/control-plane.md`.

## The import boundary

| Layer | Where | Holds |
|---|---|---|
| Core | `lazybridge` | `Agent`, engines, `Plan`, `Tool`, `Envelope`, `Memory`, `Store`, `Session`, guards, providers, `SessionRegistry` |
| Extensions | `lazybridge.ext.*` | `hil`, `approval`, `delegation`, `planners`, `knowledge`, `evals`, `otel`, `viz` |
| Integration | LazyTools (`lazytoolkit`, `import lazytools`) | MCP/connectors, code support, code bridge, projects, quota routing, document readers |

Rules enforced by tests: core never imports `ext`; ext imports only the
public `lazybridge` surface; ext-engine factories are module functions (no
`Agent.from_human`). Maturity: core Stable; `ext.hil` and `NativeTool`
Alpha; `ext.evals` / `ext.viz` Experimental.

## Human in the loop — `lazybridge.ext.hil`

```python
from lazybridge import Agent
from lazybridge.ext.hil import HumanEngine, SupervisorEngine

approve = Agent(engine=HumanEngine(timeout=300, default="reject"), name="approve")
repl = Agent(engine=SupervisorEngine(tools=[...], agents=[...]), name="repl")
```

`HumanEngine(timeout=, ui="terminal" | "web" | custom, default=)`; with
`output=Model` the terminal prompts each field. `SupervisorEngine(tools=,
agents=, store=, input_fn=, ainput_fn=, timeout=, default=)`.
`human_agent(...)` / `supervisor_agent(...)` are kwarg-splitting factories.
Both return plain `Agent`s usable as `Step` targets. Give gates a safe
`default=` for timeouts.

## Approval — `lazybridge.ext.approval`

- **`TieredGate(channel, rules)`** implements `ApprovalGate` with ordered
  `Rule(tier, name_pattern, arg_pattern=None, kind_pattern="*",
  provider_pattern="*")`; tiers `allow` / `session` / `ask` / `deny`;
  unmatched → deny. Session grants are scoped to (provider, kind, name, cwd,
  policy fingerprint); every decision lands in `gate.log` as `AuditRecord`
  (secrets redacted; long values elided keeping head and tail). Codex
  command approvals arrive with the stable name `codex-shell`; match the
  command with `arg_pattern` on `arguments["command"]`.
  `TerminalChannel()` for local use.
- **`ApprovalQueue(store, prefix="approval:")`** durable tickets:
  `create_ticket`, `approve_ticket`, `reject_ticket`, `expire_ticket`,
  `list_pending_tickets`, `claim_earlier_approval`, `mark_approval_consumed`,
  `record_denial`, `record_notify_failure`. **`StoreApprovalChannel(queue,
  task_id=, ttl=, wait_budget=, notify=, operator_only=)`** turns it into a
  `Channel`: `ttl` = how long a ticket stays answerable, `wait_budget` = how
  long one call blocks; a late approval is consumed exactly once by a later
  identical request. Wire `TieredGate(..., on_record=queue.record_denial)`.

## Background delegation — `lazybridge.ext.delegation`

For long-running hosts (an agent that stays up) that hand work to Codex or
Claude Code without blocking their own turn.

- `JobRegistry(store)` — durable job records (`write`, `update`, `find`,
  `begin_execution`, `reclaim_interrupted()` at startup, `checker_tool()`,
  `result_tool()`). Records carry engine/model/effort, cost or
  `cost_unknown`, owner pid/boot id.
- Tool factories: `make_codex_writer`, `make_claude_writer`,
  `make_background_delegate`, `make_parallel_delegate`, `make_plan_delegate`
  (linked to a `DurableBlackboard`), `make_persistent_consultant`; engine
  factories `make_codex_writer_engine_factory`,
  `make_claude_writer_engine_factory`, `make_claude_delegate_engine_factory`.
- Optional hooks (all default to old behaviour): `validate_model`,
  `admission_gate` (decision with `allowed`, optional `release()` /
  `refund()`), `pre_confirm`, `guard`, `extra_params={...: ExtraParam(...)}`,
  `accept_model_override` / `accept_effort_override` /
  `accept_session_override` / `accept_session_alias_override`,
  `session_alias` + `session_registry`, `writable_roots` (Codex),
  `max_turns` (Claude, default 60), `job_runner=JobRunner(prepare, register,
  execute, finalize, rollback)` with a `JobContext`, `structured_outcomes`,
  `stop_on_refusal`, `rollback_claim`.
- The generated tools must be awaited via `Tool.run` on the host's
  persistent event loop; `Tool.run_sync` cancels the background task.
- Public scheduling helpers: `track_background_task`,
  `schedule_with_admission_release`, `release_admission`, `refund_admission`,
  `rejection_text`.

## Planners — `lazybridge.ext.planners`

- `orchestrator_agent(agents, model=...)` / `blackboard_orchestrator_agent`
  (aliases `make_planner` / `make_blackboard_planner`), `PlanSpec` /
  `StepSpec`, `make_plan_builder_tools`, `make_execute_plan_tool`.
- **`DurableBlackboard(store, plan_id, lease_seconds=900, max_attempts=3)`**
  — a to-do list that survives crashes and processes: `set_plan`,
  `add_tasks`, `claim_next(owner)` (CAS, one winner), `claim_task(index,
  expected_text, owner=, renew=)`, `mark_done`, `mark_failed`,
  `cancel_task` (also for tasks parked `failed` after exhausting attempts,
  1.8.1), `set_task_schedule(planned_start_at=, due_at=, reason=)`,
  `snapshot()`, `render()` (marks `OVERDUE`). Leases return crashed work to
  the queue; `max_attempts` parks poison tasks. `durable_blackboard_agent(
  agents, store=, plan_id=, ...)` wraps it as an agent.

## Knowledge — `lazybridge.ext.knowledge`

`DurableKnowledgeBase(store, prefix="knowledge:")`: `save_lesson(topic=,
what_worked=, gotchas=)` (create-only), `search_lessons` / `find_lessons`,
`revise_lesson(slug=, expected_revision=, ...)` (optimistic CAS, also for
retraction). Lessons are retrospective notes, not memory or task state.

## Evals, OpenTelemetry, visualizer

- `lazybridge.ext.evals`: see `testing.md`.
- `lazybridge.ext.otel.OTelExporter` (extra `lazybridge[otel]`): add to
  `Session(exporters=[...])`.
- `lazybridge.ext.viz.Visualizer(session, store=None, host="127.0.0.1",
  port=0, auto_open=True)`: live local view of a session's graph and events.

## Control plane — `lazybridge.control_plane`

`ControlStore(path, lease_seconds=900)`: SQLite store for a fleet's work on
one machine. `create_project`, `assign(project_id, actor_id)`,
`enqueue(project_id, payload)`, `claim(owner)` → `ClaimedItem` with a rising
fence token, `finish(item_id, fence=, status=)` (stale fence →
`FenceRejected`), visibility scoped by `actor_id` (`ScopeDenied`),
append-only ledger checked by `verify_ledger()` → `LedgerProblem`s.
Qualify storage with `python -m lazybridge.control_plane.probe`.
