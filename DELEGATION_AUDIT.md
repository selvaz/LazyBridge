# Delegation audit and remediation

The audit covers LazyBridge 1.8.10 at [`43aba96`](https://github.com/selvaz/LazyBridge/commit/43aba96a6a673f2e0f5a8de8a9ab71284824ba23). It covers agent tools, background jobs, plan claims and replay, writer approvals, and cross-agent review. The baseline unit suite passed 3,366 tests on Python 3.12 with 21 skipped. Targeted checks used mock workers and SQLite. Approval findings were reviewed against the code and runtime contracts.

Severity totals at baseline are **0 confirmed critical, 7 high, and 6 medium**. Keep the `Agent` and `Plan` composition APIs. Prioritize approval scope and attempt ownership before relying on unattended writers in shared repositories.

## High

### A1 — Keep native permission grants within scope

**Fixed on this branch.** The session cache represented an approval by action and name, which could discard the scope returned by the approval gate. Permission requests now reach the gate so the adapter can return its exact decision and the native runtime can apply the session scope. Regression coverage checks that a prior session grant leaves subsequent permission decisions with the gate. [Approval cache](lazybridge/engines/coding.py) · [Codex adapter](lazybridge/engines/codex/app_server.py)

### A2 — Bind writer confirmation to the effective delegation

**Open.** The confirmation and durable approval reuse depend on the rendered request. The effective destination and session can differ between delegations that appear identical to the approver. Freeze the resolved delegation before asking, show the target and session, and bind the approval to a digest of the full request and its scope. Require a fresh decision if the target, options, or objective changes. [Writer confirmation](lazybridge/ext/delegation/writers.py) · [Approval queue](lazybridge/ext/approval/queue.py)

### A3 — Propagate a denied or failed result as an error

**Tool boundary fixed on this branch.** Codex, Claude Code, and `LLMEngine` now report failed child envelopes as tool errors. Parent tool events use the same error representation. Output guard denials remove the draft from the returned envelope, and the adapters give the parent a generic guard error. Follow-up work should apply the same disclosure policy to child-engine traces emitted before the guard decision. Tests cover provider error flags, parent events, and guard results. [Agent guard](lazybridge/agent.py) · [LLMEngine](lazybridge/engines/llm.py) · [Codex tools](lazybridge/engines/codex/dynamic_tools.py) · [Claude Code adapter](lazybridge/engines/claude_code/mcp_adapter.py)

### A4 — Prevent takeover of a live Plan checkpoint

**Open.** Two runs using `resume=True` can execute the same step when a second run takes over an active checkpoint. The first run's compare-and-swap failure occurs after the step may have performed side effects. Persist an owner and renewable lease, reconcile ambiguous recovery states, and use attempt tokens or idempotency keys at effect boundaries. With two SQLite processes, verify that a live run retains ownership and a dead run can be resumed. [Checkpoint claim](lazybridge/engines/plan/_checkpoint.py)

### A5 — Keep a blackboard claim valid for the worker's lifetime

**Open.** A live worker can outlast a `DurableBlackboard` claim, allowing a second worker to start on the same task. The default worker can also leave a claim active after a factory failure. Link the job to a unique claim attempt, renew its lease, fence rollback and result publication, and release claims when setup fails. Validate lease expiry, stale completion, and pre-start failure with independently controlled workers. [Plan delegation](lazybridge/ext/delegation/background.py) · [DurableBlackboard](lazybridge/ext/planners/durable_blackboard.py)

### A6 — Preserve live jobs during startup recovery

**Open.** Standard delegate records lack a process owner. On a shared Store, startup recovery can mark a live job interrupted and prevent its runner from executing. Stamp ownership at enqueue time and preserve it through conditional transitions. Reconcile legacy records without owners explicitly. Verify a rolling restart with a live worker and a terminated worker over the same SQLite database. [Job creation](lazybridge/ext/delegation/background.py) · [Job recovery](lazybridge/ext/delegation/jobs.py)

### A7 — Stop automatic replay after an uncertain outcome

**Agent replay fixed on this branch.** `CodexTurnUncertain` now sets `ErrorInfo.outcome_uncertain` on the returned envelope. `Agent.run()`, `Agent.stream()`, and verified agent tools preserve an uncertain outcome without calling a fallback or starting another execution. Targeted tests exercise writer and reviewer failures. Checkpoint recovery and external retry callers should honor the same signal as part of A4–A6. [Error envelope](lazybridge/envelope.py) · [Agent](lazybridge/agent.py) · [Verification](lazybridge/_verify.py)

## Medium

| ID | Finding | Resolution and regression criterion |
| --- | --- | --- |
| M1 | Named Plan steps pass only the task string to registered agents, dropping the composed context and attachments. | Deliver the full envelope to agent targets. Cover direct, named, and aliased steps with the same context and attachments. [Plan dispatch](lazybridge/engines/plan/_plan.py) |
| M2 | Delegate `ExtraParam` requirements and types appear in the advertised schema. Runtime dispatch accepts missing or mistyped values. | Validate arguments before policy checks, approval, or job creation. Invalid calls should leave the registry empty. [Delegate schema and dispatch](lazybridge/ext/delegation/background.py) |
| M3 | Cancellation during parallel scheduling cleanup can strand reservations for work that never starts. | Track each reservation through handoff and settle unhanded grants with cancellation-safe, idempotent cleanup. Test cancellation during a refund after a registry failure. [Batch scheduling](lazybridge/ext/delegation/background.py) |
| M4 | A detached Codex child can outlive its inherited concurrency slot, leaving it uncounted as other turns start. | Count detached jobs independently. Test the limit with detached and inline children, including a cap of one. [Codex slots](lazybridge/engines/codex/app_server.py) |
| M5 | A verified tool can read live sources twice and show the reviewer older evidence than the writer. | Share an evidence snapshot per attempt. Test a changing source and compare what the writer and reviewer receive. [Verified agent tool](lazybridge/agent.py) |
| M6 | Aliased agent tools can use a session inherited from another orchestrator. | Resolve implicit session identity per invocation. Verify event placement with two orchestrators sharing an aliased child. [Session inheritance](lazybridge/agent.py) |

## Delivery order

1. Finish the approval and result boundaries in A2 and A3. Verify that each approval is scoped to the resolved request and guard denials stay within the intended reporting channels.
2. Align checkpoint, claim, and job ownership across A4–A6. Exercise takeover, stale completion, cancellation, and restart against a shared SQLite database.
3. Add the runtime regressions in M1–M6 through public entry points, including the common cases of a named Plan target and an aliased agent tool.
