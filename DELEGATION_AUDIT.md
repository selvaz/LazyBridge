# Delegation audit and remediation

The audit covers LazyBridge 1.8.10 at [`43aba96`](https://github.com/selvaz/LazyBridge/commit/43aba96a6a673f2e0f5a8de8a9ab71284824ba23). It covers agent tools, background jobs, plan claims and replay, writer approvals, and cross-agent review. The baseline unit suite passed 3,366 tests on Python 3.12 with 21 skipped. Targeted checks used mock workers and SQLite. Approval findings were reviewed against the code and runtime contracts.

Severity totals at baseline are **0 confirmed critical, 7 high, and 6 medium**. Keep the `Agent` and `Plan` composition APIs. Prioritize approval scope and attempt ownership before relying on unattended writers in shared repositories.

## High

### A1 — Keep native permission grants within scope

**Fixed on this branch.** The session cache represented an approval by action and name, which could discard the scope returned by the approval gate. Permission requests now reach the gate so the adapter can return its exact decision and the native runtime can apply the session scope. Regression coverage checks that a prior session grant leaves subsequent permission decisions with the gate. [Approval cache](lazybridge/engines/coding.py) · [Codex adapter](lazybridge/engines/codex/app_server.py)

### A2 — Bind writer confirmation to the effective delegation

**Fixed for the built-in Codex writer.** Confirmations show the resolved workspace, session, and extra writable roots. The digest covers the full objective and call options, so durable approval reuse binds to the request even when the displayed objective is shortened. Execution checks the approved alias binding before launching Codex. Custom factories remain responsible for settings they derive outside the supplied options. [Writer confirmation](lazybridge/ext/delegation/writers.py) · [Approval queue](lazybridge/ext/approval/queue.py)

### A3 — Propagate a denied or failed result as an error

**Fixed for tool results, Session traces, and token streams.** Codex, Claude Code, `LLMEngine`, and Plan report failed child envelopes as tool errors. Output guard denials omit the draft and surface a generic guard error to the parent. Session filters guarded traces before custom redactors, persistence, and exporters. Guarded calls suppress ambient tokens and explicit guarded streams hold their chunks until review, including on fallback. A regression pauses a guarded writer inside a streaming Plan and checks that the draft stays private throughout. [Agent guard](lazybridge/agent.py) · [Session](lazybridge/session.py) · [Plan](lazybridge/engines/plan/_plan.py)

### A4 — Prevent takeover of a live Plan checkpoint

**Fixed for recovery on one host.** Plan checkpoints carry an owner stamp and run ID. Acquisition checks that ownership before retrying an active checkpoint and assigns an active claim before replaying a paused or failed step. The SQLite regression covers recovery while a second process lives and after it exits. Active legacy checkpoints require explicit reconciliation. External side effects still need idempotency keys or an equivalent replay policy. [Checkpoint claim](lazybridge/engines/plan/_checkpoint.py)

### A5 — Keep a blackboard claim valid for the worker's lifetime

**Fixed for delegation through `DurableBlackboard`.** The board stamps each claim with the job ID and its lease deadline. The runner renews that attempt during execution and validates it before accepting the result for review. Default setup rollback targets the same attempt. Shared-SQLite tests cover renewal across expiry intervals, cancellation on lease loss, and stale completion and rollback after reassignment to the same owner. Custom hooks that mutate board state must compare the job ID in their own CAS. [Plan delegation](lazybridge/ext/delegation/background.py) · [DurableBlackboard](lazybridge/ext/planners/durable_blackboard.py)

### A6 — Preserve live jobs during startup recovery

**Fixed for standard delegates.** Job records carry the worker process owner from enqueue time. Startup recovery consults liveness and uses compare-and-swap to mark abandoned jobs interrupted. Unknown legacy ownership requires an explicit `reclaim_unowned=True` reconciliation. The shared SQLite regression verifies both live and exited process owners. [Job creation](lazybridge/ext/delegation/background.py) · [Job recovery](lazybridge/ext/delegation/jobs.py)

### A7 — Stop automatic replay after an uncertain outcome

**Agent and Plan replay fixed on this branch.** `CodexTurnUncertain` now sets `ErrorInfo.outcome_uncertain` on the returned envelope. `Agent.run()`, `Agent.stream()`, and verified agent tools preserve an uncertain outcome without calling a fallback or starting another execution. Plan checkpoints hold uncertain steps and parallel bands for reconciliation before replay. External retry callers should honor the same signal. [Error envelope](lazybridge/envelope.py) · [Agent](lazybridge/agent.py) · [Checkpoint claim](lazybridge/engines/plan/_checkpoint.py)

## Medium

| ID | Finding | Resolution and regression criterion |
| --- | --- | --- |
| M1 | Named Plan steps pass only the task string to registered agents, dropping the composed context and attachments. | Deliver the full envelope to agent targets. Cover direct, named, and aliased steps with the same context and attachments. [Plan dispatch](lazybridge/engines/plan/_plan.py) |
| M2 | Delegate `ExtraParam` requirements and types appear in the advertised schema. Runtime dispatch accepts missing or mistyped values. | Validate arguments before policy checks, approval, or job creation. Invalid calls should leave the registry empty. [Delegate schema and dispatch](lazybridge/ext/delegation/background.py) |
| M3 | Cancellation during parallel scheduling cleanup can strand reservations for work that never starts. | **Fixed.** Ownership transfers on scheduling. The batch finishes its remaining refunds once even if cancelled repeatedly during cleanup. Tests inject registry and scheduler failures under both worker lifecycles. [Batch scheduling](lazybridge/ext/delegation/background.py) |
| M4 | A detached Codex child can outlive its inherited concurrency slot, leaving it uncounted as other turns start. | **Fixed for detached delegation.** Workers scheduled by the delegation helpers acquire independent slots. Tests use caps of one and two and verify inline re-entry and custom scheduling. [Codex slots](lazybridge/engines/codex/app_server.py) |
| M5 | A verified tool can read live sources twice and show the reviewer older evidence than the writer. | Share an evidence snapshot per attempt. Test a changing source and compare what the writer and reviewer receive. [Verified agent tool](lazybridge/agent.py) |
| M6 | Aliased agent tools can use a session inherited from another orchestrator. | Resolve implicit session identity per invocation. Verify event placement with two orchestrators sharing an aliased child. [Session inheritance](lazybridge/agent.py) |

## Delivery order

1. Address M1 and M6 through named Plan targets and aliased agent tools. Exercise context and attachments, and run two orchestrators with the same child.
2. Address M2 and M5 through delegate argument validation and a shared evidence snapshot for the writer and reviewer.
