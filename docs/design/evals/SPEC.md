# Benchmarking and Evaluations Specification

**Status:** Proposed
**Decision date:** 2026-11-18
**Scope:** the reusable `lazybridge.ext.evals` runner and the first three benchmarking experiments

## 1. Goal

Build a public, engine-agnostic evaluation runner that can execute repeatable trials, grade their artifacts, preserve transcripts and telemetry links, enforce hard resource limits, and compare systems with statistically honest reports. The runner extends the existing lightweight evaluation helpers; it does not turn production verification into an eval-only API.

The working model is **task → trial → transcript/artifacts → grader results → report**. A task describes one problem and its grader. A trial is one independently seeded attempt by one system. Graders assess immutable trial output. Reports aggregate quality, cost, latency, and approval behavior. Repeated trials distinguish the probability that at least one attempt succeeds (`pass@k`) from the probability that every attempt succeeds (`pass^k`). Held-out tasks and grader calibration reduce overfitting. Grader preference is deterministic code first, calibrated LLM judgment second, and sampled human review where automation is inadequate. Comparisons are paired on the same tasks, not inferred from unrelated aggregate scores.

## 2. Decisions

### D1 — Ownership and product boundaries

The runner lives in LazyBridge as the public, reusable `lazybridge.ext.evals` package. Task sets and experiment results owned by the operator live in a separate **private task-set repo**. A results view in LazyStudio is a separate task in a different repository and is not part of this specification.

### D2 — Experiment order

Experiments are run in this order:

1. **E1:** cross-vendor tandem review with seeded defects on approximately 50 real pull requests; arms are no review, same-vendor review, and cross-vendor review; three trials per task/arm.
2. **E2:** engine portability on approximately 20 coding tasks with executable tests across at least four engines, including coding-agent engines where configured.
3. **E3:** approval-gate statistics computed from existing bridge job data.

Committee pipelines are deferred until these experiments establish a baseline.

### D3 — Budget policy

Each experiment may consume at most 10% of each participating engine's weekly quota. Before execution, the runner produces a dry-run cost estimate. During execution it enforces configured hard caps for tokens, known USD cost, and run count, and stops cleanly before knowingly starting work that would breach a cap. Unknown prices or usage remain explicitly unknown and never count as zero.

## 3. Invariants

- **I1 — Compatibility:** the current public API (`EvalCase`, `EvalResult`, `EvalReport`, `EvalSuite`, `exact_match`, `contains`, `not_contains`, `min_length`, `max_length`, and `llm_judge`) keeps its import paths, signatures, and behavior. New runner APIs are additive.
- **I2 — Engine neutrality:** orchestration targets the LazyBridge engine/agent protocol, not provider-specific SDK objects. `Agent(verify=...)`, `Plan`/`Step`, ordinary engines, and coding-agent engines can participate through adapters without special cases in the trial state machine.
- **I3 — Reproducibility:** stored configuration, immutable task-set identity, code revision, grader revision, and seeds reproduce a run where providers honor seeds. When they do not, the report records that limitation and all observable provider/model parameters.
- **I4 — Offline unit tests:** unit tests use fakes and fixtures and make no network calls. Live/provider tests are separately marked and opt-in.
- **I5 — Evidence preservation:** raw output, failure state, usage, timestamps, and grader attempts are append-only evidence. Retries never erase or relabel an earlier failure.
- **I6 — Held-out integrity:** held-out labels and ground truth are unavailable to the system under test during generation.

## 4. Requirements

### Task sets

**R1 — Versioned manifest and identity.** A task set shall have a schema version, stable task-set ID and version, content digest, split policy, and creation metadata. Each task shall have a stable ID, prompt/input reference, split (`development`, `calibration`, or `held_out`), grader specification, and metadata. Metadata shall support source revision, tags, difficulty, language, license/handling classification, and experiment-specific fields. Unknown fields shall be rejected in strict mode and preserved only through a versioned extension namespace. Inputs may be inline or content-addressed relative artifacts; manifests shall not require machine-local absolute paths.

**R2 — Held-out boundary.** When a run selects `held_out`, the runner shall expose only task inputs and permitted fixtures to the system under test, keep expected answers/defect ground truth in the grader boundary, record the selected task IDs and manifest digest, and reject overlapping stable task IDs across development/calibration and held-out splits unless an explicit, audited migration declares the replacement.

**R3 — Per-task graders.** Every executable task shall name one primary grader and may name secondary diagnostic graders. A grader spec shall include type, version, configuration digest, artifact inputs, timeout, and failure policy. Missing or incompatible grader versions shall fail validation before provider work starts.

### Trial runner and persistence

**R4 — Trial matrix and deterministic scheduling.** The runner shall execute configurable `N >= 1` trials for every selected task/system pair, derive and persist a distinct seed for each trial, bound concurrency globally and optionally per engine, and produce stable trial IDs. If a provider cannot accept a seed, the runner shall mark seed support false rather than claim deterministic replay.

**R5 — Timeout, retry, and failure semantics.** Each generation and grading attempt shall have explicit timeouts. Retries shall be limited, use a recorded policy, and create linked attempt records. Provider errors, timeouts, invalid output, and grader errors remain distinct terminal or intermediate statuses. A successful retry may make a trial gradable but shall not hide the earlier attempt failure; reports shall expose retry/error rates and the declared scoring policy.

**R6 — Hard budgets and dry run.** Given caps for total runs, input/output/total tokens, known USD cost, and per-engine weekly quota allocation, the runner shall validate the experiment's 10% allocation and emit a dry-run estimate with assumptions and unknowns. Before dispatch and after every usage update it shall atomically reserve/reconcile budget. On a reached or conservatively projected cap it shall dispatch no new trial, allow configured in-flight work to finish or cancel safely, flush state, and end with `budget_exhausted` rather than corrupting or silently truncating the run. At least one finite hard cap is required for a live run.

**R7 — Resumable durable run.** Run configuration and state shall be stored on disk using SQLite plus content-addressed artifacts. Transactions shall make completed attempts idempotent. Restarting the same run shall verify config/task-set digests, skip committed work, reconcile orphaned in-flight attempts as interrupted, preserve budget usage, and continue pending work. A changed immutable configuration shall require a new run ID or explicit fork; it shall never mutate prior evidence.

### Grading

**R8 — Deterministic code graders.** The runner shall support command/test graders with isolated working directories, explicit command/exit-code contract, timeout, captured stdout/stderr, fixture digest, and optional structured subtest results. Code graders rank above subjective graders for a task's primary pass/fail when an executable oracle exists. Unit-test fixtures shall use sandbox fakes; production isolation mechanics are an implementation decision documented by the adapter.

**R9 — Calibrated LLM judge.** An LLM judge shall use a versioned rubric and structured verdict, be evaluated on a labeled calibration split before use, and report sample size, confusion matrix, accuracy, balanced accuracy, and agreement (Cohen's kappa; additionally weighted kappa for ordinal labels). The run config shall define minimum agreement and fail-closed behavior. Below threshold, malformed verdicts, judge transport errors, or rubric drift shall prevent automatic passing and route the item to the human queue. This capability extends, and does not change, the legacy `llm_judge` helper.

**R10 — Human sample queue.** The runner shall export a blinded queue containing a configurable random sample, all unresolved grader disagreements, and threshold failures. It shall support at least two independent labels, optional adjudication, randomized system ordering, label provenance, timestamps, and inter-rater agreement. Human labels are imported as append-only grader records; the runner shall reject task/system identities leaked into a blinded payload.

### Metrics and comparison

**R11 — Success metrics.** Reports shall include per-task successes/attempted trials and success rate; macro and micro summaries; and, for `n` observed trials containing `c` successes, the unbiased estimator `pass@k = 1 - C(n-c,k)/C(n,k)` for valid `1 <= k <= n` (equal to 1 when `n-c < k`). Reports shall include `pass^k`, defined as the fraction of tasks whose selected `k` trials all pass, alongside its selection rule; it shall never be labeled `pass@k`. Missing, errored, interrupted, and ungraded trials shall be counted and displayed separately under a declared denominator policy.

**R12 — Uncertainty.** Per-task Bernoulli success rates shall include Wilson confidence intervals, including valid bounds for all-pass and all-fail observations. Aggregate and difference estimates shall include a seeded paired/task-level bootstrap confidence interval with configured confidence level and resample count. Reports shall state unit of resampling, seed, method, denominators, and suppress intervals with an explanatory reason when the estimand is undefined.

**R13 — Paired system comparison.** Comparing systems A and B shall align identical stable task IDs and a declared within-task trial aggregation rule, report per-task wins/ties/losses and excluded/missing pairs, run an exact two-sided sign test on non-ties, and report a seeded paired task bootstrap interval for the difference. It shall report the minimum detectable difference (MDD) for configured alpha and power, the assumed baseline/variance or simulation method, and achieved sample size. Tie-heavy comparisons shall not manufacture significance; zero non-ties shall yield an undefined sign-test p-value with an explanation.

### Accounting, evidence, and outputs

**R14 — Cost, latency, and approvals.** Every attempt shall record engine/provider/model identifiers, input/output/cached/reasoning tokens when exposed, known USD cost and pricing source/version, queue/generation/grading/total wall time, retry count, and approval requests/approved/denied/timed-out counts. Missing provider fields shall be `unknown`, not zero. Reports shall aggregate totals and distributions by experiment, system, engine, task, and status without double-counting retries.

**R15 — Transcripts and OpenTelemetry.** Each attempt shall store or content-address its transcript and artifact manifest and link them to OpenTelemetry trace/span IDs when tracing is enabled. Existing exporters remain pluggable. Secrets and configured sensitive fields shall be redacted before persistence/export, with redaction policy/version recorded. A missing exporter or trace shall not fail a trial; the report shall show linkage coverage.

**R16 — Reports and provenance.** A completed, stopped, or partially resumed run shall emit a schema-versioned canonical JSON report and a human-readable Markdown report derived from it. Both shall identify run status, configuration and task-set digests, source revision, timestamps, systems/engines, seeds and seed-support flags, grader versions/calibration, budget estimate/actuals, metric definitions, exclusions, warnings, and artifact/trace references. JSON serialization shall be deterministic apart from documented volatile fields.

### Seeded defects and experiments

**R17 — Seeded-defect tooling.** Tooling for E1 shall provide a versioned defect catalogue with at least `bug`, `security`, `weakened_test`, and `leftover_file`. An injection recipe shall identify an eligible clean source revision, apply exactly one or a declared composition of deterministic mutations, verify applicability, and produce a patch plus machine-readable ground truth (defect ID/type, touched locations, detection rubric, expected clean behavior, injector version, source/patch digests). Generation sees the realistic review input but not hidden ground truth. A revert/clean-room check shall prove the fixture can be reconstructed and shall reject accidental pre-existing or non-applying defects.

**R18 — Experiment definitions.** Experiment manifests shall encode the following preregistered designs and analysis rules:

- **E1 — Tandem review:** approximately 50 eligible real pull requests, balanced/stratified seeded-defect types, arms `no_review`, `same_vendor_review`, and `cross_vendor_review`, three trials per task/arm, randomized/blinded order, primary outcome defect detection verified against hidden ground truth, secondary false-positive, repair, cost, latency, and approval outcomes. Comparisons are paired by pull request; vendor/model versions and reviewer relationship are frozen in the manifest.
- **E2 — Engine portability:** approximately 20 held-out coding tasks with deterministic tests, at least four engines (including configured coding-agent engines), common task contract and budgets, repeated trials, primary outcome test pass, and secondary setup failure, portability exclusion, cost, latency, retry, and approval outcomes. Engine-specific unavoidable capabilities are declared before running, never silently patched into results.
- **E3 — Approval gates:** a read-only, versioned extract from existing bridge job data; documented inclusion window, schema and deduplication; counts/rates of requested, approved, denied, timed-out, and bypassed gates by eligible dimensions; latency and downstream outcome where available; missingness, repeated jobs, and selection bias explicitly reported. No provider generation is required, but extract/query version and privacy/redaction checks are mandatory.

E1 precedes E2, which precedes E3. A later experiment may be prepared earlier but may not publish confirmatory conclusions out of order without a recorded amendment.

## 5. Acceptance examples

1. **Legacy compatibility** — Given an application importing and running the existing `EvalSuite` and assertion helpers, when it upgrades to the runner release, then imports, call signatures, synchronous/asynchronous behavior, and result counts remain unchanged.
2. **Held-out protection** — Given a held-out seeded-defect task, when an engine receives its trial input, then the payload contains the review material but not split labels, expected verdicts, mutation metadata, or ground truth.
3. **Flaky grader** — Given a code grader that times out once and passes on its allowed retry, when grading completes, then both attempts are stored, the trial reflects the configured scoring policy, and retry/flakiness counts appear in JSON and Markdown.
4. **Provider error mid-run** — Given a provider error after some trials commit, when the process restarts, then committed trials are not repeated, the failed attempt remains visible, orphaned work is marked interrupted, and only eligible pending/retry work runs.
5. **Budget hit** — Given a five-run cap and concurrency two, when reconciliation reaches five consumed reservations, then no sixth trial is dispatched, state is flushed, in-flight handling follows configuration, and status is `budget_exhausted`.
6. **Unknown price** — Given an engine without known token pricing, when dry-run estimation executes, then cost is `unknown` with an assumption warning rather than `$0`; the live run still requires another enforceable finite cap.
7. **All pass/all fail** — Given all-pass and all-fail tasks, when Wilson intervals and `pass@k` are computed, then bounds are finite and within `[0,1]`, and combinatorial edge cases return the defined values without division errors.
8. **Insufficient trials** — Given `k > n`, when metrics are requested, then validation rejects that metric request and the report does not extrapolate `pass@k` or `pass^k`.
9. **Tie-heavy comparison** — Given paired outcomes with many ties and zero non-ties, when comparing systems, then wins/ties/losses are retained, the exact sign-test p-value is marked undefined, and the MDD warns that the effective sample is insufficient.
10. **Judge below calibration** — Given an LLM judge whose kappa is below the configured threshold, when it grades a trial, then it cannot auto-pass the trial and the item enters the blinded human queue.
11. **Trace exporter unavailable** — Given tracing enabled and an exporter failure, when a trial completes, then grading and persistence succeed, trace linkage is marked missing, and the export error is recorded without transcript loss.
12. **Resume mismatch** — Given a stored run and a changed task-set digest, when resume is requested, then the runner refuses mutation and instructs the caller to create a forked run.

## 6. Out of scope

- Hosting task sets or owned result data in this public repository.
- The LazyStudio results UI or its storage/API contract beyond consuming the canonical JSON report.
- Committee/multi-reviewer pipelines, adaptive routing, leaderboards, model training, and automatic prompt optimization.
- A universal secure sandbox; adapters document their isolation boundary.
- Purchasing/provider quota management or inferring undisclosed provider prices.
- Replacing `Agent(verify=...)` or `lazybridge._verify`; verification may be measured by experiments but remains runtime control-plane behavior.
- Claiming exact reproducibility from providers that do not honor seeds or deterministic tool environments.

## 7. Assumption log

| ID | Assumption | Consequence / validation |
|---|---|---|
| A1 | Engine adapters can expose a common invocation result while retaining provider-specific usage metadata. | Prototype with one standard engine and both coding-agent engine families before freezing the result schema. |
| A2 | Task artifacts fit on local disk; SQLite stores indexes/state, not large blobs. | Load-test artifact manifests and document retention/export hooks. |
| A3 | Weekly quotas are supplied by configuration or a quota adapter; they are not universally discoverable. | Dry run rejects an experiment whose 10% policy cannot be evaluated unless an explicit conservative cap is supplied. |
| A4 | Provider token counts and USD prices may arrive late or remain unavailable. | Use atomic reservations and conservative estimates; preserve `unknown` fields. |
| A5 | Three E1 trials support variance estimation but may be underpowered for small effects. | Always publish confidence intervals and MDD, not only point estimates or p-values. |
| A6 | Real pull requests can be legally and safely transformed into evaluation fixtures in the private task-set repo. | Record handling/license metadata and exclude ineligible sources before injection. |
| A7 | Existing bridge job data has stable identifiers sufficient for E3 deduplication. | Run a schema/missingness audit before preregistering final E3 denominators. |
| A8 | OTel trace IDs can be captured from existing instrumentation without coupling the runner to one exporter. | Contract-test tracing enabled, disabled, and exporter-failure paths. |
| A9 | Human reviewers can be blinded to system identity for a meaningful sample. | Queue export validation rejects identity-bearing fields and randomizes order. |
| A10 | Statistical defaults will evolve. | Store method names, parameters, library/source revision, and seeds in every report. |

## 8. Validation matrix

Every requirement has an executable check; task IDs refer to the implementation DAG in `TASKS.md`.

| Requirement | Executable check | Planned task |
|---|---|---|
| R1 | `pytest -q tests/ext/evals/test_manifest.py` validates versions, digests, metadata, extensions, and relative artifacts. | T1 |
| R2 | `pytest -q tests/ext/evals/test_splits.py` asserts non-overlap and snapshots an engine payload without held-out answers. | T1 |
| R3 | `pytest -q tests/ext/evals/test_manifest.py -k grader` rejects absent/incompatible graders before fake-engine calls. | T1 |
| R4 | `pytest -q tests/ext/evals/test_runner.py -k 'matrix or concurrency or seed'` checks trial IDs, limits, and seed flags. | T3 |
| R5 | `pytest -q tests/ext/evals/test_runner_failures.py` checks attempt chains and distinct timeout/error states. | T3 |
| R6 | `pytest -q tests/ext/evals/test_budget.py` checks dry-run unknowns, quota percentage, atomic caps, and clean stop. | T2 |
| R7 | `pytest -q tests/ext/evals/test_store.py tests/ext/evals/test_resume.py` kills/restarts a fake run and checks idempotence/digest mismatch. | T2, T3 |
| R8 | `pytest -q tests/ext/evals/graders/test_code.py` runs offline fixture commands and captures structured evidence. | T4 |
| R9 | `pytest -q tests/ext/evals/graders/test_llm.py` uses a fake judge to check calibration metrics, thresholds, and fail-closed routing. | T4 |
| R10 | `pytest -q tests/ext/evals/test_human_queue.py` checks sampling, blinding, two labels, adjudication, and append-only import. | T5 |
| R11 | `pytest -q tests/ext/evals/test_metrics.py -k 'pass_at_k or pass_power_k or success'` checks formulas against enumerated small cases. | T6 |
| R12 | `pytest -q tests/ext/evals/test_metrics.py -k 'wilson or bootstrap'` checks seeded snapshots and all-pass/all-fail bounds. | T6 |
| R13 | `pytest -q tests/ext/evals/test_compare.py` checks pairing, exact sign tests, paired bootstrap, MDD, missing pairs, and zero non-ties. | T6 |
| R14 | `pytest -q tests/ext/evals/test_accounting.py` checks unknown fields, approval states, retry accounting, and aggregation totals. | T3 |
| R15 | `pytest -q tests/ext/evals/test_transcripts.py tests/ext/evals/test_otel.py` checks redaction, content links, and exporter failure. | T7 |
| R16 | `pytest -q tests/ext/evals/test_reports.py` validates JSON schema, deterministic serialization, Markdown derivation, provenance, and partial runs. | T7 |
| R17 | `pytest -q tests/ext/evals/test_seeded_defects.py` reconstructs/reverts all four catalogue types and verifies hidden ground truth. | T8 |
| R18 | `pytest -q tests/ext/evals/test_experiment_manifests.py` validates E1–E3 fixtures, ordering, arms/trials/engines, and analysis declarations. | T9 |

A final offline gate runs `pytest -q tests/ext/evals tests/test_evals.py` (using the actual legacy test path if it differs), `ruff check lazybridge/ext/evals tests/ext/evals`, and `python -m build`. Live smoke tests are opt-in and must use synthetic/non-held-out tasks.
