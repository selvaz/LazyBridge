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

### D4 — E1 cost/quality hypothesis (owner, 2026-11-18)

E1 tests two preregistered hypotheses. **H1:** cross-vendor review provides its largest benefit with cheap writers; frontier writers avoid more defects before review, so review's marginal quality gain shrinks. **H2:** a cheap writer paired with a cross-vendor cheap or mid-tier reviewer can match the quality of a frontier writer alone at lower cost. E1 therefore seeks the break-even region rather than assuming that review is always beneficial: it must identify when a tandem saves money and must show configurations whose added review cost produces no quality gain as dominated.

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

**R6 — Hard budgets and dry run.** Given caps for total runs, input/output/total tokens, known USD cost, and per-engine weekly quota allocation, the runner shall validate the experiment's 10% allocation and emit a dry-run estimate with assumptions and unknowns. Every provider invocation—not merely each top-level trial—shall receive an enforceable maximum for tokens or cost and time; this includes writer and reviewer calls, nested agents, verification, grading, and retries. Adapters that cannot enforce the declared invocation bound shall fail preflight for live runs. Before dispatch the ledger atomically reserves a conservative upper bound for the complete invocation tree, and after every usage update reconciles actual usage without releasing capacity still reachable by children or retries. On a reached or conservatively projected cap it shall dispatch no new invocation, allow configured in-flight work to finish or cancel safely, flush state, and end with `budget_exhausted` rather than corrupting or silently truncating the run. At least one finite hard cap is required for a live run. Offline tests shall make actual nested/retry demand exceed its estimate and prove that provider-side limits and the ledger stop further work without overspend.

**R7 — Resumable durable run.** Run configuration and state shall be stored on disk using SQLite plus content-addressed artifacts. Transactions shall make completed attempts idempotent. Restarting the same run shall verify config/task-set digests, skip committed work, reconcile orphaned in-flight attempts as interrupted, preserve budget usage, and continue pending work. A changed immutable configuration shall require a new run ID or explicit fork; it shall never mutate prior evidence.

### Grading

**R8 — Deterministic code graders.** The runner shall support command/test graders with isolated working directories, explicit command/exit-code contract, timeout, captured stdout/stderr, fixture digest, and optional structured subtest results. Code graders rank above subjective graders for a task's primary pass/fail when an executable oracle exists. Unit-test fixtures shall use sandbox fakes; production isolation mechanics are an implementation decision documented by the adapter.

**R9 — Calibrated LLM judge.** An LLM judge shall use a versioned rubric and structured verdict, be evaluated on a labeled calibration split before use, and report sample size, confusion matrix, accuracy, balanced accuracy, and agreement (Cohen's kappa; additionally weighted kappa for ordinal labels). The run config shall define minimum agreement and fail-closed behavior. Below threshold, malformed verdicts, judge transport errors, or rubric drift shall prevent automatic passing and route the item to the human queue. This capability extends, and does not change, the legacy `llm_judge` helper.

**R10 — Human sample queue.** The runner shall export a blinded queue containing a configurable random sample, all unresolved grader disagreements, and threshold failures. It shall support at least two independent labels, optional adjudication, randomized system ordering, label provenance, timestamps, and inter-rater agreement. Human labels are imported as append-only grader records; the runner shall reject task/system identities leaked into a blinded payload.

### Metrics and comparison

**R11 — Success metrics.** Reports shall include per-task successes/attempted trials and success rate; macro and micro summaries; and, for `n` observed trials containing `c` successes, the unbiased estimator `pass@k = 1 - C(n-c,k)/C(n,k)` for valid `1 <= k <= n` (equal to 1 when `n-c < k`). Reports shall include `pass^k`, defined as the fraction of tasks whose selected `k` trials all pass, alongside its selection rule; it shall never be labeled `pass@k`. Missing, errored, interrupted, and ungraded trials shall be counted and displayed separately under a declared denominator policy.

**R12 — Uncertainty.** Per-task Bernoulli success rates shall include Wilson confidence intervals, including valid bounds for all-pass and all-fail observations. Aggregate and difference estimates shall include a seeded paired/task-level bootstrap confidence interval with configured confidence level and resample count. When the bootstrap distribution degenerates at an all-pass/all-fail boundary, the preregistered fallback is a cluster-aware Wilson or score interval (or an exact paired interval for a paired difference), not a zero-width percentile interval. Synthetic boundary and correlated-cluster simulations shall check coverage before release. Reports shall state unit of resampling, seed, method or fallback, denominators, and suppress intervals with an explanatory reason when the estimand is undefined.

**R13 — Paired system comparison.** Comparing systems A and B shall align identical stable task IDs and a declared within-task trial aggregation rule, report per-task wins/ties/losses and excluded/missing pairs, run an exact two-sided sign test on non-ties, and report a seeded paired task bootstrap interval for the difference. Before dispatch, the manifest shall record prospective power and MDD for the corrected primary analysis, including alpha, target power, baseline, intracluster correlation or simulation assumptions, expected missingness, and the available per-engine quota/cost budget. E1's default 21 cells × approximately 50 pull-request clusters × three trials is approximately 3,150 top-level trials; the dry run must reconcile that matrix and all nested work with each engine's 10% quota or reduce scope by a preregistered, power-preserving amendment. Incomplete paired blocks are never silently converted to independent observations: the manifest fixes a complete-case or valid missing-data/sensitivity analysis, reports exclusions by arm, and marks an underpowered or broken pairing inconclusive. The report shall give achieved sample size and revised detectable effect. Tie-heavy comparisons shall not manufacture significance; zero non-ties shall yield an undefined sign-test p-value with an explanation.

### Accounting, evidence, and outputs

**R14 — Token provenance, cost, latency, and approvals.** Token accounting is mandatory for every trial and shall be retained per agent and per nesting level, with writer and reviewer reported separately: input tokens, output tokens, cached input tokens (including cache-read and cache-creation fields where supplied), and reasoning/thinking tokens. The preferred source is the engine's own usage report—such as Claude Code JSON `usage` (input, output, cache read, and cache creation), Codex turn usage including reasoning tokens, or an API provider's usage fields—and each such value is labeled `measured`. If a field is not reported, it is `unknown`, never zero. When engine usage is wholly or partly unavailable, the runner may estimate the missing field from the exact sent/received text with the model's tokenizer; only if that is unavailable may it use the approximately four-characters-per-token heuristic. Every estimate is labeled `estimated`, names its method, and includes an error margin and confidence level. Measured and estimated values shall never be silently combined: any total containing both displays the measured and estimated subtotals separately.

Every attempt shall also record engine/provider/model identifiers, total wall time, writer/reviewer wall time where separable, number of verify rounds, queue/generation/grading time, retry count, and approval requests/approved/denied/timed-out counts. A versioned price table shall map each model to input, output, and cached-token USD prices and record the effective date and source. Reports convert token usage to USD using that table even for subscription runs. Subscription results label this conversion **list-price-equivalent USD (would have cost at metered prices)**, retain measured tokens, and report subscription quota percentage consumed when the engine exposes it; unavailable quota is `unknown`. Reports shall aggregate totals and distributions by experiment, system, engine, task, status, agent role, and nesting level without double-counting retries.

Each adapter shall document and fixture-test its provider's disjoint accounting fields. In particular, it shall state whether reasoning/thinking tokens are already included in output tokens, keep cache reads and cache creation separate where their prices differ, and derive billable totals without adding overlapping fields. Golden usage payloads for every supported provider/model family shall verify normalization and price calculation; unknown semantics fail closed rather than guessing.

**R15 — Transcripts and OpenTelemetry.** Each attempt shall store or content-address its transcript and artifact manifest and link them to OpenTelemetry trace/span IDs when tracing is enabled. Existing exporters remain pluggable. Secrets and configured sensitive fields shall be redacted before persistence/export, with redaction policy/version recorded. A missing exporter or trace shall not fail a trial; the report shall show linkage coverage.

**R16 — Reports and provenance.** A completed, stopped, or partially resumed run shall emit a schema-versioned canonical JSON report and a human-readable Markdown report derived from it. Both shall identify run status, configuration and task-set digests, source revision, timestamps, systems/engines, seeds and seed-support flags, grader versions/calibration, budget estimate/actuals, metric definitions, exclusions, warnings, and artifact/trace references. JSON serialization shall be deterministic apart from documented volatile fields.

### Seeded defects and experiments

**R17 — Seeded-defect tooling.** Tooling for E1 shall provide a versioned defect catalogue with at least `bug`, `security`, `weakened_test`, and `leftover_file`. An injection recipe shall identify an eligible clean source revision, apply exactly one or a declared composition of deterministic mutations, verify applicability, and produce a patch plus machine-readable ground truth (defect ID/type, touched locations, detection rubric, expected clean behavior, injector version, source/patch digests). Every defect needs a defect-specific witness: preferably a deterministic test that passes on the clean revision and fails only after mutation, otherwise independently adjudicated evidence under a versioned rubric. Validation runs the witness on clean and mutated trees, reconstructs and reverts both states, and rejects ineffective, pre-existing, non-applying, or non-specific mutations. Generation sees the realistic review input but not hidden ground truth.

**R18 — Experiment definitions.** Experiment manifests shall encode the following preregistered designs and analysis rules:

- **E1 — Tandem review:** approximately 50 eligible real pull requests using the same balanced/stratified seeded-defect task set in every cell. The factorial factors are writer tier `{cheap, mid, frontier}`, review arm `{none, same_vendor, cross_vendor}`, and reviewer tier `{cheap, mid, frontier}`, with three trials per task/cell and randomized/blinded order. The `none` arm has no reviewer, so its three duplicate reviewer-tier combinations are pruned to one `reviewer_tier: not_applicable` cell per writer tier; this is the only default pruning (21 cells total). Any further infeasible combination must be listed before dispatch with an operational or availability justification, never silently omitted because of expected performance. Vendor/model versions, tier assignments, reviewer relationship, pruning, and the price-table version are frozen in the manifest.

  E1 uses a pre-injection writer design: each writer receives the same clean source task and produces a candidate patch; the injector then applies the assigned deterministic mutation to that candidate before any reviewer sees it. Writer-alone success is scored on the unmutated candidate against the task oracle, while review-arm success additionally requires detecting or correctly repairing the injected defect without introducing a regression. Clean negative-control candidates measure false accusations. Writer outputs are immutable shared artifacts: the same candidate, mutation, context window, and permitted metadata are reused across all applicable review arms and reviewer tiers, with no reviewer feedback returned to the writer. This separates whether frontier writers produce better clean patches from whether reviewers detect a controlled defect; a separate natural-defect observational outcome may assess writer-originated defects but may not be used to claim that post-hoc injection proves frontier writers avoid them.

  Reviewers are blinded to arm labels, vendor and writer identity, tier, model names, allocation probabilities, other reviewers' outputs, and hidden defect metadata; prompts and artifacts remove or pseudonymize those signals while retaining only review-relevant code and ordinary change context. Random allocation is concealed until the review record is committed, automated leakage checks inspect every engine-visible payload, and a sampled human audit confirms blinding. The no-review arm receives no reviewer call and is scored by the same automated oracle.

  E1 has two outcome levels so that every comparison has a defined baseline. **Defect catch rate** (detection or correct repair of the injected defect, verified against hidden ground truth) is defined only for review arms; the `none` arm has no reviewer and no catch rate. **End-to-end success** is defined for every cell: the final output passes the task oracle and contains no injected defect. For the `none` arm it is writer-alone success on the unmutated candidate; for a review arm it is success on the mutated candidate after review. Comparisons against the `frontier`/`none` baseline (H2, break-even, non-inferiority) use end-to-end success; comparisons between review arms and tiers (H1) use defect catch rate.

  The primary review-arm outcome is defect catch rate; the primary cross-arm outcome is end-to-end success. For every cell, reports give end-to-end success, defect catch rate (review arms) and `pass^k` with confidence intervals, cost per task, and cost per caught defect. They include a cost-quality Pareto frontier chart and machine-readable table and mark strictly dominated cells, including review cells that add cost without a quality gain. Break-even uses paired task-level quality differences against the frontier-writer-alone (`frontier`/`none`) baseline. The manifest must preregister a clinically/operationally justified non-inferiority margin `delta` on the absolute end-to-end success scale; a candidate qualifies only when the multiplicity-adjusted lower confidence bound for `candidate - baseline` is greater than `-delta`. Equality, a bound crossing `-delta`, missing pairs, or inadequate power is **inconclusive**, not “not significantly worse.” Qualifying cells are then ordered by tokens and USD; if none qualifies, that is the result. The report ends with a plain-language **“when the tandem saves money”** conclusion.

  Before dispatch the manifest partitions tests into hypothesis families: H1 tier-by-review-arm interactions, H2 non-inferiority comparisons with the frontier-alone baseline, and secondary/exploratory outcomes. It names the primary contrast in each family, controls family-wise error (Holm by default) or uses preregistered simultaneous confidence intervals, and labels all uncorrected searches exploratory. Selection of the cheapest qualifying cell is covered by the simultaneous intervals or validated with a nested task-level bootstrap that repeats selection; ordinary post-selection intervals are prohibited. The H1 contrast is the difference-in-differences in review benefit between cheap and frontier writers, paired by pull request. Secondary outcomes are false positives, repair, latency, verify rounds, and approvals.
- **E2 — Engine portability:** approximately 20 held-out coding tasks with deterministic tests, at least four engines (including configured coding-agent engines), common task contract and budgets, repeated trials, primary outcome test pass, and secondary setup failure, portability exclusion, cost, latency, retry, and approval outcomes. Each engine receives a fresh isolated workspace containing only the permitted task projection and fixtures; manifests, expected outputs, grader code, defect ground truth, task-set repository history, and host paths are mounted outside the engine-visible boundary. A tool-enabled canary test attempts file search, VCS history inspection, parent-directory traversal, and artifact-store access and must demonstrate that hidden data cannot be read. Engine-specific unavoidable capabilities are declared before running, never silently patched into results.
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
13. **Provider without token counts** — Given a provider that returns no usage counts, when a trial completes, then each absent field is `unknown` or is explicitly `estimated` from exact text with a tokenizer/error margin (or the lower-confidence four-character heuristic); no field or USD total presents the missing usage as zero.
14. **Cached and mixed usage** — Given measured input/output/cache-read/cache-creation usage and an estimated missing reasoning count, when reporting the trial, then cached tokens are priced with the dated model price entry and the total shows measured and estimated subtotals separately rather than presenting one undifferentiated measured total.
15. **Dominated review cell** — Given an E1 review cell with greater cost and no quality improvement over its no-review counterpart, when the Pareto analysis runs, then the chart and table retain the cell and label it dominated; the break-even conclusion does not claim that tandem saves money for that comparison.

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

A final offline gate runs `pytest -q tests/ext/evals tests/unit/test_evals.py`, `ruff check lazybridge/ext/evals tests/ext/evals`, and `python -m build`. Live smoke tests are opt-in and must use synthetic/non-held-out tasks.
