# Benchmarking and Evaluations Task DAG

**Status:** Planned
**Source of truth:** [`SPEC.md`](SPEC.md)
**Delivery rule:** every task below is one independently reviewable pull request. Do not combine tasks merely because they share files.

## 1. Execution conventions

- **Dependency notation:** `A → B` means B starts only after A is merged. Tasks on separate branches may be prepared in parallel when all of their dependencies are merged.
- **Executor:** `cloud` means hermetic CI/cloud-agent work with fakes and no provider credentials; `local` means an authorized operator environment is required because quota credentials or private task/job data are involved. Public implementation and offline fixtures still land through a normal PR.
- **Opposite-engine review:** a PR implemented with Codex is reviewed by Claude Code; a PR implemented with Claude Code is reviewed by Codex. The named `review_by` is mandatory and must not equal the executor engine. Statistical/security-sensitive tasks also request the listed human specialty.
- **Checks:** commands are run from the LazyBridge repository root. Each command is exact; if a planned path does not exist until that PR, the PR creates it. Provider/live checks are never part of the default unit gate.
- **Data boundary:** the public repository contains runner code and synthetic fixtures only. Private tasks, derived job extracts, and owned results stay in the private task-set repo.
- **Compatibility:** every PR touching `lazybridge/ext/evals` runs the legacy tests and preserves the API invariant in SPEC I1.

## 2. DAG

```text
T0 (legacy contract)
 ├── T1 (manifest/splits) ─┬── T3 (runner/resume/accounting) ─┬── T5 (human queue)
 │                         │                                  └── T7 (reports/OTel)
 │                         ├── T4 (graders) ───────────────────┘
 │                         └── T8 (seeded defects) ────────────────┐
 └── T2 (store/budget) ────┘                                       │
T1 ─────────────────────────── T6 (statistics) ─────────────────────┤
T3 + T4 + T5 + T6 + T7 + T8 ─────────────────────────────────── T9 (experiment schemas)
T9 ── T10 (public CLI) ── T11 (E1 dry run/local execution) ── T12 (E2 dry run/local execution) ── T13 (E3 local analysis)
T7 + T9 + T10 ── T14 (public runner documentation/release gate)
```

T14 can merge after T7, T9, and T10; it does not wait for private experiment execution. Experiment conclusions follow the D2 order even when later preparation overlaps.

## 3. Pull-request tasks

### T0 — Freeze the legacy eval compatibility contract

- **Requirements:** supports invariant I1; prerequisite for R4–R16.
- **Files:** `tests/ext/evals/test_legacy_api.py`; minimal test-only fixtures under `tests/ext/evals/`.
- **Dependencies:** none.
- **Needs:** inventory the exported symbols and synchronous/asynchronous behavior of the current `lazybridge.ext.evals`; no implementation changes.
- **Executor:** cloud, Claude Code.
- **review_by:** Codex.
- **Checks:**
  - `pytest -q tests/ext/evals/test_legacy_api.py`
  - `python -c "from lazybridge.ext.evals import EvalCase, EvalReport, EvalResult, EvalSuite, contains, exact_match, llm_judge, max_length, min_length, not_contains"`
  - `ruff check tests/ext/evals/test_legacy_api.py`

### T1 — Add task-set schemas, validation, and held-out boundaries

- **Requirements:** R1, R2, R3.
- **Files:** `lazybridge/ext/evals/manifest.py`, `lazybridge/ext/evals/schema/task-set-v1.json`, `tests/ext/evals/test_manifest.py`, `tests/ext/evals/test_splits.py`, `tests/ext/evals/fixtures/task_sets/`.
- **Dependencies:** T0.
- **Needs:** versioned strict models, canonical digest rules, extension namespace, relative/content-addressed artifacts, split-overlap audit, grader registry/version preflight, and a safe engine-facing task projection.
- **Executor:** cloud, Codex.
- **review_by:** Claude Code.
- **Checks:**
  - `pytest -q tests/ext/evals/test_manifest.py tests/ext/evals/test_splits.py`
  - `ruff check lazybridge/ext/evals/manifest.py tests/ext/evals/test_manifest.py tests/ext/evals/test_splits.py`
  - `python -m json.tool lazybridge/ext/evals/schema/task-set-v1.json >/dev/null`

### T2 — Implement durable storage and atomic budget ledger

- **Requirements:** R6, R7.
- **Files:** `lazybridge/ext/evals/store.py`, `lazybridge/ext/evals/budget.py`, `tests/ext/evals/test_store.py`, `tests/ext/evals/test_budget.py`.
- **Dependencies:** T0.
- **Needs:** migration-versioned SQLite schema, content-addressed artifact index, transactional attempt/reservation writes, conservative token/USD/run estimates, a versioned per-model input/output/cached-token price table with effective date and source, quota-allocation validation, unknown-price/usage behavior, and clean budget-stop states.
- **Executor:** cloud, Claude Code.
- **review_by:** Codex.
- **Checks:**
  - `pytest -q tests/ext/evals/test_store.py tests/ext/evals/test_budget.py`
  - `ruff check lazybridge/ext/evals/store.py lazybridge/ext/evals/budget.py tests/ext/evals/test_store.py tests/ext/evals/test_budget.py`
  - `python -m compileall -q lazybridge/ext/evals`

### T3 — Build the engine-neutral trial runner and resume flow

- **Requirements:** R4, R5, R6, R7, R14.
- **Files:** `lazybridge/ext/evals/runner.py`, `lazybridge/ext/evals/models.py`, additive exports in `lazybridge/ext/evals/__init__.py`, `tests/ext/evals/test_runner.py`, `tests/ext/evals/test_runner_failures.py`, `tests/ext/evals/test_resume.py`, `tests/ext/evals/test_accounting.py`.
- **Dependencies:** T1, T2.
- **Needs:** engine/agent adapter protocol, stable IDs and seeds, global/per-engine concurrency, timeout/retry attempt chains, crash reconciliation, and clean cancellation. Usage accounting captures engine-native measured usage (including Claude Code cache fields, Codex reasoning, and provider API usage), exact-text tokenizer estimates with error margins and a flagged four-character fallback, measured/estimated splits, writer/reviewer and nesting-level attribution, wall time, verify rounds, subscription quota percentage when exposed, approvals, and compatibility with standard and coding-agent engines without provider branching.
- **Executor:** cloud, Codex.
- **review_by:** Claude Code.
- **Checks:**
  - `pytest -q tests/ext/evals/test_legacy_api.py tests/ext/evals/test_runner.py tests/ext/evals/test_runner_failures.py tests/ext/evals/test_resume.py tests/ext/evals/test_accounting.py`
  - `ruff check lazybridge/ext/evals tests/ext/evals`
  - `mypy lazybridge/ext/evals/runner.py lazybridge/ext/evals/models.py`

### T4 — Add code graders and calibrated LLM judges

- **Requirements:** R3, R8, R9.
- **Files:** `lazybridge/ext/evals/graders/__init__.py`, `lazybridge/ext/evals/graders/code.py`, `lazybridge/ext/evals/graders/llm.py`, `tests/ext/evals/graders/test_code.py`, `tests/ext/evals/graders/test_llm.py`.
- **Dependencies:** T1, T2.
- **Needs:** grader protocol/registry, isolated command workspace contract, evidence capture, structured judge verdicts, calibration metrics and thresholds, failure routing, rubric digests, and no behavior change to legacy `llm_judge`.
- **Executor:** cloud, Claude Code.
- **review_by:** Codex; human review by an evaluation-methods owner.
- **Checks:**
  - `pytest -q tests/ext/evals/graders/test_code.py tests/ext/evals/graders/test_llm.py tests/ext/evals/test_legacy_api.py`
  - `ruff check lazybridge/ext/evals/graders tests/ext/evals/graders`
  - `mypy lazybridge/ext/evals/graders`

### T5 — Add blinded human sampling and adjudication queue

- **Requirements:** R9, R10.
- **Files:** `lazybridge/ext/evals/human.py`, `tests/ext/evals/test_human_queue.py`, `tests/ext/evals/fixtures/human_labels/`.
- **Dependencies:** T3, T4.
- **Needs:** seeded sampling, disagreement/threshold routing, identity-field denylist, randomized order, two-label/adjudication state machine, provenance, append-only imports, and inter-rater agreement.
- **Executor:** cloud, Codex.
- **review_by:** Claude Code; human privacy review.
- **Checks:**
  - `pytest -q tests/ext/evals/test_human_queue.py`
  - `ruff check lazybridge/ext/evals/human.py tests/ext/evals/test_human_queue.py`
  - `mypy lazybridge/ext/evals/human.py`

### T6 — Implement metrics and paired comparisons

- **Requirements:** R11, R12, R13, and E1 analysis in R18.
- **Files:** `lazybridge/ext/evals/metrics.py`, `lazybridge/ext/evals/compare.py`, `tests/ext/evals/test_metrics.py`, `tests/ext/evals/test_compare.py`.
- **Dependencies:** T1.
- **Needs:** exact combinatorial `pass@k`, declared `pass^k` selection, denominator policy, Wilson bounds, seeded task bootstrap, exact sign test, paired bootstrap, MDD calculation/report inputs, and explicit undefined states. E1 analysis adds per-cell defect catch rate and `pass^k` intervals, cost per task/caught defect, Pareto frontier and dominated-cell classification, and preregistered non-inferiority break-even selection against frontier-writer-alone with token/USD savings.
- **Executor:** cloud, Claude Code.
- **review_by:** Codex; human statistical review.
- **Checks:**
  - `pytest -q tests/ext/evals/test_metrics.py tests/ext/evals/test_compare.py`
  - `ruff check lazybridge/ext/evals/metrics.py lazybridge/ext/evals/compare.py tests/ext/evals/test_metrics.py tests/ext/evals/test_compare.py`
  - `mypy lazybridge/ext/evals/metrics.py lazybridge/ext/evals/compare.py`

### T7 — Link transcripts/OTel and generate canonical reports

- **Requirements:** R14, R15, R16.
- **Files:** `lazybridge/ext/evals/transcripts.py`, `lazybridge/ext/evals/reporting.py`, `lazybridge/ext/evals/schema/report-v1.json`, `tests/ext/evals/test_transcripts.py`, `tests/ext/evals/test_otel.py`, `tests/ext/evals/test_reports.py`.
- **Dependencies:** T3, T4, T5, T6.
- **Needs:** content-addressed transcript manifests, versioned redaction, optional OTel context/exporter links, deterministic JSON, Markdown derived only from JSON model, partial/budget-stopped reports, and complete provenance/warnings. Reports preserve measured versus estimated token subtotals, model price-table date/source, list-price-equivalent subscription cost and exposed quota percentage, writer/reviewer nesting detail, verify rounds, E1 Pareto chart/table, and the plain-language “when the tandem saves money” conclusion.
- **Executor:** cloud, Codex.
- **review_by:** Claude Code.
- **Checks:**
  - `pytest -q tests/ext/evals/test_transcripts.py tests/ext/evals/test_otel.py tests/ext/evals/test_reports.py`
  - `ruff check lazybridge/ext/evals/transcripts.py lazybridge/ext/evals/reporting.py tests/ext/evals/test_transcripts.py tests/ext/evals/test_otel.py tests/ext/evals/test_reports.py`
  - `python -m json.tool lazybridge/ext/evals/schema/report-v1.json >/dev/null`

### T8 — Build deterministic seeded-defect tooling

- **Requirements:** R17.
- **Files:** `lazybridge/ext/evals/defects.py`, `lazybridge/ext/evals/defect_catalogue.py`, `tests/ext/evals/test_seeded_defects.py`, `tests/ext/evals/fixtures/defects/`.
- **Dependencies:** T1.
- **Needs:** catalogue/version schema, deterministic injector API for bug/security/weakened-test/leftover-file classes, eligibility checks, patch and hidden-ground-truth digests, reconstruction/revert proof, and synthetic public fixtures only.
- **Executor:** cloud, Claude Code.
- **review_by:** Codex; human security review.
- **Checks:**
  - `pytest -q tests/ext/evals/test_seeded_defects.py`
  - `ruff check lazybridge/ext/evals/defects.py lazybridge/ext/evals/defect_catalogue.py tests/ext/evals/test_seeded_defects.py`
  - `mypy lazybridge/ext/evals/defects.py lazybridge/ext/evals/defect_catalogue.py`

### T9 — Encode and validate E1–E3 experiment manifests

- **Requirements:** R18, with R1–R17 as referenced schema contracts.
- **Files:** `lazybridge/ext/evals/experiments.py`, `tests/ext/evals/test_experiment_manifests.py`, `tests/ext/evals/fixtures/experiments/e1.yaml`, `tests/ext/evals/fixtures/experiments/e2.yaml`, `tests/ext/evals/fixtures/experiments/e3.yaml`.
- **Dependencies:** T3, T4, T5, T6, T7, T8.
- **Needs:** preregistration fields, enforced experiment order/amendments, E1's 3 writer tiers × 3 review arms × 3 reviewer tiers with three trials on the same seeded tasks, canonical pruning of reviewer tiers for `none`, explicit justification for any further pruning, tier/vendor/model/price freezes, paired break-even/non-inferiority and Pareto analysis declarations, E2 four-engine minimum/test oracle, E3 read-only extract contract, privacy metadata, dry-run estimates, and synthetic examples without owned results.
- **Executor:** cloud, Codex.
- **review_by:** Claude Code; human statistical and privacy review.
- **Checks:**
  - `pytest -q tests/ext/evals/test_experiment_manifests.py`
  - `ruff check lazybridge/ext/evals/experiments.py tests/ext/evals/test_experiment_manifests.py`
  - `python -c "import yaml; [yaml.safe_load(open(p, encoding='utf-8')) for p in ('tests/ext/evals/fixtures/experiments/e1.yaml','tests/ext/evals/fixtures/experiments/e2.yaml','tests/ext/evals/fixtures/experiments/e3.yaml')]"`

### T10 — Implement the public eval CLI

- **Requirements:** executable entry point for R1–R18 workflows.
- **Files:** `lazybridge/ext/evals/__main__.py`, `lazybridge/ext/evals/cli.py`, `tests/ext/evals/test_cli.py`.
- **Dependencies:** T9.
- **Needs:** public argument parsing and command dispatch for `validate`, `estimate`, `run`, `report`, `analyze-approvals`, and `audit-privacy`; stable exit codes; actionable errors; and no provider or private-data access during `--help`, validation, or dry runs.
- **Executor:** cloud, Codex.
- **review_by:** Claude Code.
- **Checks:**
  - `python -m lazybridge.ext.evals --help`
  - `python -m lazybridge.ext.evals validate --help`
  - `python -m lazybridge.ext.evals run --help`
  - `pytest -q tests/ext/evals/test_cli.py`
  - `ruff check lazybridge/ext/evals/cli.py lazybridge/ext/evals/__main__.py tests/ext/evals/test_cli.py`

### T11 — Prepare and execute E1 in the private task-set repo

- **Requirements:** operational validation of R1–R18 for E1.
- **Files:** private task-set repo only: versioned E1 manifest, source eligibility ledger, injected fixtures/ground truth, dry-run report, run database/artifacts, JSON/Markdown result reports, and preregistration/amendments.
- **Dependencies:** T10.
- **Needs:** approximately 50 eligible pull requests, one balanced seeded-defect task set shared by all 21 default factorial cells (three writer tiers; no/same-vendor/cross-vendor review; three reviewer tiers where review exists), three trials per task/cell, and explicit preregistered justification for any additional pruning. Freeze tier/model/vendor assignments and the dated/source-linked input/output/cached price table. Capture per-role/per-nesting measured usage, estimation provenance and error margins for gaps, measured/estimated splits, list-price-equivalent subscription USD, exposed quota percentage, wall time, and verify rounds. Produce cell-level quality intervals and costs, Pareto/dominated results, break-even token/USD savings versus frontier-writer-alone, and “when the tandem saves money.” Weekly quota data, credentials, legal/handling review, and blinded human capacity are required. First run a synthetic smoke; approve live dispatch only if every engine estimate is at or below 10% of weekly quota and a finite hard cap is configured.
- **Executor:** local authorized operator, Claude Code for orchestration.
- **review_by:** Codex; human experiment owner before dispatch and before publication.
- **Checks:**
  - `python -m lazybridge.ext.evals validate task-set.yaml experiment-e1.yaml`
  - `python -m lazybridge.ext.evals estimate experiment-e1.yaml --format json --output dry-run.json`
  - `python -m lazybridge.ext.evals run experiment-e1.yaml --dry-run`
  - `python -m lazybridge.ext.evals report run.sqlite --json report.json --markdown report.md`

### T12 — Prepare and execute E2 in the private task-set repo

- **Requirements:** operational validation of R1–R16 and R18 for E2.
- **Files:** private task-set repo only: versioned E2 task/experiment manifests, test fixtures, capability declarations, dry-run report, run database/artifacts, and JSON/Markdown reports.
- **Dependencies:** T11 complete and E1 baseline recorded; implementation may be prepared after T10.
- **Needs:** approximately 20 held-out coding tasks, deterministic test commands, at least four configured engines including coding-agent engines where available, equalized contracts/budgets, per-engine weekly quota inputs, credentials, and explicit portability exclusions.
- **Executor:** local authorized operator, Codex for orchestration.
- **review_by:** Claude Code; human experiment owner.
- **Checks:**
  - `python -m lazybridge.ext.evals validate task-set.yaml experiment-e2.yaml`
  - `python -m lazybridge.ext.evals estimate experiment-e2.yaml --format json --output dry-run.json`
  - `python -m lazybridge.ext.evals run experiment-e2.yaml --dry-run`
  - `python -m lazybridge.ext.evals report run.sqlite --json report.json --markdown report.md`

### T13 — Analyze E3 approval gates in the private task-set repo

- **Requirements:** operational validation of R14, R16, R18 for E3.
- **Files:** private task-set repo only: extract schema/query version, redacted immutable extract or digest, inclusion/deduplication manifest, missingness audit, analysis config, and JSON/Markdown reports.
- **Dependencies:** T12 complete and E2 baseline recorded; schema/audit preparation may begin after T10.
- **Needs:** authorized read-only access to existing bridge job data, privacy review, stable job/attempt/gate keys, declared observation window, deduplication and missingness policy, and no provider credentials unless a separate validation demands them.
- **Executor:** local authorized operator, Claude Code for analysis.
- **review_by:** Codex; human privacy/data owner.
- **Checks:**
  - `python -m lazybridge.ext.evals validate experiment-e3.yaml extract-manifest.json`
  - `python -m lazybridge.ext.evals analyze-approvals experiment-e3.yaml --input extract.jsonl --output run.sqlite`
  - `python -m lazybridge.ext.evals report run.sqlite --json report.json --markdown report.md`
  - `python -m lazybridge.ext.evals audit-privacy extract-manifest.json report.json`

### T14 — Publish runner documentation and run the release gate

- **Requirements:** documentation and release evidence for R1–R18; invariants I1–I6.
- **Files:** `docs/guides/mid/evals.md`, `docs/design/evals/SPEC.md`, `docs/design/evals/TASKS.md`, `docs/changelog.md`, and any navigation entry required by the documentation build.
- **Dependencies:** T7, T9, T10.
- **Needs:** migration-free legacy examples, new manifest/runner/metric examples using synthetic data, warnings about held-out leakage and `pass@k` versus `pass^k`, budget/resume tutorial, private-data boundary, and links to JSON schema.
- **Executor:** cloud, Codex.
- **review_by:** Claude Code.
- **Checks:**
  - `pytest -q tests/ext/evals tests/unit/test_evals.py`
  - `ruff check lazybridge/ext/evals tests/ext/evals`
  - `mkdocs build --strict`
  - `python -m build`

## 4. Merge and release gates

A task PR is mergeable only when its exact checks pass, its dependency PRs are merged, the opposite-engine review approves, and new behavior has executable offline coverage. T11–T13 additionally require human approval at the specified dispatch/publication gates; their private artifacts are not copied into LazyBridge.

Before the first public runner release:

1. T0–T10 and T14 are merged.
2. The validation matrix in `SPEC.md` has no requirement without a passing executable check.
3. Legacy API tests, the entire offline eval suite, lint, docs, and package build pass.
4. A synthetic interrupted run resumes bit-for-bit at the report model level, except documented timestamps.
5. A synthetic cap-hit run proves no work is dispatched beyond reserved capacity.
6. Security/privacy review signs off transcript redaction, held-out projection, command-grader boundaries, and human-queue blinding.
