# Cross-provider review benchmark v2

This example measures whether a reviewer from another provider changes errors
in final implementations. Publish positive and negative results. Arm A alone
cannot establish a review effect. **Do not call ClaudeCodeEngine live until
the operator releases its reserved quota.** Only Codex A runs during calibration.

## Why v2 exists

The frozen [v1 set and results](v1/README.md) had a coding ceiling:

| v1 category | Codex A | Pass rate | Wilson 95% CI |
|---|---:|---:|---:|
| Coding | 20/20 | 100% | 83.9–100% |
| LazyBridge API | 2/5 | 40% | 11.8–76.9% |

Review cannot improve 20/20. The five API items exposed invented verification
types/sentinel helpers and a missing required run argument, but n=5 is small.
v1's original JSONL, same-candidate regrading audit, protocol archive and notes
are retained unchanged in [v1/results](v1/results). Versioned v1 JSON preserves
its specs, corrected tests and references; original top-level modules remain
compatibility entry points. v1 task SHA-256:
`b3fb9b526e61175fde9021f10b5c54ed8b9d01b5e2fb303503b2726b46dfbb4f`.
Do not append new-runner records to historical generation fingerprints.

v2 contains 20 harder original coding tasks, authored for this example without
copying public benchmarks, and 25 executable API items: five unchanged v1
anchors plus twenty new. Difficulty is a design hypothesis; report observed
calibration and frozen baseline even when the target is missed.

## Calibration and pre-registration

The [protocol](v2/CALIBRATION.md) was registered before v2 generation. Targets
are 40–70% separately: 8–14/20 coding, 10–17/25 API. Run the entire initial
design, allow at most TWO design revisions, publish every run. Stop early only
when both targets are met; otherwise select the latest design at the revision
limit. Never relax a grader to accept an observed output. Strengthening a spec
may add prewritten tests, retaining existing assertions.

[v2/calibration](v2/calibration) contains every JSONL, summary, manifest and
pre-call `*.tasks.zip` snapshot. Every record has task and protocol hashes.
Calibration candidates never become the frozen paired baseline.

<!-- v2-results-start -->
Every calibration used all 45 items. No individual candidates or runs were discarded.

| Calibration design | Coding pass rate | Wilson 95% CI | API pass rate | Wilson 95% CI |
|---|---:|---:|---:|---:|
| Initial | 20/20 (100.0%) | 83.9–100.0% | 11/25 (44.0%) | 26.7–62.9% |
| Revision 1 | 19/20 (95.0%) | 76.4–99.1% | 11/25 (44.0%) | 26.7–62.9% |
| Revision 2 | 19/20 (95.0%) | 76.4–99.1% | 12/25 (48.0%) | 30.0–66.5% |

**Coding calibration missed the 40–70% target after both permitted revisions.**
The final design still has a near ceiling; do not describe v2 as solving
the coding difficulty problem. The expanded API set provides substantially
more failure opportunities. The predeclared selection rule chooses the latest
design, not the lowest-scoring run.

Frozen task SHA-256: `660d9ccf90a01ec6ba59397c250c4df11ca632e25a363642bcdfe8c021d04b8a`.
Registered at `2026-10-11T23:38:29.097123+00:00` (UTC), before the fresh A draw
and before any B/C/D generation. [freeze.json](v2/freeze.json) records
all calibration hashes, timestamps and the selection rule. Frozen execution
rejects changed task files. No redesign after freezing.

Fresh frozen baseline (the A sample for later paired comparisons):

| Category | Arm | Pass rate | Wilson 95% CI |
|---|---|---:|---:|
| coding | A | 19/20 (95.0%) | 76.4–99.1% |
| api | A | 9/25 (36.0%) | 20.2–55.5% |

The fresh API score is outside the calibration target. Completed engine
errors remain failures, and the frozen set was not redesigned in response.

See [JSONL](v2/results/arm_a.jsonl), [summary](v2/results/arm_a.md),
and [audit notes](v2/results/NOTES.md). B/C/D remain pending; no review
effect or paired p-value can be inferred from A alone. Later arms append
compatibly to the baseline JSONL, using this exact source checkout.

Live runs request `gpt-6.1-sol`, reasoning `high`; reported models: `gpt-6.1-sol`.
UTC generation range: `2026-10-11T21:21:28.843197+00:00` to
`2026-10-12T00:30:41.564272+00:00`. Python 3.11.13, LazyBridge
1.8.0, codex-cli 0.160.0, pytest 9.1.1;
CLI-reported model IDs are not immutable weight/version identifiers.
Claude has not run; its requested default is `sonnet`, and its reported
model/version/date must be recorded when the operator releases quota.
<!-- v2-results-end -->

## Method

| Arm | Writer | Reviewer |
|---|---|---|
| A | CodexEngine | none |
| B | ClaudeCodeEngine | none |
| C | CodexEngine | ClaudeCodeEngine |
| D | ClaudeCodeEngine | CodexEngine |

Engines use subscription runtimes. All arms receive the same signature/docstring
and fixed writer prompt in [harness.py](harness.py). The fixed reviewer prompt
there sees the spec and candidate, never hidden tests, and checks correctness,
boundaries, validation, Unicode, dates, rounding and mutation. It replies
`approved` or `rejected: <defects and correction>`. Both may run their own
`python_check` examples against a copy of the public package. Codex native
filesystem/shell/browser tools are disabled.

C/D use real `Agent(verify=judge, max_verify=2)`: at most TWO TOTAL writer
attempts, each reviewed. The last candidate is returned even if rejected.
A/B have one attempt. Review adds compute and another drafting opportunity;
this experiment lacks same-provider review and extra-draft controls and cannot
separate those effects from reviewer quality.

Each arm independently generates its first draft. Pairing compares outcomes on
the same task, not review of an identical A/B candidate. Consequently a task
classified as fixed or broken can reflect sampling variation as well as review.

Order uses deterministic private seed `20261011`, not deterministic model
sampling. Native model retries and format retries are disabled. Default budgets
are 180s/model call, 20s/hidden pytest process, 5s/own check. Optional
`lazytools.projects.quota_telemetry` checks only selected providers before
construction/calls and stops if any used-percent window exceeds 95%.
Unavailable telemetry emits a warning and skips the check. A never queries
Claude telemetry. Quota stops retain partial work without marking it completed.

Hidden tests grade only the final candidate in a fresh pytest subprocess against
this checkout's actual package. Only the target function is imported; candidate
imports cannot shadow oracle imports. Tests include mutation/error cases and
independent brute-force/exact arithmetic where applicable. API graders use
inert clients, MockAgent or supplied fake engines; no real provider calls.
References and negative controls are validated before model calibration.

`tasks.json` holds visible specs; `hidden_tests.json` and `mock_solutions.json`
are evaluator-only. They are public for reproduction but never copied to model
workspaces or inserted into prompts. `python_check` runs arbitrary host Python:
this is cooperative isolation, not a hostile-code OS sandbox. Audit candidate
code and tool logs before claiming contamination-free evaluation.

JSONL records include pass/fail, attempts, verdicts, candidate, complete provider
calls, own checks, wall time, errors, quota readings, requested/reported models,
runtime versions, dates and fingerprints. Completed errors count as failures.
Task wall times include local setup and hidden grading; they are not a
controlled provider-latency comparison.
Reports keep coding/API separate, use Wilson 95% intervals, and pair A/C and
B/D on common task IDs, listing fixed/broken tasks and two-sided exact McNemar
p. Missing arms remain pending; calibration scores never enter paired tests.

## Reproduce

From repository root with Python 3.11+ and test dependencies (`pip install -e
".[test]"`). Install `tzdata` on Windows for the date tasks and `ruff` for
lint/format checks. Later live B/C/D also require `pip install -e
".[claude-code]"`, the Claude Agent SDK and authenticated Claude Code. Live A
requires an authenticated Codex CLI. This example installs no runtime itself.

```console
# Entire scripted MockAgent pipeline, no model calls
python -m examples.cross_review_benchmark.harness --suite v2 --mock --output mock-v2.jsonl

# Frozen Codex baseline; resumes by skipping completed records
python -m examples.cross_review_benchmark.harness --suite v2 --arms A --output examples/cross_review_benchmark/v2/results/arm_a.jsonl

# ONLY AFTER operator releases Claude quota
python -m examples.cross_review_benchmark.harness --suite v2 --arms B,C,D --output examples/cross_review_benchmark/v2/results/arm_a.jsonl

# Budgeted smoke run in a separate file
python -m examples.cross_review_benchmark.harness --suite v2 --arms A --tasks api --max-tasks 3 --quota-threshold 90 --output api-smoke.jsonl

# v1 through shared runner into a NEW file
python -m examples.cross_review_benchmark.harness --suite v1 --mock --output mock-v1.jsonl
```

`--tasks` accepts `all`, `coding`, `api` or comma-separated IDs. `--arms`,
`--max-tasks`, models/effort and timeouts are configurable. Resume skips failures
too and rejects malformed/duplicate data or changed mode, task files,
runner/prompts, package code and model budgets. Use a new output for changed
configuration. Default suite remains v1 for compatibility; explicitly select v2.

Original calibration command (run_02/run_03 use distinct paths after recording
the revised design):

```console
python -m examples.cross_review_benchmark.harness --suite v2 --phase calibration --arms A --output examples/cross_review_benchmark/v2/calibration/run_01_arm_a.jsonl
python -m examples.cross_review_benchmark.freeze <run_01.jsonl> [<run_02.jsonl> <run_03.jsonl>]
```

Published v2 is frozen. Replay old calibration snapshots in an isolated checkout
by restoring their three JSON files; do not change the final set in place.
Freeze refuses overwrite, incomplete/live-non-A/mixed runs or more than three.
Task SHA-256 covers `tasks.json`, `hidden_tests.json`, `mock_solutions.json`,
in that order: for each append UTF-8 filename, NUL, decimal byte length, NUL,
bytes with CRLF normalized to LF. Verify with [suites.py](suites.py).
The broader protocol hash also covers run configuration, runner and package
Python source. The historical v1 generation archive remains reproducible.

## Difficulty spread and API coverage

| Coding task | Main interacting constraints |
|---|---|
| transaction_frames | Nested failure isolation, ignored-move validation, self transfers |
| escaped_records | Escaped delimiters, quotes, LF data and terminal records |
| capped_apportion | Two-level grouped allocation, saturation, exact remainders/ties |
| wall_occurrences | DST gaps/both folds, half-open UTC range, half-hour transitions |
| patch_spans | Original offsets, insertion order, overlap validation |
| ttl_lru | Nested rollback, persistent clock/observations, expiry and LRU |
| nested_expansion | Slicing enormous symbolic repeats, eager syntax validation |
| ragged_walk | Retained direction, disconnected restarts, empty/ragged rows |
| dependency_waves | Cancellation, maximum-cardinality resource-constrained waves |
| significant_round | Exact rational quotients, ties/carry, thousands-digit coefficients |
| escaped_glob | Eager parsing, escaped ranges, lexicographic star capture lengths |
| rational_expression | Lazy branches, exact fractions, symbolic polynomial cancellation |
| priority_overlay | Open/closed boundaries, singletons, infinities, label merging |
| prefix_cents | Exact cumulative ties-to-even and signed cancellation |
| utf8_rechunk | Split UTF-8/UTF-16, BOM preservation, whole code points, byte budgets |
| folded_spans | Fold expansion, mark elision, multiple original boundary pairs |
| lease_trace | Global fencing tokens, inclusive expiry, zero-duration success |
| coupon_route | Required node subsequences, nonadjacent discounts, path tie breaks |
| revision_log | Winning revisions, undo replay, reinsertion order, deep copies |
| indented_records | Tab stops, ignored comments, JSON strings, nesting levels |

All coding items are intended harder than v1, with stateful bookkeeping,
parsers, arithmetic and graph algorithms. API items cover Plan/Step naming,
sentinels, round trips/checkpoints, verify/verified aliases, Store memory,
CAS/prefix/provenance, SessionRegistry bind/rename/forget and engine aliases,
Session events, Envelope factories/fields, Tool cloning/schema construction,
and Codex policy/native-review options. Grading needs no live Claude engine.

## Limitations

Small n, correlated tasks, adaptive calibration on one writer, one sample per
task/arm, two providers, small Python tasks only. API results measure executable
LazyBridge usage, not general factual accuracy. Repeated themes and reused API
anchors reduce effective sample size. Calibration can overfit Codex difficulty
and favor/disfavor later reviewers. Original tasks do not guarantee the models
have never seen package code or relevant documentation in training.

Subscription routing/defaults change; requested IDs are not immutable weights.
Retain dates, reported metadata, effort and CLI/SDK/package versions. A ceiling
or floor after allowed revisions must be reported. Wilson intervals are
descriptive binomial intervals, with dependence/adaptation limiting inference.
McNemar tests are exploratory, uncorrected for multiple comparisons. Finite
hidden tests miss bugs. Binary task scores weight a wrong exception type and
a wrong core algorithm equally; inspect the recorded failures for severity.
No result here validates a universal marketing claim.
