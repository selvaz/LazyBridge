# Cross-provider review benchmark

This example measures task-level correctness of generated Python with and without
LazyBridge's cross-provider `verify=` loop. Publish failures as well as successes.
The separate API section measures executable use of the real LazyBridge API,
including invented or misused constructor parameters. It is not a general factual
knowledge benchmark.

## Reproduce

From the repository root, with Python 3.11+:

```powershell
python -m pip install -e ".[test,claude-code]"
python -m pip install ruff
# CI/local dry run: no authentication or provider calls, all four arms.
python -m examples.cross_review_benchmark.harness --mock --output mock_results.jsonl
# Codex subscription baseline only (also the initial published experiment).
python -m examples.cross_review_benchmark.harness --arms A --output cross_review_results.jsonl
# AFTER the operator's Claude quota reset: append the remaining arms.
python -m examples.cross_review_benchmark.harness --arms B,C,D --output cross_review_results.jsonl
```

Live Codex requires the locally authenticated Codex CLI. Live Claude requires the
Claude Agent SDK and locally authenticated Claude Code subscription. No API keys
are configured by this example. The example installs no CLI automatically.
`--mock` never constructs either live engine and never reads quota telemetry.

Budget controls:

```powershell
python -m examples.cross_review_benchmark.harness --arms A --tasks coding --max-tasks 3 --quota-threshold 90 --output pilot.jsonl
python -m examples.cross_review_benchmark.harness --arms A --tasks coding_03_cent_parts,api_04_review_agent --output subset.jsonl
python -m examples.cross_review_benchmark.harness --help
```

`--max-tasks` limits selected task IDs, not task/arm calls. A reviewed arm can make
two writer calls and two reviewer calls per task. All arms run serially; tasks
are shuffled by a private `random.Random(20261011)` (override `--seed`). The seed
controls ordering and mock behavior, **not** provider sampling. CLI runtimes do
not expose a common deterministic generation seed.

An existing JSONL is automatically resumed by `(task_id, arm)`; completed failures
and engine errors are also skipped. To rerun them, choose a new output file and
report it as a separate experiment. No overwrite or cherry-picking is performed.
The protocol hash covers model configuration, seeds, timeouts, exact prompts,
task specs, hidden tests, scripted fixtures, harness source and package Python
source. CRLF/LF differences are normalized. Changing any of these rejects
resumption; changing selected arms/tasks, output path or quota threshold does not.
Malformed/truncated records and duplicate task/arm records fail explicitly.
Do not run multiple processes against one output file.

To extend the **published baseline**, append later arms to the corrected log,
using the same default model settings, after the Claude quota reset:

```powershell
python -m examples.cross_review_benchmark.harness --arms B,C,D --output examples/cross_review_benchmark/results/arm_a_regraded.jsonl
```

The quota brake uses `lazytools.projects.quota_telemetry.read_quota` if importable,
checking **only the providers needed by the selected arm**, before construction
and before every model call. It stops if any returned usage window is greater
than `--quota-threshold` (default 95 percent). Missing, broken or timed-out
telemetry produces a warning and skips the check. There is no Claude usage query
in arm A. A mid-review stop writes `.partial.json` for audit and leaves that task
unfinished; resume reruns it. `.status.json` records the stop reason. Provider
transport retries are disabled at engine level; LazyBridge's Codex transport may
still retry a failed startup before any turn was sent.

## Method and output

| Arm | Writer | Reviewer | Maximum writer attempts |
|---|---|---|---:|
| A | CodexEngine | none | 1 |
| B | ClaudeCodeEngine | none | 1 |
| C | CodexEngine | ClaudeCodeEngine via `verify=` | 2 |
| D | ClaudeCodeEngine | CodexEngine via `verify=` | 2 |

All writers receive the identical `WRITER_PROMPT` plus the task's signature and
docstring. The fixed reviewer system prompt is `REVIEWER_PROMPT` in [harness.py](harness.py).
LazyBridge supplies the reviewer with the candidate source and original writer
task in its own `Evaluate this output: ... Original task: ...` wrapper. The
reviewer starts with `approved` or `rejected: <reason>`. On rejection the original
writer receives only that reason as additional context. No hidden test results
are fed back. `max_verify=2` means at most **two total attempts**, each reviewed;
the second output is returned even if rejected. Rejection itself is not a task
failure: final hidden pytest correctness determines pass/fail.

Each task/arm has fresh engines and an ephemeral workspace/session. Codex's shell,
unified exec, apps, plugins, browser, computer use, code host, memory, host skill
discovery and multi-agent features are disabled through App Server config.
Native web search is disabled. Claude has no file roots or native web tools and
uses the engine's default empty setting sources. Both receive a `python_check`
tool that can execute their own assertions in an isolated workspace containing
only a copy of the public LazyBridge package. The check is logged and time-bounded.
They can inspect that public API, so API scores measure tool-assisted API use.

[tasks.py](tasks.py) contains only model-visible specs. [hidden_tests.py](hidden_tests.py)
and [mock_fixtures.py](mock_fixtures.py) stay in the evaluator process. Hidden tests
are materialized **after** the model finishes, in a different temporary directory.
Each candidate is graded in a fresh pytest subprocess, without repository
conftests, project pytest configuration or auto-loaded plugins. The subprocess
imports LazyBridge from this checkout, irrespective of an unrelated editable
installation. There is a hidden-test wall timeout (20 seconds by default).
This is protocol separation for cooperative coding models, not an OS security
sandbox for hostile code: arbitrary Python can access the host. Run unfamiliar
models/code in an appropriately isolated VM/container. Models receive no path
to the repository, hidden tests, reference solutions or previous results.
“Hidden” means withheld during inference; publishing this directory deliberately
makes tests available to humans for independent replication.

JSONL includes candidate code, original prompts and retry context, writer and
reviewer calls, verdicts, attempts, per-call envelope metadata (including reported
model/provider), own checks, hidden pytest output, errors, time, quota readings,
date, interpreter/platform and installed package/CLI versions. Subscription cost
fields are not a monetary estimate. `.md` is regenerated after every completed
record. Runtime and hidden-test errors count as failures; inspect their logs to
distinguish infrastructure failures from wrong solutions. Pending tasks have no
denominator. Coding and API items are never pooled in the summary.

Saved outputs can be regraded without any engine or telemetry call, preserving
the original log and all old grades in a new file. Every new row records the
original generation protocol hash, previous grade, reason and regrade date.
Regrading checks that specs/prompts, seed, mode and model configuration still
match, and that the candidate equals the recorded model output. Disclose any
changes to expectations; do not tune tests to favor a model.

```powershell
python -m examples.cross_review_benchmark.regrade --input cross_review_results.jsonl --output regraded_results.jsonl --reason "Describe the grader change"
```

The table reports pass counts/rates and Wilson 95% intervals with z=1.9599639845.
For A/C and B/D it joins task IDs, lists baseline-fail/review-pass (“fixed”) and
baseline-pass/review-fail (“broken”), and computes the two-sided exact McNemar
p-value: twice the smaller Binomial(discordant_n, 0.5) tail, capped at 1. Zero
discordances yields p=1; no paired records yields a pending comparison. The JSONL
is the source of truth. Task lists appear in the table's Markdown comments.

Mock responses cycle through four scripted scenarios by task index: correct and
approved; wrong then repaired; correct then regressed after harmful feedback;
wrong but approved. The pipeline executes real hidden pytest tests and the real
Agent verification loop. Mock scores are **not model evidence**.

## Original task inventory and intended difficulty

Written specifically for this example; no public benchmark tasks were copied or
consulted. Difficulty labels are prospective judgments, not calibrated scores.
Simple cases should be reliable for a strong model; combined requirements should
leave room for residual errors. Specs, test inputs and expected answers were frozen
before the live baseline. The grader's import isolation was repaired after the run;
assertions and task expectations were not changed. Both logs and the original
protocol snapshot are published so this repair can be independently audited.

| Coding item | Intended level | Principal pitfalls |
|---|---|---|
| gap_runs | easy | duplicate/negative runs, no mutation |
| clipped_windows | easy | short tails, code points, bool validation |
| cent_parts | hard | half-up cents, largest remainders, large Decimals |
| fold_counts | medium | NFC/casefold/NFC, stable ordering |
| utc_day | medium | offset crossing UTC date, reject naive timestamps |
| month_anchor | hard | direct multi-month shift, leap years, bounds |
| quiet_ranges | hard | half-open union/complement, validate empty horizon |
| escaped_fields | medium | escape parity, empty fields, dangling escape |
| carry_round | hard | prefix half-even rounding, cancellation, precision |
| recent_keys | easy | hit promotion, zero capacity, bool validation |
| braid_rows | medium | ragged alternating traversal, input mutation |
| stable_scores | medium | NaN/infinity/zero ties, stable order |
| near_duplicate | easy | inclusive distance, distinct indices, empty inputs |
| path_reduce | medium | relative unmatched parents, lexical paths |
| select_rank | easy | duplicates, invalid rank, no in-place sorting |
| dense_labels | easy | exact Unicode comparison, empty string |
| retry_slots | medium | inclusive budget, zeros, validation beyond stop |
| end_match | easy | longest Unicode suffix, empty vs no match |
| local_midnights | hard | DST, folds, skipped dates, UTC instant comparison |
| aligned_batches | hard | negative global offsets, short first batch |

Seven easy, seven medium and six hard coding tasks. Five separate API items
exercise `Tool.wrap`, named `Plan` composition with `MockAgent`, `Envelope`
construction, `Agent(verify=..., max_verify=...)`, and `Step` sentinel context.
The pipeline API test observes actual `Plan` and `MockAgent` execution; merely
returning the expected string does not pass.

## Limitations and interpretation

- Small n (20 coding + 5 API items), one run per task/arm and two providers. Wilson
  intervals are descriptive; these hand-selected tasks are not a random sample of
  all programming work. Tasks are related, so independence assumptions are weak.
- A/C and B/D use the same specs but independent initial generations. “Fixed”
  and “broken” are paired outcome transitions, not proof the reviewer caused that
  exact candidate's change. There is no shared initial candidate, self-review arm,
  same-provider review control or matched extra-compute control. Extra retries,
  model randomness and check usage can explain changes as well as provider diversity.
- No broad hallucination claim follows from small Python coding and API-use tasks.
  Passing finite tests is not proof of full correctness; the suite can miss bugs.
  The reference solutions check test consistency, not exhaustive oracle validity.
- Engine/tool environments differ despite matched Python checks and prompts. The
  real package is available for API checks. Subscription limits may interrupt runs.
- Default Codex model is explicitly `gpt-6.1-sol`, reasoning `high`; Claude defaults
  to the moving `sonnet` alias. Pin `--claude-model` to an available dated ID for a
  later experiment, using a new JSONL if configuration changes. Compare reported
  model metadata and execution dates: subscriptions/backends can change silently.
- Publishing makes future training contamination possible. This first run avoids
  copying known benchmarks but cannot prove zero overlap with private training data.
- Two paired comparisons per category are exploratory; no multiple-comparison
  correction or preregistered significance claim. Do not market improvements from
  mock data or an unpaired baseline. Keep the homepage claim unsubstantiated until
  live reviewed arms are measured, and publish any negative result.

The initial live baseline scored **20/20 coding** (Wilson 95% CI 83.9%-100%) and
**2/5 API** (11.8%-76.9%) on 2026-10-11 using `gpt-6.1-sol`, high reasoning,
Codex CLI 0.160.0 and LazyBridge 1.8.0. All 25 candidates were regraded after the
isolation repair; **no pass/fail outcome changed**. The coding baseline has a
ceiling effect: C cannot improve its pass count on this run. There is no measured
cross-provider review benefit or refutation yet, since B/C/D have not run.

See [results/NOTES.md](results/NOTES.md) for execution details and failures,
[results/arm_a_regraded.md](results/arm_a_regraded.md) for the corrected summary,
and [results/arm_a_regraded.jsonl](results/arm_a_regraded.jsonl) for appendable data.
The original [results/arm_a.jsonl](results/arm_a.jsonl) and
[results/generation_protocol.zip](results/generation_protocol.zip) are retained.
Arms B/C/D are intentionally left for the operator after the Claude quota reset.

## Verification

```powershell
python -m pytest tests/unit/test_cross_review_benchmark.py -q
python -m ruff check examples/cross_review_benchmark tests/unit/test_cross_review_benchmark.py
python -m ruff format --check examples/cross_review_benchmark tests/unit/test_cross_review_benchmark.py
python -m pytest
```

The harness contract tests were written and run before implementation (initial
failure: missing benchmark module). They exercise all 100 mock task/arm records,
all 25 reference solutions, resume protection, grader timeouts/syntax errors,
statistics, a quota stop during verification, namespace isolation, future imports
in own checks, and regrading provenance. The repository's default full
suite excludes explicitly marked `live` and `heavy_render` tests.
