# Initial Codex baseline — 2026-10-11

Only arm A was run live. No ClaudeCodeEngine was constructed or called by the
benchmark and no Claude quota telemetry was queried. B/C/D remain pending.

| Category | Passed | Rate | Wilson 95% CI |
|---|---:|---:|---:|
| Original Python coding tasks | 20/20 | 100% | 83.9%-100% |
| Separate LazyBridge API items | 2/5 | 40% | 11.8%-76.9% |

No paired comparison or McNemar p-value is available yet. The perfect coding
baseline is a ceiling effect for A/C on these items. It suggests that this task
set is too easy to measure coding improvements for this particular writer/run;
it does not establish or refute a cross-provider review effect. Future harder
suites should be specified in advance and published as distinct experiments.

The three failures were executable API-use mistakes:

- `api_02_mock_plan`: the generated helper omitted required `task` when calling
  `Agent.run`. It also constructed Agent without the Plan engine and used
  `envelope.text` as an attribute rather than calling it. The observed first
  failure was the missing required `task`, before any provider execution.
- `api_04_review_agent`: generated code searched for nonexistent verification
  configuration types (`OutputVerification`, `VerificationConfig`, etc.) and
  raised `TypeError`, instead of supplying the judge directly to `verify=`.
- `api_05_step_context`: generated code searched for guessed callable sentinel
  names and raised `ImportError`; it did not use `from_prev` and `from_step`.

Full candidate code and hidden pytest traces are in the logs. Do not substitute
repaired solutions for these measured outputs. Hidden tests were never fed back.

Execution metadata:

- Model requested/reported: `gpt-6.1-sol`; reasoning effort `high`.
- Subscription runtime: Codex CLI `0.160.0`, via LazyBridge CodexEngine.
- LazyBridge `1.8.0`; Python `3.11.13`; pytest `9.1.1`.
- Windows `10.0.26100`. Installed Claude Agent SDK `0.2.129` was not used for a
  live engine call. Its presence is recorded as environment metadata only.
- Seed/order: `20261011`; one writer invocation per task; no review calls.
- First start: `2026-10-11T20:29:57.103779+00:00`.
- Last finish: `2026-10-11T20:40:09.154733+00:00`.
- Sum of recorded task wall time: `590.061` seconds. The UTC execution span also
  includes preflight telemetry outside task timers (about 10 minutes 12 seconds).
- Quota brake: 95%; available Codex telemetry reported 19% throughout; no stop.
- Model own checks: zero logged `python_check` calls.

After generation, an audit demonstrated that `from candidate import *` could
overwrite the test module's datetime helpers and falsely fail a correct module-
qualified implementation. The corrected grader imports only the requested
function. A regression test first reproduced that defect and then passed after
the repair. No specifications, test inputs or expected answers were changed.
The Python check tool also now compiles model code separately so leading
`from __future__` imports work; zero own checks in this run means that repair
cannot have affected these generations.

All 25 saved candidates were regraded, with no additional model calls and no
outcome changes. The original immutable [arm_a.jsonl](arm_a.jsonl), its summary
[arm_a.md](arm_a.md), and the original source files in
[generation_protocol.zip](generation_protocol.zip) remain available. The archive
contains the five original example modules; restore them into the example folder
of the same package source revision to reproduce the original protocol hash.
The package Python source was unchanged throughout the experiment.

[arm_a_regraded.jsonl](arm_a_regraded.jsonl) retains original model calls and
generation timestamps, adds `generation_protocol_hash` and a `regrade` object
with prior grades/errors and repair reason, and uses the corrected protocol hash
for compatible later arms. A regraded row's `wall_seconds` remains its original
task duration; additional grading time appears under `regrade.wall_seconds`.
Both hashes are included for audit. The corrected results are the baseline to
extend when the operator has Claude quota again.

Repository validation: the full default pytest suite was run once after the
feature commit: 3,093 passed, 19 skipped, 7 deselected, with one failure in the
standalone import check for `regrade.py`. All ten benchmark contract tests passed.
The relative import causing that failure was corrected to an absolute package
import; the exact failed test then passed on its targeted rerun. The full suite
was not repeated. Repository-wide Ruff lint and format checks passed (309 Python
files). This import-only correction changes neither generation nor grading and
does not invalidate the recorded protocol hashes.
