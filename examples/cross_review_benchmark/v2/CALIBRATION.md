# Calibration protocol registered before generation

The target for each category separately is 40–70%: 8–14 of 20 coding tasks,
10–17 of 25 API tasks. These are design targets, not guaranteed outcomes.
Only Codex arm A may run during calibration. No Claude engine is constructed
and no Claude quota telemetry is queried. Default quota stop remains 95%.

Initial design: twenty original harder coding tasks and twenty new API items
plus the five unchanged v1 API anchors. Validate evaluator syntax, reference
solutions and negative controls before requesting model responses. This
pre-generation validation is not model calibration.

Run the whole initial 45-item set once, using the existing fixed prompts,
seed 20261011, gpt-6.1-sol, high reasoning and the same engine/check budgets.
Archive the exact three task JSON files before each calibration call. All
completed records, including errors, remain in their original append-only
JSONL; publish each run's summary, snapshot, fingerprint and dates.

If either category misses its target, allow at most TWO design revisions
(at most three complete calibration runs including the initial design).
Revise the task specification/requirements, never relax or change a hidden
grader to accept a model's answer. Retain the five v1 API anchors unchanged.
New requirements may have additional prewritten tests and reference code;
retain all existing tests when the specification is strengthened. Record
the changed task IDs, requirements and rationale before the next run.
Do not select individual candidates or seed-search to improve scores.
Stop revisions early only when both categories reach the target. Otherwise
choose the latest design at the revision limit and explicitly report misses.

Freeze the chosen task files with SHA-256 and a UTC registration timestamp,
record the calibrated scores, then run a FRESH arm A on the frozen set.
That fresh run is the paired baseline for later C, distinct from calibration.
Do not redesign tasks after freezing or after seeing any B/C/D outputs.
Calibration is deliberately adaptive on one model; it is not independent
evidence of cross-provider review, and final arm A can deviate from target.
