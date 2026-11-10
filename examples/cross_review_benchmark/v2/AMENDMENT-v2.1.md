# Protocol amendment v2.1 (10/11/2026)

Registered after an operational pilot, before any v2.1 scored row.

## Why
The first B/C/D pilot under the frozen v2 protocol (hash 13166f9f, 10/11/2026 02:09-02:23 PST) produced 6 rows: 5 failed with
`Claude Code returned an error result: Reached maximum number of turns (12)`, for Claude as writer and as reviewer; 1 passed.
Diagnosis (independent review by Codex plus local measurement):
- `--check-timeout` was 5 s while a cold `import lazybridge` took ~7.7 s on this machine, so every own check that imported the
  API timed out (27 check timeouts in 6 rows); Claude retried checks until the turn cap.
- In Claude Code every `python_check` call consumes a turn; `max_turns=12` was a Claude-only ceiling with no Codex equivalent.
- Windows child processes printed with cp1252 (6 results with UnicodeEncodeError).
- `--claude-model` defaulted to the bare alias `sonnet`, which the current Claude Code CLI resolves to Sonnet 5, not Sonnet 5.5.
Arm A never used own checks (0 in 180 rows), so it was not exposed to these problems.

## Changes
- `--check-timeout` default 5 s -> 20 s (same as `--test-timeout`).
- `python_check` children run with `-X utf8`.
- New `--claude-max-turns` (default 40), part of the protocol hash and of `model_config`; a provisional safety ceiling, not an
  equal compute budget.
- `--claude-model` default `claude-sonnet-5-5`.
Tasks, hidden tests, graders and Codex parameters are unchanged.

## Analysis rules
- The 6 pilot rows (old hash) are reported but excluded from the v2.1 primary analysis, including the one pass.
- Arm A is re-drawn under v2.1 (the Codex CLI changed from 0.160.0 to 0.166.0-alpha since the October draw); the October arm A
  (28/45) stays as historical evidence.
- Under v2.1 every engine error, including a turn cap, counts as a failure.
- Smoke before the scored run (throwaway file): api_04 and api_16, arms B and C, all four passed; 6 of 14 own checks still hit
  the 20 s check timeout (slow cold import of the API copy).

## Known limitation found after the run started (10/11/2026)
Codex could not call `python_check` in v2.1: with Codex CLI 0.166 dynamic tools go through the code-mode host, which the
harness disabled. Codex writers (A, C) and the Codex reviewer (D) therefore had no own checks, while Claude Code did.
Do not read v2.1 A-vs-B or C-vs-D differences as model quality alone. Fixed in v2.2 (AMENDMENT-v2.2.md).
