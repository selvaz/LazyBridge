# Frozen v2 arm A audit

3 complete calibration draws plus one fresh frozen draw: 180
live arm A records, 180 recorded writer calls,
0 reviewer calls and 0 own checks.
2 records contain provider-call errors. Hidden-test failures
are distinct and remain failures in all pass-rate denominators.

Engine/model-call failure details:

- `api_07_named_history`: `RuntimeError: `, writer call 180.08s; hidden grade available: False.
- `api_10_checked_run`: `RuntimeError: `, writer call 180.07s; hidden grade available: False.

The history and checked-run errors have empty diagnostic messages and occur
at the 180-second request budget, consistent with request timeouts. Their
precise underlying exceptions were not preserved by the engine wrapper.
Do not label them API hallucinations. No failed record was retried or replaced.

All quota readings are Codex-only; maximum observed used percent 24.0,
stop threshold 95%. No live Claude constructor/call or Claude quota read was used.
The installed Claude SDK version is metadata only.

The v1 result-file copies were checked byte-for-byte against the original
results. Every earlier hidden test string is a prefix of the next design
(including all existing assertions); all 25 API specs/tests/references remained
unchanged across calibration. Final reference solutions passed before generation.

Freeze timestamp `2026-10-11T23:38:29.097123+00:00`; fresh baseline starts
`2026-10-11T23:39:08.707856+00:00`. Task SHA-256:
`660d9ccf90a01ec6ba59397c250c4df11ca632e25a363642bcdfe8c021d04b8a`.

The calibration difficulty target was missed for coding. All runs are
published, the latest design was selected under the predeclared rule, and no
further design revisions were made. Small validation failures and algorithmic
errors count equally; consult each stored traceback before interpreting severity.

The observed coding_16 failure is specifically the exception class for
ignore_marks=1: TypeError instead of the specified ValueError. The algorithmic
assertions preceding that check passed. This score does not demonstrate a
Unicode matching algorithm failure.

Fresh baseline failures:

- `api_16_registry_bind`: Hidden pytest failed
- `api_04_review_agent`: Hidden pytest failed
- `api_06_named_step`: Hidden pytest failed
- `api_17_registry_rename`: Hidden pytest failed
- `api_21_task_envelope`: Hidden pytest failed
- `api_20_error_envelope`: Hidden pytest failed
- `api_13_store_cas`: Hidden pytest failed
- `api_05_step_context`: Hidden pytest failed
- `api_08_checkpoint_plan`: Hidden pytest failed
- `api_07_named_history`: RuntimeError:
- `api_09_plan_roundtrip`: Hidden pytest failed
- `api_10_checked_run`: RuntimeError:
- `coding_16_folded_spans`: Hidden pytest failed
- `api_18_alias_engine`: Hidden pytest failed
- `api_11_verified_alias_tool`: Hidden pytest failed
- `api_24_engine_policy`: Hidden pytest failed
- `api_02_mock_plan`: Hidden pytest failed

Arms B/C/D and both paired comparisons remain pending. This is a baseline,
not evidence that cross-provider review helps or hurts.
