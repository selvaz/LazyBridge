# Cross-provider review benchmark results

Mock data: False

Errors count as failures. Only completed task/arm records enter denominators. Missing arms are pending.

| Category | Arm | Pass / completed | Pass rate | Wilson 95% CI |
|---|---|---:|---:|---:|
| coding | A | 19/20 | 95.0% | 76.4% to 99.1% |
| coding | B | pending | — | — |
| coding | C | pending | — | — |
| coding | D | pending | — | — |
| api | A | 9/25 | 36.0% | 20.2% to 55.5% |
| api | B | pending | — | — |
| api | C | pending | — | — |
| api | D | pending | — | — |

| Category | Pair | Paired n | Fixed by review | Broken by review | McNemar exact p |
|---|---|---:|---:|---:|---:|
| coding | A vs C | 0 | pending | pending | — |
| coding | B vs D | 0 | pending | pending | — |
| api | A vs C | 0 | pending | pending | — |
| api | B vs D | 0 | pending | pending | — |

Failed tasks:

- A / api_16_registry_bind: Hidden pytest failed
- A / api_04_review_agent: Hidden pytest failed
- A / api_06_named_step: Hidden pytest failed
- A / api_17_registry_rename: Hidden pytest failed
- A / api_21_task_envelope: Hidden pytest failed
- A / api_20_error_envelope: Hidden pytest failed
- A / api_13_store_cas: Hidden pytest failed
- A / api_05_step_context: Hidden pytest failed
- A / api_08_checkpoint_plan: Hidden pytest failed
- A / api_07_named_history: RuntimeError:
- A / api_09_plan_roundtrip: Hidden pytest failed
- A / api_10_checked_run: RuntimeError:
- A / coding_16_folded_spans: Hidden pytest failed
- A / api_18_alias_engine: Hidden pytest failed
- A / api_11_verified_alias_tool: Hidden pytest failed
- A / api_24_engine_policy: Hidden pytest failed
- A / api_02_mock_plan: Hidden pytest failed

This table alone does not establish an effect of cross-provider review. See README for design limitations.
