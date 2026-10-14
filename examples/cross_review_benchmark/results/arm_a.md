# Cross-provider review benchmark results

Mock data: False

Errors count as failures. Only completed task/arm records enter denominators. Missing arms are pending.

| Category | Arm | Pass / completed | Pass rate | Wilson 95% CI |
|---|---|---:|---:|---:|
| coding | A | 20/20 | 100.0% | 83.9% to 100.0% |
| coding | B | pending | — | — |
| coding | C | pending | — | — |
| coding | D | pending | — | — |
| api | A | 2/5 | 40.0% | 11.8% to 76.9% |
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

- A / api_02_mock_plan: Hidden pytest failed
- A / api_04_review_agent: Hidden pytest failed
- A / api_05_step_context: Hidden pytest failed

This table alone does not establish an effect of cross-provider review. See README for design limitations.
