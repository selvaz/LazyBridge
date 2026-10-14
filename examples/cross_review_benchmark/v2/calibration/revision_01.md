# Design revision 1, registered before run_02

Run_01 completed all 45 tasks: coding 20/20 (100%, Wilson 95% CI
83.9–100%); API 11/25 (44%, CI 26.7–62.9%). No engine errors.
The coding set still has a ceiling; API is within the 40–70% target.

Change ONLY these eight coding specifications, introducing actual new
requirements rather than changing acceptance of an existing output:

- coding_03_capped_apportion: two-level grouped allocation, including
  zero-cap positive weights at group level.
- coding_06_ttl_lru: nested transactions; rollback restores LRU/state but
  not clock or observations, and immediately expires restored stale entries.
- coding_07_nested_expansion: slicing enormous repeated outputs without
  allocating intermediate expansions; thousands-digit counts.
- coding_09_dependency_waves: resource capacities; maximum-cardinality
  feasible ready subsets before lexicographic tie breaking.
- coding_12_rational_expression: bindings and lazy conditional arithmetic,
  while eagerly validating syntax, names and arities in unselected branches.
- coding_13_priority_overlay: closed/open endpoint mode, singleton points,
  label merging and uncovered shared boundaries.
- coding_16_folded_spans: mark elision and multiple original boundary pairs
  when transformed units become empty.
- coding_18_coupon_route: ordered required node subsequences and multiple
  nonconsecutive discounted edges.

All original hidden test functions/assertions are retained verbatim; append
tests for the new requirements. All 45 reference implementations for this
design were validated before model generation. All 25 API specs/tests stay
unchanged, including the five anchors. No grader acceptance was relaxed to
match an observed answer. Run_02 is a full new 45-task Codex-only draw, with
unchanged prompts, model, reasoning effort, order and budgets.

If this still exceeds 70% coding, the final permitted revision will add
exact rational-quotient rounding, symbolic rational-function normalization,
glob star capture tie breaking, BOM-aware UTF-16 byte chunking, and revision
undo replay. These five features are selected by functional coverage, before
seeing run_02 outcomes. If API falls below 40%, add actual public API name
clues to the consecutive block of NEW items 06–10, with identical graders;
if it exceeds 70%, retain the harder original API wording. No task candidates
will be selected by success, and every complete run will be published.
