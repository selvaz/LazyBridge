# Design revision 2, registered before run_03

Run_02 completed all 45 tasks: coding 19/20 (95%, Wilson 95% CI
76.4–99.1%); API 11/25 (44%, CI 26.7–62.9%). No engine errors.
Coding remains above target. API remains within target, so all API specs,
tests and references stay unchanged. The proposed API name clues are NOT used.

Apply the five coding features selected in revision_01.md before observing
run_02 outcomes:

- coding_10_significant_round: exact rational-quotient rounding without
  decimal-context dependence, including thousands-digit numerator/denominator.
- coding_11_escaped_glob: optional star capture spans, adjacent stars,
  lexicographically minimal capture-length tuples across matching parses.
- coding_12_rational_expression: optional symbolic rational functions in x,
  exact polynomial coefficients, GCD cancellation and monic denominators,
  including negative powers and constant expressions as exponents.
- coding_15_utf8_rechunk: optional BOM-aware UTF-16, preserving byte order,
  all original bytes and whole supplementary code points without extra BOMs.
- coding_19_revision_log: undo retained earlier ordinary events, replay the
  remaining winners in order, including insertion order and losing undos.

These are additional requirements in the visible specifications. Every prior
hidden test function/assertion is retained; additional tests cover these modes.
All 45 references for this final design passed before requesting run_03.
No acceptance rule was relaxed to match a model answer. The five v1 anchors
remain exact copies. Prompts, model, effort, order and budgets are unchanged.

Run_03 is the third and FINAL complete calibration draw. Freeze this latest
design regardless of its score; explicitly report any missed 40–70% target.
No further revisions, seed searches or individual candidate selection.
After freezing, make a fresh 45-item A draw for later paired comparisons.
No B/C/D generation has occurred.
