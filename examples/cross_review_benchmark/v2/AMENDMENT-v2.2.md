# Protocol amendment v2.2

The owner identified a ceiling problem: if either model already solves nearly every task alone, a stronger reviewer acts as an oracle and cross-review is uninformative. This amendment tests whether two imperfect models together beat the better standalone model, and at what cost.

Before any v2.2 row, the primary comparisons are C and D against max(A,B), and against the same-provider controls E and F respectively. A and B are standalone Codex and Claude Code; C and D exchange reviewers; E is Codex writing and reviewing, and F is Claude Code writing and reviewing. All reviewed arms use the same prompts, tools, verification loop and max_verify. Use `--codex-model gpt-6-luna --claude-model claude-haiku-5-5` for v2.2 runs, with Codex effort explicitly selected on the CLI. The existing CLI defaults are unchanged.

On tasks completed by both A and B, max(A,B) is the larger standalone pass rate; the task-wise oracle passes if either standalone arm passes. Report their failure overlap (both fail, only A fails, only B fails). Report C, D, E and F on tasks also completed by A and B, with denominators; comparisons require a common task set. Errors count as failures, and pending rows are excluded. Wilson intervals describe each arm's observed pass rate; they do not establish a paired effect.

Sum all recorded writer and reviewer calls. Claude Code uses its reported USD cost; Codex uses the LazyBridge OpenAI list-price table and token counts, charging uncached input, cached input and output separately. If cached tokens are unavailable, charge all input at the uncached rate and label the result an upper bound. Missing usage or an unknown price makes the cost unknown. Divide total arm cost by completed rows and by passed tasks; the latter is undefined with zero passes. Costs are API-equivalent estimates; subscription users pay a flat fee.

Run `python -m examples.cross_review_benchmark.cost_report RESULTS.jsonl` on a coherent v2.2 results file. Do not mix protocols or model configurations in that file; the harness enforces its protocol fingerprint when resuming.
