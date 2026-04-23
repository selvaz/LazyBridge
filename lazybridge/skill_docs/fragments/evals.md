## signature
EvalCase(
    input: str,
    check: Callable[[str], bool] | Callable[[str, Any], bool],
    expected: Any = None,
    description: str = "",
)

EvalSuite(*cases: EvalCase)
suite.run(agent) -> EvalReport
await suite.arun(agent) -> EvalReport

EvalReport(results: list[EvalResult])
  .total, .passed, .failed, .errors

# Built-in checks
exact_match(expected: str) -> Callable
contains(substring: str) -> Callable
not_contains(substring: str) -> Callable
min_length(n: int) -> Callable
max_length(n: int) -> Callable
llm_judge(agent: Agent, criteria: str) -> Callable   # cheap Agent as judge

## rules
- ``EvalSuite`` feeds each ``EvalCase.input`` to the agent, captures
  the output text, runs ``check(output)``. ``expected`` is metadata
  only — checks are closed over their expected values.
- ``check`` can return ``bool`` or raise. Raising counts as an error,
  not a failure.
- ``llm_judge`` accepts an Agent and a string policy; the judge
  evaluates output and must respond with ``"approved"`` to pass.

## example
```python
from lazybridge import Agent, EvalCase, EvalSuite, contains, llm_judge

bot   = Agent("claude-opus-4-7", system="You are a helpful assistant.")
judge = Agent("claude-opus-4-7", name="judge",
              system='Respond "approved" or "rejected: <reason>".')

suite = EvalSuite(
    EvalCase("What's the capital of France?",
             check=contains("Paris")),
    EvalCase("Write a poem about bees.",
             check=llm_judge(judge,
                 "Output must be a poem of at least 4 lines mentioning bees.")),
    EvalCase("hello",
             check=lambda out: len(out) < 500,
             description="brevity check"),
)

report = suite.run(bot)
print(report)                      # "2/3 passed (66%)"
assert report.passed == report.total, [r.case.input for r in report.results if not r.passed]
```

## pitfalls
- ``llm_judge`` costs tokens on every run; use a ``cheap`` tier agent.
- Evals test the agent's text output (``Envelope.text()``), not the
  typed payload. If you're evaluating a structured-output agent, the
  check sees the JSON serialisation.
- ``EvalSuite.run`` is synchronous; use ``arun`` in async test harnesses.

