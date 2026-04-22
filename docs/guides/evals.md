# EvalSuite

`EvalSuite` is LazyBridge's minimum-viable test framework for agents.
It's unit testing, not benchmarking — small, fast, runnable in CI.

You write `EvalCase`s with an input prompt and a check function. The
built-in checks cover the common cases (exact match, substring, length
bounds). For anything richer — "is this factually correct?", "is the
tone appropriate?" — `llm_judge` wraps a cheap agent as a pass/fail
oracle.

Pattern: keep a suite alongside each agent you ship, run it on every
refactor, track `report.passed / report.total` over time. Combined
with `llm_judge`, it's also a rudimentary regression guard against
prompt changes.

## Example

```python
from lazybridge import Agent, LLMEngine, EvalCase, EvalSuite, contains, llm_judge

# System prompts live on the engine, not the Agent constructor.
bot = Agent(engine=LLMEngine("claude-opus-4-7",
                             system="You are a helpful assistant."))
judge = Agent(
    engine=LLMEngine("claude-opus-4-7",
                     system='Respond "approved" or "rejected: <reason>".'),
    name="judge",
)

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

## Pitfalls

- ``llm_judge`` costs tokens on every run; use a ``cheap`` tier agent.
- Evals test the agent's text output (``Envelope.text()``), not the
  typed payload. If you're evaluating a structured-output agent, the
  check sees the JSON serialisation.
- ``EvalSuite.run`` is synchronous; use ``arun`` in async test harnesses.

!!! note "API reference"

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

!!! warning "Rules & invariants"

    - ``EvalSuite`` feeds each ``EvalCase.input`` to the agent, captures
      the output text, runs ``check(output)``. ``expected`` is metadata
      only — checks are closed over their expected values.
    - ``check`` can return ``bool`` or raise. Raising counts as an error,
      not a failure.
    - ``llm_judge`` accepts an Agent and a string policy; the judge
      evaluates output and must respond with ``"approved"`` to pass.

## See also

[guards](guards.md), [verify](verify.md), [agent](agent.md)
