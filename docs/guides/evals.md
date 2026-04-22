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
    name="judge",                      # label used in session.usage_summary()
)

# EvalCase(input_prompt, check=<predicate>, description=<label for the report>)
#   - The first positional arg is the prompt fed to the agent under test.
#   - check=  is a Callable[[str], bool] applied to Envelope.text().
#             contains("Paris") builds one from the library; you can also
#             pass a lambda or llm_judge(...) for LLM-graded checks.
#   - description=  is a free-text label that appears in the report output;
#                   no impact on grading, just makes failures easier to read.
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

## `arun()` — concurrent case execution

`suite.run(agent)` feeds cases to the agent sequentially.  For large
suites or slow agents, `await suite.arun(agent)` runs every case
concurrently via `asyncio.gather`.  Same report shape, often
dramatically faster end-to-end.

```python
# What this shows: dropping a 50-case suite from ~50 * median_latency
# to ~median_latency by running them in parallel. Each case is
# independent (no shared state) so concurrent execution is safe by
# construction.
# Why arun: run() is a simple loop — good for 3-5 cases or
# debugging. arun() is the production shape — CI should always use
# arun() unless the provider's concurrency budget is very tight.

import asyncio
from lazybridge import EvalSuite, EvalCase, contains

suite = EvalSuite(
    EvalCase("what is 2+2?", check=contains("4")),
    EvalCase("capital of France", check=contains("Paris")),
    # ... 48 more
)

report = asyncio.run(suite.arun(agent))
print(f"{report.passed}/{report.total} passed")
```

If you're rate-limited by the provider, wrap the agent in an
`asyncio.Semaphore` adapter or use `Agent.parallel`'s
`concurrency_limit=` — `arun` itself has no throttle knob.

## Custom check signatures

Built-in checks (`contains`, `exact_match`, `min_length`, etc.) cover
the common cases.  For anything richer, write a plain callable —
LazyBridge accepts two signatures and dispatches based on arity.

```python
# What this shows: the two supported check shapes. Pick whichever
# fits the assertion you want to make.
# Why two: single-arg is the 80% case (the output is all you need to
# grade). Two-arg is the "compare against expected" case where the
# test data lives on EvalCase.expected and the check keeps its
# comparison logic out of the case body.

from lazybridge import EvalCase

# Shape 1: check(output) -> bool. Simplest form.
def starts_with_bullet(output: str) -> bool:
    return output.lstrip().startswith("- ") or output.lstrip().startswith("* ")

EvalCase("list three fruits", check=starts_with_bullet)

# Shape 2: check(output, expected) -> bool.
# LazyBridge detects the two-arg signature and passes EvalCase.expected
# as the second argument. Useful when you have parametric expectations.
def jaccard_similar(output: str, expected: str, threshold: float = 0.7) -> bool:
    a = set(output.lower().split())
    b = set(expected.lower().split())
    return len(a & b) / len(a | b) >= threshold

EvalCase("summarise X", expected="the canonical summary of X",
         check=jaccard_similar)
```

Raising from inside a check counts as an **error**, not a failure.
`EvalReport.errors` lists those separately from `.failed` so a genuine
bug in the check code doesn't silently count as a failing test.

## `llm_judge` in depth

`llm_judge(agent, criteria)` returns a check callable that delegates
grading to a cheap Agent.  The judge must respond with a verdict
starting with `"approved"` (case-insensitive) to pass — any other
response is a fail.  The criteria string is the judge's policy.

```python
# What this shows: wrapping a Haiku-tier agent as a pass/fail grader
# for subjective "did the output meet these criteria?" checks.
# Why keyword "approved" (not "yes" or "pass"): unique enough that
# the judge is unlikely to produce it accidentally in an otherwise-
# negative verdict, but common enough that Anthropic/OpenAI models
# follow the instruction without fighting the prompt.

from lazybridge import Agent, LLMEngine, EvalCase, llm_judge

# Make the judge cheap and specific. One criterion per judge beats
# one catch-all judge — the feedback is better and the cost is lower.
judge = Agent(
    engine=LLMEngine(
        "claude-haiku-4-5",
        system='Respond "approved" or "rejected: <reason>". '
               'Be strict.',
    ),
    name="tone-judge",
)

EvalCase(
    "explain gravity to a 5-year-old",
    check=llm_judge(
        judge,
        criteria=(
            "The explanation must use everyday analogies, avoid the "
            "word 'force' and any term a 5-year-old wouldn't know, "
            "and be at most 3 sentences."
        ),
    ),
)
```

Running the same case 10x with an LLM judge is still deterministic
*enough* for CI if `temperature=0` is set on the judge's engine —
LazyBridge defaults the judge's sampling to the provider's default
but you can pass `temperature=0.0` on `LLMEngine` for near-zero
variance.

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
