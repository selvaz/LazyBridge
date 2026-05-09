# Evals

A thin pytest-shaped harness for testing an agent's *output behaviour*.
Define cases, run them through the agent, get a pass/fail report.
Pair deterministic checks (`exact_match`, `contains`) with an
LLM-judge for grading subjective outputs.

## Signature

```python
from lazybridge.ext.evals import (
    EvalCase,
    EvalSuite,
    EvalReport,
    EvalResult,
    # built-in checks
    exact_match,
    contains,
    not_contains,
    min_length,
    max_length,
    llm_judge,
)


EvalCase(
    input,                         # str — task to feed the agent
    check,                         # callable: (output) or (output, expected) -> bool
    expected=None,                 # optional metadata for the check
    description="",                # human-readable label for the report
)


EvalSuite(*cases)
suite.run(agent)                   # returns EvalReport
await suite.arun(agent)            # async variant


EvalReport(results)
  .total                           # int
  .passed                          # int
  .failed                          # int
  .errors                          # int (checks that raised)
```

### Built-in check builders

| Builder | Returns a check that |
|---|---|
| `exact_match(expected)` | `True` iff the output equals `expected` |
| `contains(substring)` | `True` iff `substring in output` |
| `not_contains(substring)` | `True` iff `substring not in output` |
| `min_length(n)` | `True` iff `len(output) >= n` |
| `max_length(n)` | `True` iff `len(output) <= n` |
| `llm_judge(agent, criteria)` | `True` iff the judge agent replies `"approved"` |

## Synopsis

`EvalSuite` is a deterministic test harness for "given this input,
the agent must produce an output that satisfies this check". Each
`EvalCase` carries an input string, a check callable, and an
optional `expected` value (carried for reporting, not enforced —
the check closes over its expected value).

The suite feeds each input to the agent, captures
`Envelope.text()`, and runs `check(output)`. A check that returns
`False` counts as a failure; one that raises counts as an error.
Both surface in the `EvalReport` so you can see the difference
between "agent produced an unexpected answer" and "the check itself
crashed".

`llm_judge(judge, criteria)` returns a check that calls the judge
agent with the candidate output and the criteria; it passes only
when the judge replies with a string starting with `"approved"`
(case-insensitive).

## When to use it

- **Behaviour regression.** "After this prompt change, do my five
  canonical inputs still produce answers that mention the right
  city / contain the right phrase / pass the policy judge?"
- **CI gates on agent quality.** Run an `EvalSuite` against a
  staging agent before promotion; gate on `report.passed ==
  report.total`.
- **Subjective grading at scale.** When deterministic substring
  checks aren't enough, an `llm_judge` lets you encode policies in
  English ("must be a poem of at least 4 lines mentioning bees")
  and grade in batch.

## When NOT to use it

- **Unit-testing internal helpers.** That's pytest's job. Use
  `EvalSuite` when the unit under test is the agent's *response*,
  not a function.
- **Runtime gating of every call.** `EvalSuite` is for offline /
  CI batches. For live "judge every output and retry if rejected"
  semantics use [`verify=`](verify.md).
- **High-frequency probes.** Each case is a full agent run. If you
  need millisecond-level checks, write a deterministic check
  outside the LLM path.

## Example

```python
from lazybridge import Agent, LLMEngine
from lazybridge.ext.evals import (
    EvalCase,
    EvalSuite,
    contains,
    llm_judge,
)


bot = Agent(
    engine=LLMEngine(
        "claude-opus-4-7",
        system="You are a helpful assistant.",
    ),
)
judge = Agent(
    engine=LLMEngine(
        "claude-opus-4-7",
        system='Respond "approved" or "rejected: <reason>".',
    ),
    name="judge",
)


suite = EvalSuite(
    EvalCase(
        "What's the capital of France?",
        check=contains("Paris"),
    ),
    EvalCase(
        "Write a poem about bees.",
        check=llm_judge(
            judge,
            "Output must be a poem of at least 4 lines mentioning bees.",
        ),
    ),
    EvalCase(
        "hello",
        check=lambda out: len(out) < 500,
        description="brevity check",
    ),
)


report = suite.run(bot)
print(f"{report.passed}/{report.total} passed ({report.passed / report.total:.0%})")

# In CI:
assert report.passed == report.total, [
    r.case.input for r in report.results if not r.passed
]
```

## Pitfalls

- **`llm_judge` costs tokens on every case.** Use a cheap-tier
  agent as the judge (`claude-haiku-...` / `gpt-4o-mini` /
  equivalent). One judge across all cases is fine; per-case judges
  multiply the cost.
- **Evals see `Envelope.text()`, not the typed payload.** When
  testing a structured-output agent (`output=PydanticModel`) the
  check receives the JSON serialisation of the payload, not the
  model instance. Write check predicates against the JSON shape
  accordingly.
- **`EvalSuite.run` is synchronous.** It uses
  `asyncio.run`-style internals to drive the agent, so calling it
  inside an existing event loop fails. Use `await suite.arun(agent)`
  in async test harnesses (pytest-asyncio, FastAPI startup hooks,
  etc.).
- **A check that raises is an error, not a failure.** The report
  separates the two — `report.errors` is for checks that crashed,
  `report.failed` is for checks that returned `False`. When
  surfacing the report in a log, surface both counters.
- **`expected=` is metadata only.** It's stored on the case for
  reporting; the check is responsible for using it. Most built-in
  builders close over the expected value at construction time
  (`contains("Paris")` already encodes "Paris" in the closure).

## See also

- [verify=](verify.md) — runtime sibling: rather than gating in CI,
  the judge runs on every call and retries the agent up to
  `max_verify` times with feedback.
- [Guards](guards.md) — hard-gate filtering at run time; use
  guards for "this output must never pass" and evals for "this
  output should usually be good".
- [Agent](../basic/agent.md) — the surface every eval runs against.
- *Guides → Full → Testing (MockAgent)* (Phase 3) — deterministic
  agent doubles for unit tests of orchestration code that
  *contains* an agent (rather than testing the agent's response
  itself).
