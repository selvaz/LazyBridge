# Testing — MockAgent & scripted inputs

CI should never pay for LLM tokens, never flap on provider outages, and
never wait on a human at a REPL.  `lazybridge.testing` is the set of
test doubles that replace those dependencies.  Import from
`lazybridge.testing`, not from the top-level package — these helpers
are for tests, not production.

Two things live here:

1. `MockAgent` — an `Agent` stand-in that returns canned responses and
   records every call.  Drop it into `tools=[...]`, `Plan(Step(target=...))`,
   `Agent.chain(...)` or `Agent.parallel(...)` exactly like a real
   agent.  Lets you unit-test pipeline *composition and data flow*
   without an API key.
2. `scripted_inputs` / `scripted_ainputs` — deterministic `input_fn` /
   `ainput_fn` providers for `SupervisorEngine` and `HumanEngine`.
   Feed the next line from a pre-canned list; exhaust the list and the
   loop stops.  No stdin prompting, no timeouts, no flakes.

## Why these exist

Real agents have three sources of nondeterminism: the model, the
network, and human input.  All three break test suites.  The framework
accepts test doubles through the **same interfaces** as the real
thing — `Agent` recognises the duck-typed `_is_lazy_agent` marker on `MockAgent`, the
Plan engine detects it, and `SupervisorEngine` accepts any callable as
`input_fn`.  That means no special "test mode"; you wire up a pipeline
exactly as you would in production and swap the implementations in
your fixtures.

## MockAgent — basic example

```python
# What this shows: piping two mock agents through a Plan to verify the
# second step receives the first step's payload, without any LLM call.
# Why this matters: exercising the from_prev sentinel and Envelope
# threading is 80% of what pipeline tests need to cover.

from lazybridge import Agent, Plan, Step
from lazybridge.testing import MockAgent

# Dict responses: keys are substrings checked against env.task in
# insertion order; "*" is the catch-all.  The first key whose substring
# appears in env.task wins; a missing key + missing "*" raises.
researcher = MockAgent(
    {"weather": "sunny", "market": "bullish", "*": "no data"},
    name="researcher",
    # default_input_tokens / default_output_tokens populate the mock's
    # Envelope.metadata so assertions about cost roll-up still work.
    default_input_tokens=50, default_output_tokens=30,
)

# Callable responses: receive the incoming Envelope and return whatever
# (str / BaseModel / Envelope / ErrorInfo / raised exception).  Use
# env.text() to stringify regardless of upstream output type.
writer = MockAgent(
    lambda env: f"Report based on: {env.text()}",
    name="writer",
)

plan = Plan(
    Step(target=researcher, task="weather today"),
    Step(target=writer),                    # from_prev (default) sentinel
)
env = Agent.from_engine(plan)("daily brief")

assert "sunny" in env.text()
assert researcher.call_count == 1
assert writer.call_count == 1
# nested_* buckets roll up from inner Envelope.metadata through Tool
# boundaries — useful when asserting observability invariants.
assert env.metadata.nested_input_tokens + env.metadata.nested_output_tokens > 0
```

What you verified: the plan ran, both agents were called exactly once,
the second saw the first's output (because `from_prev` is the default
sentinel on `Step`), and nested cost metadata propagated through the
tool boundary.  Zero API calls.

## List responses — simulating a sequence of outcomes

```python
# What this shows: driving a retry loop with a pre-scripted sequence
# ("failure, failure, success") to test verify= feedback handling.

from lazybridge.testing import MockAgent
from lazybridge.envelope import ErrorInfo

flaky = MockAgent(
    # Three successive calls yield: error, error, success.  The fourth
    # call raises StopIteration unless cycle=True.
    [
        ErrorInfo(type="RateLimit", message="429", retryable=True),
        ErrorInfo(type="RateLimit", message="429", retryable=True),
        "finally succeeded",
    ],
    name="flaky",
)

# Assert the consumer correctly retries on retryable errors without
# retrying past the third call.
```

## Error injection — asserting error-path code runs

```python
# What this shows: raising a real exception from the mock so tests can
# verify that an outer Agent converts it into Envelope.error and does
# not propagate up the call stack.

from lazybridge import Agent
from lazybridge.testing import MockAgent

# Passing a BaseException INSTANCE in responses raises it from run();
# contrast with ErrorInfo, which returns a clean error Envelope.
boom = MockAgent(RuntimeError("provider exploded"), name="boom")

wrapper = Agent("claude-opus-4-7", tools=[boom], name="wrapper")
# Engine-level exception handling wraps boom's raise into an error
# Envelope via Envelope.error_envelope(exc); wrapper sees a tool-error
# but keeps running.
```

## Assertions & recording

```python
# Every call is appended to mock.calls (a list of MockCall records).
mock = MockAgent("ok", name="m")
mock("one")
mock("two")

assert mock.call_count == 2
assert mock.last_call.env_in.task == "two"

# Structured assertion helpers:
mock.assert_called_with(task_contains="one")    # in any prior call
mock.assert_call_count(2)

# Clear state between tests:
mock.reset()
assert mock.call_count == 0
```

## Scripted inputs for HumanEngine / SupervisorEngine

```python
# What this shows: non-interactive end-to-end test of a SupervisorEngine
# pipeline.  Real production code calls input() or reads stdin; tests
# hand over a finite list of "typed" lines.

from lazybridge import Agent, Store
from lazybridge.ext.hil import SupervisorEngine
from lazybridge.testing import scripted_inputs, MockAgent

researcher = MockAgent("draft one", name="researcher")
store = Store()
store.write("policy", "peer-reviewed only")

# scripted_inputs returns a Callable[[str], str] that pops the next line
# each time the REPL prompts.  The REPL's last "continue" terminates it.
inputs = scripted_inputs([
    "store policy",                         # REPL command: print policy
    "retry researcher: be more specific",   # REPL command: rerun researcher
    "continue",                             # accept + return
])

supervisor = Agent(
    engine=SupervisorEngine(
        tools=[],
        agents=[researcher],
        store=store,
        input_fn=inputs,       # <- deterministic, no stdin
    ),
    name="supervisor",
)
env = supervisor("review the draft")
```

For async harnesses use `scripted_ainputs(...)` with
`SupervisorEngine(ainput_fn=...)` — same semantics, awaited.

## Pitfalls

- Dict-match is **substring**, not equality: `{"weather": "sunny"}` also
  fires on `"what's the weather forecast?"`.  Pick keys that are unique
  to each test branch.
- Exhausting a list response with `cycle=False` (default) raises
  `RuntimeError` mid-test.  Either size the list to the expected call
  count or set `cycle=True` for background noise agents.
- A `MockAgent` wrapped in `Agent(tools=[mock])` is registered on the
  outer agent's session via `_is_lazy_agent`, so `usage_summary()`
  rolls mock tokens into `by_agent[mock.name]` just like a real agent.

!!! note "API reference"

    MockAgent(
        responses: Callable | dict | list | Any,
        *,
        name: str = "mock_agent",
        description: str | None = None,
        output: type = str,
        cycle: bool = False,                  # list: cycle instead of StopIteration
        delay_ms: float = 0.0,                # simulate latency
        default_input_tokens: int = 10,
        default_output_tokens: int = 20,
        default_cost_usd: float = 0.0,
        default_latency_ms: float | None = None,
        default_model: str = "mock",
        default_provider: str = "mock",
    )

    mock.calls: list[MockCall]            # full history
    mock.call_count: int
    mock.last_call: MockCall | None
    mock.reset() -> None
    mock.assert_called_with(...) -> None
    mock.assert_call_count(n: int) -> None

    # MockCall record:
    #   env_in: Envelope, env_out: Envelope, elapsed_ms: float

    scripted_inputs(lines: Iterable[str]) -> Callable[[str], str]
    scripted_ainputs(lines: Iterable[str]) -> Callable[[str], Awaitable[str]]

!!! warning "Rules & invariants"

    - `MockAgent` carries `_is_lazy_agent=True`, so every place that
      accepts an Agent accepts a MockAgent: `Plan(Step(target=mock))`,
      `Agent(tools=[mock])`, `mock.as_tool()`, `Agent.chain`,
      `Agent.parallel`.
    - Response resolution is: callable → dict → list → scalar.  A callable
      can return any of the other shapes; the value is then fed back
      through the rules.
    - `default_*` metadata populates the Envelope so
      `session.usage_summary()` aggregations produce realistic-looking
      numbers in tests.
    - Exceptions passed in `responses` are **raised**, not returned.
      `ErrorInfo` is **returned** as `Envelope(error=...)`.  Pick the
      one that matches what your SUT should observe.

## See also

[agent](agent.md), [tool](tool.md),
[supervisor](supervisor.md), [plan](plan.md),
[evals](evals.md)
