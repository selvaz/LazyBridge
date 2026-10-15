# Testing LazyBridge code

Read when writing tests for agents, pipelines, human-in-the-loop flows or
coding-engine integrations, or when building evaluation suites. Source of
truth: `lazybridge/testing.py`, `lazybridge/ext/evals/`; examples in
`tests/unit/test_doc_examples_runtime.py` and
`docs/guides/project-patterns.md`.

## MockAgent: deterministic agent double

```python
from lazybridge.testing import MockAgent

research = MockAgent(responses=["research-key"], name="research")
write = MockAgent(responses=["done"], name="write")

result = build_pipeline(research=research, write=write)("run-42")

research.assert_call_count(1)
write.assert_called_with(contains="research-key")
assert result.ok
```

`MockAgent(responses, *, name="mock_agent", description=None, output=str,
cycle=False, delay_ms=0.0, default_input_tokens=10, default_output_tokens=20,
default_cost_usd=0.0, default_model="mock", default_provider="mock")`.
It works anywhere an `Agent` does: `tools=[mock]`, `Plan(Step(mock))`,
`mock.as_tool()`, `Agent.chain` / `Agent.parallel`.

`responses` is resolved per call:

1. callable `fn(env)` (sync or async), whose return is resolved again;
2. dict: substring of `env.task` → response, `"*"` as catch-all;
3. list: one per call (exhaustion raises unless `cycle=True`);
4. scalar: same every call.

A response may be an `Envelope` (returned verbatim; test error paths), an
`ErrorInfo`, an exception instance (raised from `run()`), a Pydantic model,
or any value used as payload. Inspect `mock.calls`, `mock.call_count`,
`assert_call_count(n)`, `assert_called_with(task=None, contains=None)`.

## Make pipelines injectable

Build every pipeline behind a factory whose parameters are its agents (and
its `Store`), so tests pass `MockAgent`s and a `Store()` in memory. Test the
composition, not the model: data flowing between steps, routing branches
(cover each predicate outcome), resume (run twice with the same
run-specific `checkpoint_key`), and `result.ok` on the failure path
(a mock returning an `ErrorInfo`).

## Human-in-the-loop tests

`lazybridge.testing.scripted_inputs(lines)` returns an `input_fn` that
replays answers in order (`scripted_ainputs` for async). Pass it to
`SupervisorEngine(input_fn=...)`; for `HumanEngine` use a custom `ui=`
object. Never let a test block on `input()`.

## Coding engines without credentials

`ClaudeCodeEngine(client=...)` and `CodexEngine(client=...)` accept an
injected transport. Unit tests in the repository use fake clients
(`tests/unit/engines/codex/`, `tests/unit/engines/claude_code/`); copy that
pattern instead of calling a live runtime. Live smoke scripts live in
`examples/codex/` and `examples/claude_code/` and need an authenticated CLI.
`parse_usage_report(text)` (`lazybridge.engines.claude_code.usage`) parses
quota prose offline.

## Evals (quality, not unit tests)

```python
from lazybridge.ext.evals import EvalCase, EvalSuite, contains, llm_judge

suite = EvalSuite(
    EvalCase("Capital of France?", check=contains("Paris")),
    EvalCase("Summarise the memo", check=llm_judge(judge_agent, "Is it faithful?")),
)
report = suite.run(agent)          # await suite.arun(agent) in async code
```

Helpers: `exact_match`, `contains`, `not_contains`, `min_length`,
`max_length`, `llm_judge`. Results come back as `EvalReport` /
`EvalResult`. The module is Experimental.

## Before finishing

Grep the diff for `for step in`, `state = {}`, `json.dumps(` inside a prompt,
`try:` around an agent call, and new `sqlite3.connect(` / `duckdb.connect(`
(see `plan-and-pipelines.md`, rule 12). Run the repository suite with the
project's interpreter; for this repo the drift gate
`python -m lazybridge.skill_docs._build --check` also runs in CI.
