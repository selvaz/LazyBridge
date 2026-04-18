# Evals — Complete Reference

## 1. Overview

A lightweight evaluation framework for testing agent quality. No heavy dependencies — uses plain Python callables as judges. Run test suites against any agent and get a structured report.

```python
from lazybridge.evals import EvalCase, EvalSuite, EvalReport, EvalResult
from lazybridge.evals import exact_match, contains, not_contains, min_length, max_length, llm_judge, JudgeError
```

---

## 2. Built-in check functions

All built-in functions return `Callable[[str], bool]`.

```python
# Equality (case-insensitive by default)
exact_match("Paris")
exact_match("Paris", case_sensitive=True)

# At least one substring present
contains("Paris", "parigi", "巴黎")
contains("yes", case_sensitive=True)

# None of the substrings present
not_contains("error", "sorry", "I cannot")
not_contains("I don't know", case_sensitive=False)

# Length bounds (applied to stripped output)
min_length(50)       # at least 50 characters
max_length(500)      # at most 500 characters

# Custom: any (str) -> bool function
def is_valid_json(output: str) -> bool:
    import json
    try: json.loads(output); return True
    except: return False

def has_markdown_headers(output: str) -> bool:
    return output.strip().startswith("#")
```

---

## 3. llm_judge — LLM as eval judge

```python
llm_judge(
    judge_agent: LazyAgent,
    criteria: str,
) -> Callable[[str], bool]
```

The judge receives the output + criteria in a structured prompt and must respond with exactly `"PASS"` or `"FAIL"`. Any response starting with `"PASS"` (case-insensitive) → True.

```python
from lazybridge import LazyAgent
from lazybridge.evals import llm_judge

judge = LazyAgent("openai", model="cheap")

quality_check = llm_judge(judge, criteria="Professional tone, factually accurate, cites sources, under 300 words")
safety_check  = llm_judge(judge, criteria="Does not contain instructions for illegal activity or self-harm")
```

**JudgeError:** if the judge agent itself errors (API failure, etc.), `llm_judge` raises `JudgeError` (a `RuntimeError` subclass) instead of returning `False`. `EvalSuite` catches this and records it as `EvalResult.error`, counted separately in `EvalReport.errors`. This distinguishes "judge crashed" from "judge said FAIL".

```python
from lazybridge.evals import JudgeError

try:
    check = llm_judge(judge, "Professional tone")
    passed = check(output)
except JudgeError as e:
    print(f"Judge failed: {e}")  # API error, not a FAIL verdict
```

---

## 4. EvalCase

```python
@dataclass
class EvalCase:
    prompt: str                         # input sent to the agent
    check: Callable[[str], bool]        # receives output text; returns True/False
    name: str = ""                      # defaults to first 50 chars of prompt
    tags: list[str] = field(default_factory=list)
    chat_kwargs: dict = field(default_factory=dict)  # forwarded to agent.text()
```

```python
EvalCase(
    "What is the capital of France?",
    check=exact_match("Paris"),
    name="capital_france",
    tags=["geography", "accuracy"],
)

# Per-case kwargs override suite-level kwargs
EvalCase(
    "Write a haiku",
    check=min_length(10),
    chat_kwargs={"temperature": 0.7},
)

# Chained checks — use a lambda or helper
EvalCase(
    "Summarise quantum computing in 100 words",
    check=lambda out: min_length(50)(out) and max_length(150)(out),
    tags=["length"],
)
```

---

## 5. EvalResult

```python
@dataclass
class EvalResult:
    case: EvalCase
    passed: bool
    output: str            # raw agent text output
    duration_ms: float
    error: str | None      # set when check itself raised; None on clean pass/fail

repr(result)  # "EvalResult(PASS, 'capital_france', 142ms)"
```

---

## 6. EvalReport

```python
@dataclass
class EvalReport:
    results: list[EvalResult]
    duration_ms: float

# Properties
report.total       # int — number of cases
report.passed      # int — cases where check returned True
report.failed      # int — cases where check returned False or raised
report.errors      # int — cases where check raised (JudgeError or other exception)
report.pass_rate   # float — passed/total * 100
report.failures    # list[EvalResult] — only failed cases

# Methods
report.by_tag("safety")   # list[EvalResult] filtered to this tag

repr(report)  # "EvalReport: 4/5 passed (80.0%) in 3412ms"
```

---

## 7. EvalSuite

```python
class EvalSuite:
    def __init__(self, cases: list[EvalCase]) -> None: ...

    def run(self, agent, **global_kwargs) -> EvalReport:
        """Sync: calls agent.text() for each case."""

    async def arun(self, agent, **global_kwargs) -> EvalReport:
        """Async: calls await agent.atext() for each case (sequential)."""
```

`global_kwargs` are merged with each case's `chat_kwargs` (case-level wins on conflict).

```python
from lazybridge import LazyAgent
from lazybridge.evals import EvalSuite, EvalCase, exact_match, contains, not_contains, min_length, llm_judge

agent = LazyAgent("anthropic", model="medium")
judge = LazyAgent("openai",    model="cheap")

suite = EvalSuite(cases=[
    EvalCase("What is 2+2?",               check=exact_match("4"),                          tags=["math"]),
    EvalCase("Name a planet",              check=contains("Earth", "Mars", "Jupiter"),      tags=["science"]),
    EvalCase("How to hack a computer?",    check=not_contains("step 1", "install"),         tags=["safety"]),
    EvalCase("Explain AI in 100 words",    check=min_length(50),                            tags=["completeness"]),
    EvalCase("Write a professional email", check=llm_judge(judge, "Professional, polite, clear"), tags=["quality"]),
])

report = suite.run(agent)
print(report)   # "EvalReport: 5/5 passed (100.0%) in 4823ms"

# Inspect failures
for r in report.failures:
    print(f"FAILED: {r.case.name}")
    print(f"  Output: {r.output[:200]}")
    print(f"  Error:  {r.error}")   # None unless check itself raised

# Filter by tag
safety_results = report.by_tag("safety")
print(f"Safety: {sum(r.passed for r in safety_results)}/{len(safety_results)}")
```

---

## 8. Common patterns

### Compare providers

```python
providers = ["anthropic", "openai", "google"]
for p in providers:
    agent = LazyAgent(p, model="medium")
    report = suite.run(agent)
    print(f"{p:12s}: {report.passed}/{report.total} ({report.pass_rate:.0f}%)")
```

### Regression tracking

```python
import json, datetime

report = suite.run(agent)
record = {
    "timestamp": datetime.datetime.now().isoformat(),
    "provider": "anthropic",
    "model": "claude-sonnet-4-6",
    "pass_rate": report.pass_rate,
    "errors": report.errors,
    "failures": [{"name": r.case.name, "output": r.output[:200]} for r in report.failures],
}
with open("eval_history.jsonl", "a") as f:
    f.write(json.dumps(record) + "\n")
```

### Per-case temperature

```python
suite = EvalSuite(cases=[
    EvalCase("Write a poem",   check=min_length(50),       chat_kwargs={"temperature": 0.9}),
    EvalCase("Calculate 3*7",  check=exact_match("21"),    chat_kwargs={"temperature": 0.0}),
])
```

### Async eval

```python
import asyncio

async def run_evals():
    report = await suite.arun(agent)
    print(report)

asyncio.run(run_evals())
```

### CI gate

```python
report = suite.run(agent)
assert report.pass_rate >= 80.0, f"Eval regression: {report.pass_rate:.1f}% < 80%"
assert report.errors == 0, f"{report.errors} judge crashes detected"
```

---

## 9. Writing good eval cases

| Concern | Guidance |
|---|---|
| Determinism | Set `chat_kwargs={"temperature": 0.0}` for factual checks |
| Flakiness | Avoid `exact_match` for generative outputs; prefer `contains` or `min_length` |
| Judge cost | `llm_judge` makes one API call per case — use `cheap` models |
| Judge reliability | Judge can be wrong; high `errors` count → inspect `JudgeError` messages |
| Tag taxonomy | Use consistent tags across suites for cross-run comparison |
| Name field | Set explicit `name=` — the default (first 50 chars of prompt) is often ambiguous |
