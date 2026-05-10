# Parallel

Deterministic, scripted fan-out. The same task goes to N agents
concurrently. The wrapper returns ONE `Envelope` whose `text()` is the
labelled-text join of every branch's output — same shape as `Plan`'s
`from_parallel_all` aggregator. For typed per-branch access call
`parallel.run_branches(task)` (async). No orchestrator LLM is involved.

## Signature

```python
from lazybridge import Agent

# Canonical — Agent.parallel IS the canonical form for scripted fan-out.
multi = Agent.parallel(
    *agents,                       # one or more Agent instances
    concurrency_limit=None,        # int | None; cap on simultaneous in-flight calls
    step_timeout=None,             # float | None; per-agent asyncio.wait_for deadline (seconds)
    name="parallel",
    description=None,
    session=None,
)

env = multi(task)                  # ONE Envelope: text() = labelled join, payload = joined str
print(env.text())                  # "[a]\n<a-output>\n\n[b]\n<b-output>" ...

# Typed per-branch access (advanced):
import asyncio
branches = asyncio.run(multi.run_branches(task))   # list[Envelope] in input order
```

`Agent.parallel(...)` returns a `ParallelAgent` — a sibling class of
`Agent`. Its `__call__` returns ONE folded `Envelope` (so the runner
plugs into another agent's `tools=[...]` uniformly with every other
agent in the framework). For the underlying `list[Envelope]`, use
`run_branches(task)`. The closest from-primitives equivalent is
hand-written `asyncio.gather(*[a.run(task) for a in agents])` — see
[Canonical vs sugar](../../concepts/canonical-vs-sugar.md) for the
full nuance.

## Synopsis

`Agent.parallel(a, b, c)(task)` runs the same task against three
agents concurrently and returns three envelopes in the order you
passed them. Errors in one branch surface as `Envelope.error_envelope(...)`
in the corresponding slot — the call never raises; the positional
contract is preserved.

`concurrency_limit=` caps simultaneous in-flight calls (a
`asyncio.Semaphore`). `step_timeout=` wraps each per-agent call in
`asyncio.wait_for` so a slow branch can't stall the rest.

`Agent.parallel` is **deterministic and scripted** — every input
agent runs unconditionally. If you want the LLM to decide which
sub-agents to invoke (and whether in parallel), put the agents in
`tools=[...]` of a regular `Agent` instead; the engine fans out tool
calls automatically when the model emits more than one in a turn.

## When to use it

- **Multi-region / multi-source fan-out.** Three researchers, one
  per region, called against the same query.
- **Ensemble voting.** Same task, several models, then a downstream
  step picks the best or aggregates.
- **Independent sub-tasks that share an input.** Where you've
  already decided every branch must run.
- **Throttled fan-out** when downstream APIs are rate-limited —
  use `concurrency_limit=` to cap parallelism without serialising.

## When NOT to use it

- **LLM-directed dispatch.** Use `Agent(tools=[a, b, c])`. The
  engine emits parallel tool calls automatically when the model
  asks for more than one — so "the model decides which subset to
  call" comes for free. `Agent.parallel` runs every branch; you
  cannot opt one out at runtime.
- **Conditional / routed flows.** Use `Plan` with parallel bands
  (`Step(..., parallel=True)`) plus routing, or use
  `from_parallel_all("name")` to aggregate.  `Agent.parallel` runs
  every branch unconditionally; if you need concurrent steps that
  must converge in a typed downstream `Step`, use `Plan`.
- **Anything that needs typed downstream consumption.** Each
  envelope's payload is whatever the corresponding agent produced
  — possibly several different shapes. If the next step needs a
  uniform typed input, normalise via a follow-up summariser or
  use `Plan` parallel bands with `from_parallel_all`.

## Example

```python
from lazybridge import Agent, LLMEngine


def search_us(query: str) -> str:
    """Search US sources for ``query``."""
    return "..."

def search_eu(query: str) -> str:
    """Search EU sources for ``query``."""
    return "..."

def search_asia(query: str) -> str:
    """Search Asian sources for ``query``."""
    return "..."


us = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    tools=[search_us],
    name="us",
)
eu = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    tools=[search_eu],
    name="eu",
)
asia = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    tools=[search_asia],
    name="asia",
)


# 1) Three branches, one task, results in input order.
multi = Agent.parallel(
    us,
    eu,
    asia,
    concurrency_limit=3,           # cap simultaneous in-flight calls
    step_timeout=30.0,
)
results = multi("AI policy news")
for env in results:
    print(env.metadata.model, env.text()[:100])


# 2) Aggregate into a single answer with a follow-up agent.
def join_branches(envs: list) -> str:
    return "\n\n".join(f"[{e.metadata.model}] {e.text()}" for e in envs)


synthesiser = Agent(
    engine=LLMEngine(
        "claude-opus-4-7",
        system="Combine the regional briefings into one global summary.",
    ),
    name="synth",
)
summary = synthesiser(join_branches(results))
print(summary.text())


# 3) Failure isolation — one branch's error doesn't kill the others.
results = multi("a deliberately tricky question")
for env in results:
    if not env.ok:
        print(f"branch failed ({env.error.type}): {env.error.message}")
    else:
        print(env.text()[:100])
```

## Pitfalls

- **`__call__` returns ONE Envelope, not `list[Envelope]`** (since 0.8.0).
  Read `env.text()` for the labelled-text join, or call
  `multi.run_branches(task)` (async) for the typed list.
- **`concurrency_limit=None` (default) fires everything at once.**
  When the underlying providers are rate-limited or your CPU /
  network is the bottleneck, set a cap.
- **`step_timeout` returns an error envelope, not a raise.** The
  positional contract is preserved — the slot for the timed-out
  agent contains an error envelope; siblings keep their results.
  Check `env.ok` before reading `env.text()`.  The first non-`None`
  branch error also propagates as the wrapper Envelope's `error`
  so downstream consumers can short-circuit.
- **Not LLM-directed.** If you want "the model decides whether to
  call all three", use `Agent(tools=[us, eu, asia])` instead;
  parallel tool dispatch happens automatically when the engine
  emits multiple tool calls in a single turn.
- **Automatic aggregation since 0.8.0.** The wrapper Envelope's
  `payload` is a labelled-text join of every branch — same shape
  as `Plan`'s `from_parallel_all` aggregator — and `metadata.nested_*`
  rolls every branch's cost up.  Use `as_tool()` to plug the runner
  into another agent's `tools=[...]` list with no extra adapter.
- **Cost rollup.** Each branch's metadata is preserved in its
  envelope; the wrapper folds them into `metadata.nested_*` so
  parallel-of-parallel composes cleanly.  Sessions aggregate cost
  across all branches via `usage_summary()`.

## See also

- [Chain](chain.md) — sequential composition; complements
  parallel.
- [As tool](as-tool.md) — `multi.as_tool()` exposes the fan-out as
  a single `Tool` that delegates to `run()` (since 0.8.0 the
  wrapper Envelope already carries the labelled-text join).
- *Guides → Full → Plan* (Phase 3) — `Plan` parallel bands
  (`Step(..., parallel=True)`) and `from_parallel_all("name")`
  aggregation when concurrent steps must produce a single
  downstream input.
- [Canonical vs sugar](../../concepts/canonical-vs-sugar.md) — why
  `Agent.parallel` is its own primitive, not sugar over `Agent`.
