# Agent.parallel

`Agent.parallel` is deterministic fan-out sugar. Every input agent
receives the same task and runs concurrently; you get back a list of
`Envelope`s in input order. No LLM sits on top of them deciding which
to call — you asked for N concurrent runs, you get N concurrent runs.

This is **not a third kind of parallelism** in the framework. Parallel
tool execution inside an engine is automatic and emergent (the LLM
decides, `asyncio.gather` implements). `Plan` with
`Step(parallel=True)` declares concurrent branches in a workflow.
`Agent.parallel` is just the case where you know exactly what you want
to run at once and want a one-liner for it.

Typical uses: comparing responses across providers; fanning out a
symmetric question (e.g. "search for this topic" across three
angle-specific researchers); running N idempotent probes.

## Example

```python
from lazybridge import Agent

us   = Agent("claude-opus-4-7", name="us", tools=[search_us])
eu   = Agent("claude-opus-4-7", name="eu", tools=[search_eu])
asia = Agent("claude-opus-4-7", name="asia", tools=[search_asia])

results = Agent.parallel(us, eu, asia,
                          concurrency_limit=3,
                          step_timeout=30.0)("AI policy news")

for env in results:
    print(env.metadata.model, env.text()[:100])
```

## Pitfalls

- The helper returns ``list[Envelope]``, not an ``Envelope``.  If you
  need a single aggregated answer, feed the list into a summariser
  agent as a follow-up.
- ``concurrency_limit=`` caps simultaneous in-flight calls; default
  (``None``) fires everything at once. Use a cap when you're rate-
  limit sensitive.
- ``step_timeout=`` wraps each per-agent call in ``asyncio.wait_for``.
  Timeouts return an error Envelope in the slot, preserving the
  positional contract.

!!! note "API reference"

    Agent.parallel(
        *agents: Agent,
        concurrency_limit: int | None = None,   # semaphore
        step_timeout: float | None = None,      # per-agent wait_for
        **kwargs,                                # name, description, session
    ) -> _ParallelAgent
    
    parallel_agent(task) -> list[Envelope]   # one entry per input agent, order preserved

!!! warning "Rules & invariants"

    - Deterministic fan-out only. Every input agent receives the same
      ``task``; every per-run Envelope appears in the returned list in
      input order.
    - No orchestrator LLM mediates the call. This is just
      ``asyncio.gather`` under a thin wrapper with optional semaphore
      (``concurrency_limit``) and per-agent ``wait_for`` (``step_timeout``).
    - Errors in a per-agent run surface as ``Envelope.error_envelope(...)``
      in the corresponding slot — the call never raises.
    - If you want the **LLM** to decide which agents to call (and whether
      in parallel), this is the wrong tool. Use ``Agent(tools=[a, b, c])``
      instead — the engine fans out tool calls automatically when the
      model emits more than one in a turn.

## See also

[agent](agent.md), [chain](chain.md),
[plan](plan.md), [parallel_steps](parallel-steps.md),
decision tree: [parallelism](../decisions/parallelism.md)
