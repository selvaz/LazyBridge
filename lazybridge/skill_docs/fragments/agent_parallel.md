## signature
Agent.parallel(
    *agents: Agent,
    concurrency_limit: int | None = None,   # semaphore
    step_timeout: float | None = None,      # per-agent wait_for
    **kwargs,                                # name, description, session
) -> _ParallelAgent

parallel_agent(task) -> list[Envelope]   # one entry per input agent, order preserved

## rules
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

## example
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

## pitfalls
- The helper returns ``list[Envelope]``, not an ``Envelope``.  If you
  need a single aggregated answer, feed the list into a summariser
  agent as a follow-up.
- ``concurrency_limit=`` caps simultaneous in-flight calls; default
  (``None``) fires everything at once. Use a cap when you're rate-
  limit sensitive.
- ``step_timeout=`` wraps each per-agent call in ``asyncio.wait_for``.
  Timeouts return an error Envelope in the slot, preserving the
  positional contract.

