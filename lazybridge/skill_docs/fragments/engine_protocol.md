## signature
# Protocol contract every engine implements.

@runtime_checkable
class Engine(Protocol):
    async def run(
        self,
        env: Envelope,
        *,
        tools: list[Tool],
        output_type: type,
        memory: Memory | None,
        session: Session | None,
    ) -> Envelope: ...

    async def stream(
        self,
        env: Envelope,
        *,
        tools: list[Tool],
        output_type: type,
        memory: Memory | None,
        session: Session | None,
    ) -> AsyncIterator[str]: ...

# Built-ins implementing this:
#   LLMEngine, HumanEngine, SupervisorEngine, Plan

## rules
- ``run`` is the primary entry point: receives an Envelope, returns an
  Envelope. It must not raise; wrap exceptions in
  ``Envelope.error_envelope(exc)``.
- ``stream`` is optional for non-streaming engines; yield the final
  text as a single chunk if no incremental output is available.
- Engines receive an already-wrapped ``list[Tool]`` (Agent calls
  ``build_tool_map`` / ``wrap_tool`` before invoking the engine). You do
  NOT need to handle raw functions / Agents in the engine body.
- Agents set ``engine._agent_name`` before invocation. Use it when
  emitting events for observability.

## narrative
`Engine` is the abstraction that makes LazyBridge's uniform API
possible. An `Agent` is an engine-agnostic envelope; every engine
(LLM, Human, Supervisor, Plan) implements the same two-method
contract, so the Agent doesn't change its call surface when you swap
the engine.

To ship a new engine you implement `run` (and optionally `stream`).
The engine gets the request as an `Envelope`, the pre-wrapped tool
list, the output type, memory, and the session. It produces an
`Envelope`.

Typical custom engines people write:

* `ReactEngine` — tool-calling loop with explicit thought/action/
  observation steps for research agents.
* `RouterEngine` — pattern-matches the task string and dispatches to
  sub-agents without calling an LLM at all.
* `CachingEngine` — wraps another engine with result caching keyed on
  the input envelope hash.

`SupervisorEngine` is a good reference implementation to read — it's
~280 LOC and covers the common patterns (event emission, memory
integration, async-to-sync bridging).

## example
```python
from lazybridge import Agent, Envelope
from lazybridge.session import EventType
from lazybridge.engines.base import Engine
from typing import AsyncIterator

class EchoEngine:
    """Trivial engine that returns the task prefixed with a tag."""

    async def run(self, env, *, tools, output_type, memory, session):
        if session:
            session.emit(EventType.AGENT_START,
                         {"agent_name": getattr(self, "_agent_name", "echo"),
                          "task": env.task})
        return Envelope(task=env.task, payload=f"echo:{env.task}")

    async def stream(self, env, *, tools, output_type, memory, session) -> AsyncIterator[str]:
        out = await self.run(env, tools=tools, output_type=output_type,
                             memory=memory, session=session)
        yield out.text()

# Runtime-check: EchoEngine satisfies the Engine Protocol.
assert isinstance(EchoEngine(), Engine)

# Plug into Agent — same surface as any built-in engine.
print(Agent(engine=EchoEngine())("hello").text())
```

## pitfalls
- Skipping ``stream`` entirely breaks ``agent.stream(...)``. Implement
  it to at least yield the final text once, per the pattern above.
- Not emitting session events makes your engine invisible in tracing.
  At minimum emit AGENT_START + AGENT_FINISH.
- The engine receives ``tools``, not ``agent._tool_map``. Treat it as
  a flat list; don't assume the Agent's internal structure.

## see-also
[agent](agent.md), [base_provider](base-provider.md),
[supervisor](supervisor.md)
