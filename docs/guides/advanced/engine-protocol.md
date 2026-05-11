# Engine protocol

The single abstraction every engine satisfies. `LLMEngine`, `Plan`,
`HumanEngine`, and `SupervisorEngine` all implement it. Implement it
yourself when you want LazyBridge to drive a decision-making layer
that none of the built-ins covers — a deterministic rule engine, a
network of human approvers, a scripted dispatcher for tests.

## Signature

```python
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from lazybridge import Envelope


@runtime_checkable
class Engine(Protocol):
    """Contract every engine must satisfy."""

    async def run(
        self,
        env: Envelope,
        *,
        tools: list,                # list[Tool] — already normalised
        output_type: type,
        memory: Any | None,         # Memory | None
        session: Any | None,        # Session | None
        store: Any | None = None,   # Store | None — Plan checkpoint surface; other engines ignore
        plan_state: Any | None = None,  # ditto
    ) -> Envelope: ...

    async def stream(
        self,
        env: Envelope,
        *,
        tools: list,
        output_type: type,
        memory: Any | None,
        session: Any | None,
    ) -> AsyncIterator[str]: ...
```

The protocol is `@runtime_checkable`, so
`isinstance(my_engine, Engine)` is a true assertion you can put in
test code. `store` and `plan_state` are accepted by every engine
but only `Plan` uses them; `LLMEngine` / `HumanEngine` /
`SupervisorEngine` declare them as `accepted-and-ignored` so the
calling shape is uniform.

## Synopsis

Every `Agent` delegates to its `engine`. The agent normalises the
user-supplied `tools=[...]` (functions / Agents / `Tool` instances /
`ToolProvider`s all collapse to `Tool`), then calls
`engine.run(envelope, tools=..., output_type=..., memory=..., session=...)`.
The engine is responsible for:

- Producing an `Envelope` from the input. Errors must be wrapped in
  `Envelope.error_envelope(exc)` rather than raised — propagating
  exceptions breaks resilience layers (`fallback=`, `verify=`).
- Optionally calling tools. The agent has already wrapped them; you
  invoke them with `tool.run(...)` (async) or `tool.run_sync(...)`
  (sync, drives async coroutines to completion).
- Optionally emitting `Session` events for observability. At minimum
  emit `AGENT_START` and `AGENT_FINISH` — without those, your engine
  is invisible in tracing and graph rendering.
- Implementing `stream`. If you don't have incremental output, yield
  the final text as a single chunk so `agent.stream(...)` callers
  don't break.

The agent stamps `engine._agent_name = self.name` before the first
call so the engine can tag emitted events with the wrapping agent's
name. Read it via `getattr(self, "_agent_name", "<engine_class>")`.

## When to use it

- **The built-in engines don't fit the shape** of decision-making
  you want — a deterministic rule engine driven by external state, a
  multi-human voting layer, a recorded-script dispatcher for replay
  testing.
- **You need an engine that reuses the rest of the framework's
  state primitives** — `Memory`, `Session`, `Store`, `tools=[...]`
  normalisation — without forking.
- **Test doubles.** A `MockEngine` that returns canned envelopes
  lets you exercise every other moving part of an agent (memory,
  guards, output validation, fallback) without provider calls.

## When NOT to use it

- **You just want to wrap an LLM call.** Subclass `BaseProvider`
  instead — that's the layer below an engine. `LLMEngine` is the
  framework's adapter from "any provider" to the agent.
- **You want a one-off non-LLM step inside a pipeline.** Drop a
  plain callable into `Step(target=callable, name=...)` — `Plan`
  dispatches callables directly, no custom engine required.
- **You want to inject behaviour into an existing engine.** Use
  `guard=` for input/output filtering, `verify=` for output
  judging, `fallback=` for failover; don't subclass `LLMEngine`.

## Example

```python
from collections.abc import AsyncIterator
from typing import Any

from lazybridge import Agent, Envelope
from lazybridge.engines.base import Engine
from lazybridge.session import EventType


class EchoEngine:
    """Trivial engine that returns the task prefixed with a tag."""

    async def run(
        self,
        env,
        *,
        tools,
        output_type,
        memory,
        session,
        store: Any | None = None,
        plan_state: Any | None = None,
    ):
        agent_name = getattr(self, "_agent_name", "echo")
        if session:
            session.emit(
                EventType.AGENT_START,
                {"agent_name": agent_name, "task": env.task},
            )
        result = Envelope(task=env.task, payload=f"echo:{env.task}")
        if session:
            session.emit(
                EventType.AGENT_FINISH,
                {"agent_name": agent_name, "payload": result.text()},
            )
        return result

    async def stream(
        self,
        env,
        *,
        tools,
        output_type,
        memory,
        session,
    ) -> AsyncIterator[str]:
        out = await self.run(
            env,
            tools=tools,
            output_type=output_type,
            memory=memory,
            session=session,
        )
        yield out.text()


# Runtime-check: EchoEngine satisfies the Engine Protocol.
assert isinstance(EchoEngine(), Engine)


# Plug into Agent — same surface as any built-in engine.
# Non-LLM engines require an explicit ``name=`` (T7 since 0.7.9).
agent = Agent(engine=EchoEngine(), name="echo")
result = agent("hello")
print(result.text())   # "echo:hello"
```

For a full reference implementation, read `lazybridge.ext.hil.supervisor`
(~280 LOC). It covers event emission, memory integration, async-to-sync
bridging via `asyncio.to_thread`, and the optional `ainput_fn` async
prompt path.

## Pitfalls

- **Skipping `stream` entirely breaks `agent.stream(...)`.**
  Implement it to at least yield the final text once (the pattern
  in the example above). Most callers don't need true streaming;
  they just need the method to exist.
- **Not emitting session events makes your engine invisible.** No
  cost rollup, no graph node, no audit trail. At minimum emit
  `AGENT_START` and `AGENT_FINISH`. Add `TOOL_CALL` / `TOOL_RESULT`
  / `TOOL_ERROR` when you dispatch tools.
- **Raising instead of wrapping breaks resilience layers.**
  `Agent(fallback=...)`, `Agent(verify=...)`, and
  `Plan(checkpoint_key=...)` all expect the engine to return an
  error envelope, not raise. Wrap your `try/except` body and
  return `Envelope.error_envelope(exc)`.
- **The engine receives a normalised `list[Tool]`.** Do not assume
  the agent's internal `_tool_map` shape is available, do not
  re-wrap functions, do not call `_wrap_tool` yourself. Treat
  `tools` as a flat list of `Tool` instances ready to invoke.
- **`store` and `plan_state` kwargs.** Even if your engine doesn't
  use them, declare them in `run`'s signature with default `None`
  — `Plan.run` and `Agent.run` may pass them positionally-by-keyword
  through the engine boundary. The protocol declares them, so
  satisfying the protocol means accepting them.
- **`engine._agent_name` is set by the wrapping agent**, not by
  you. Don't override it from inside the engine; read it
  defensively (`getattr(self, "_agent_name", "<class>")`) so the
  engine still works when invoked outside an `Agent`.

## See also

- [BaseProvider](base-provider.md) — the layer below an engine; for
  custom LLM backends rather than custom decision-making
  mechanisms.
- [Plan](../full/plan.md) — example of a non-LLM engine you can
  read for reference.
- [SupervisorEngine](../full/supervisor.md) — the most complete
  reference engine implementation.
- [Mental model](../../concepts/mental-model.md) — where the
  engine sits in the Agent = Engine + Tools + State decomposition.
