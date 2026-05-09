# Agent

**Use `Agent` for** any single LLM interaction — one-shot calls, tool
use, or structured output. It's the only class you need at the Basic
tier; all other features are opt-in via keyword args.

**Move to `Plan`** when steps need typed hand-offs, conditional
routing, or crash-resume semantics. **Use `Agent.chain`** for a simple
linear sequence where text hand-offs between agents are enough.

The structured-config objects (`ResilienceConfig`, `ObservabilityConfig`,
`AgentRuntimeConfig`) are for codebases that want to reuse the same
knobs across many `Agent` instances — define once, pass as `runtime=`.
The flat kwargs work the same and override the config objects.

## Example

```python
from lazybridge import Agent, Session
from lazybridge.core.types import ResilienceConfig
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    bullets: list[str]

# 1) Two-line agent.
print(Agent("claude-opus-4-7")("hello").text())

# 2) Tools — auto-schema from type hints + docstring.
def search(query: str) -> str:
    """Search the web for ``query`` and return the top 3 hits."""
    return "..."

print(Agent("claude-opus-4-7", tools=[search])("AI news April 2026").text())

# 3) Structured output — read .payload, not .text().
resp = Agent("claude-opus-4-7", output=Summary)("summarise LazyBridge")
print(resp.payload.title, resp.payload.bullets)

# 4) Tool-is-Tool composition (Agents wrap Agents).
researcher = Agent("claude-opus-4-7", tools=[search], name="researcher")
editor     = Agent("claude-opus-4-7", tools=[researcher], name="editor")
print(editor("find papers and write a one-paragraph summary").text())

# 5) Production-shape: timeout + cache + provider fallback + tracing.
fb = Agent("gpt-5", tools=[search], name="fallback")
prod = Agent(
    "claude-opus-4-7",
    tools=[search],
    timeout=30.0,
    cache=True,
    fallback=fb,
    session=Session(db="events.sqlite", batched=True, console=True),
)
prod("draft a one-pager on the LazyBridge audit findings")
```

## Pitfalls

- Passing ``output=SomeModel`` and then calling ``.text()`` gives you
  the JSON dump of the payload, which is rarely what you want. Read
  ``.payload`` instead.
- ``Agent.parallel`` is sugar for deterministic fan-out returning
  ``list[Envelope]``. It is **not** "a different kind of parallelism" —
  if you want the LLM to decide, put the candidates in ``tools=[]``.
- ``verify=`` expects a judge that returns a verdict starting with
  ``"approved"`` (case-insensitive) to accept. Anything else is
  treated as rejection + feedback.
- ``timeout=None`` (default) leaves the run unbounded; tool calls
  inside a runaway agent can block forever. Pick a deadline that
  matches your SLO.
- ``fallback=`` runs the fallback's full pipeline (tools, memory,
  guard) on the same envelope, with the primary's error threaded
  into ``context``. Configure compatible ``output=`` /
  ``tools=`` on both agents.

!!! note "API reference"

    Agent(
        model_or_engine: str | Engine = "claude-opus-4-7",
        *,
        tools: list[Tool | Callable | Agent | ToolProvider] | None = None,
        output: type = str,
        memory: Memory | None = None,
        sources: list = (),
        guard: Guard | None = None,
        verify: Agent | None = None,
        max_verify: int = 3,
        name: str | None = None,
        description: str | None = None,
        session: Session | None = None,
        verbose: bool = False,
        # Convenience — pass provider name + model separately:
        model: str | None = None,
        engine: Engine | None = None,        # kwarg alias for the first positional
        native_tools: list[NativeTool | str] | None = None,
        # Structured config objects (compose with the flat kwargs below):
        runtime: AgentRuntimeConfig | None = None,
        resilience: ResilienceConfig | None = None,
        observability: ObservabilityConfig | None = None,
        # Resilience / safety knobs (each also reachable via ``resilience=``):
        output_validator: Callable[[Any], Any] | None = None,
        max_output_retries: int = 2,
        timeout: float | None = None,         # total deadline for run()
        max_retries: int = 3,                 # provider transient-error retries
        retry_delay: float = 1.0,
        fallback: Agent | None = None,
        cache: bool | CacheConfig = False,    # prompt caching
    ) -> Agent

    Sync:    agent(task) -> Envelope
    Async:   await agent.run(task) -> Envelope
    Stream:  async for chunk in agent.stream(task): ...

    Factories:
      Agent.from_model(model: str, **kw) -> Agent           # explicit LLM
      Agent.from_engine(engine: Engine, **kw) -> Agent      # explicit Plan / Supervisor / custom
      Agent.from_provider(name: str, *, tier: str = "medium", **kw) -> Agent

    Composition sugar (NOT new paradigms):
      Agent.chain(*agents, **kw)    -> Agent                 # sequential
      Agent.parallel(*agents, **kw) -> _ParallelAgent        # deterministic fan-out → list[Envelope]

!!! warning "Rules & invariants"

    - ``tools=`` accepts plain functions, ``Tool`` instances, other
      ``Agent`` instances, and tool providers (``MCPServer`` etc.). The
      framework normalises everything to ``Tool`` at construction; you never call a wrapper yourself.
    - When a nested Agent has no ``session=`` of its own, it inherits the
      caller's session and is registered on the graph with an ``as_tool``
      edge. Observability flows through the whole tree.
    - When the engine emits multiple tool invocations in a single step,
      they execute concurrently via ``asyncio.gather``. This is a
      capability, not a config knob; there is no serial mode.
    - ``output=str`` (default) makes ``Envelope.payload`` the model's
      text. Pass a Pydantic class to activate structured output —
      ``Envelope.payload`` becomes an instance of that class, with
      retry-with-feedback up to ``max_output_retries`` on validation
      failure.
    - ``verify=`` wraps the run in a judge/retry loop (max ``max_verify``
      attempts). The judge can be an Agent or a plain callable.
    - ``guard=`` filters both input and output. Blocked runs return an
      error Envelope without invoking the engine.
    - ``timeout=`` is the **total** deadline for ``agent.run()``; on
      expiry an ``AGENT_FINISH`` event with ``cancelled=True`` is emitted
      and a ``TimeoutError`` Envelope is returned.
    - ``cache=True`` enables prompt caching where the provider supports
      it (Anthropic explicit, OpenAI / DeepSeek auto). Pass
      ``CacheConfig(ttl="1h")`` for the longer Anthropic TTL.
    - ``runtime=`` / ``resilience=`` / ``observability=`` config objects
      group related kwargs. Precedence: explicit flat kwarg > config
      object > documented default.

## See also

- [Tool](tool.md) — how plain functions become tools.
- [Envelope](envelope.md) — the universal request/response object.
- [Session](session.md) — observability container.
