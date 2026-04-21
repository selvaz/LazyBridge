# Agent

`Agent` is the only public abstraction in LazyBridge. Every pipeline —
one-shot LLM calls, REPL supervisors, multi-step Plans — is an `Agent`
with a different engine plugged into it. The call surface does not change.

The key contract to internalise is **tool is tool**. When you pass things
to `tools=[...]`, it doesn't matter whether each entry is a plain function,
another Agent, or an Agent that itself has Agents as tools. The composition
is closed under `Tool`, so you can build deeply-nested agent trees and the
outer engine treats them identically to a one-line function.

The second contract is **parallelism is free, not configured**. If the
engine decides to call five tools in one step, LazyBridge fans them out
with `asyncio.gather`. You do not flip a switch. When you want the
orchestration to be deterministic — the fan-out shape is fixed by you,
not by the model — you use `Plan` or the `Agent.parallel` sugar.

## Example

```python
from lazybridge import Agent
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    bullets: list[str]

# Tier 1: two lines.
print(Agent("claude-opus-4-7")("hello").text())

# Tier 2: with tools (plain functions; schema auto-generated from hints).
def search(query: str) -> str:
    """Search the web for ``query`` and return the top 3 hits."""
    return "..."

print(Agent("claude-opus-4-7", tools=[search])("AI news April 2026").text())

# Tier 3: structured output.
resp = Agent("claude-opus-4-7", output=Summary)("summarise LazyBridge")
print(resp.payload.title, resp.payload.bullets)

# Nested agent-of-agent — uniform surface, no special ceremony.
researcher = Agent("claude-opus-4-7", tools=[search], name="researcher")
editor     = Agent("claude-opus-4-7", tools=[researcher], name="editor")
print(editor("find papers and write a one-paragraph summary").text())
```

## Pitfalls

- Passing ``output=SomeModel`` without tools and then calling ``.text()``
  gives you the JSON dump of the payload, which is rarely what you want.
  Read ``.payload`` instead.
- ``Agent.parallel`` is sugar for deterministic fan-out returning
  ``list[Envelope]``. It is **not** "a different kind of parallelism" —
  if you want the LLM to decide, put the candidates in ``tools=[]``.
- ``verify=`` expects a judge that returns a verdict string starting
  with ``"approved"`` (case-insensitive) to accept. Anything else is
  treated as rejection + feedback.

!!! note "API reference"

    Agent(
        model_or_engine: str | Engine = "claude-opus-4-7",
        *,
        tools: list[Tool | Callable | Agent] = (),
        output: type = str,
        memory: Memory | None = None,
        sources: list = (),
        guard: Guard | None = None,
        verify: Agent | None = None,
        max_verify: int = 3,
        session: Session | None = None,
        verbose: bool = False,
        name: str | None = None,
        description: str | None = None,
        model: str | None = None,     # tier alias when first arg is a provider name
        engine: Engine | None = None, # kwarg alias for the first positional
    ) -> Agent
    
    Sync:   agent(task) -> Envelope
    Async:  await agent.run(task) -> Envelope
    Stream: async for chunk in agent.stream(task): ...
    
    Factories:
      Agent.from_model(model: str, **kw) -> Agent       # explicit LLM
      Agent.from_engine(engine: Engine, **kw) -> Agent  # explicit Plan / Supervisor / custom
      Agent.from_provider(name: str, *, tier: str = "medium", **kw) -> Agent
    
    Composition sugar (NOT new paradigms):
      Agent.chain(*agents, **kw)  -> Agent          # sequential
      Agent.parallel(*agents, **kw) -> _ParallelAgent  # deterministic fan-out → list[Envelope]

!!! warning "Rules & invariants"

    - ``tools=`` accepts functions, Tool instances, Agent instances, and
      Agents-of-Agents. ``wrap_tool`` normalises everything at construction.
    - When a nested Agent has no ``session=``, it inherits the caller's session
      and is registered on the graph with an ``as_tool`` edge. Observability
      flows through the whole tree.
    - When the engine emits multiple tool invocations in a single step, they
      execute concurrently via ``asyncio.gather``. This is a capability, not
      a config knob; there is no serial mode.
    - ``output=`` defaulting to ``str`` means ``Envelope.payload`` is the
      model's text. Passing a Pydantic class sets up structured output and
      ``Envelope.payload`` becomes an instance of that class.
    - ``verify=`` wraps the run in a judge/retry loop (max ``max_verify``
      attempts). The judge can be an Agent or a plain callable.
    - ``guard=`` filters both input and output. Blocked runs return an
      error Envelope without invoking the engine.

## See also

[tool](tool.md), [envelope](envelope.md),
[chain](chain.md), [agent_parallel](agent-parallel.md),
[as_tool](as-tool.md), [session](session.md),
decision tree: [pick_tier](../decisions/pick-tier.md)
