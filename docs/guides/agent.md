# Agent

**Use `Agent` for** any single LLM interaction — one-shot calls, tool use, or structured output.
It's the only class you need at the Basic tier; all other features are opt-in via keyword args.

**Move to `Plan`** when steps need typed hand-offs, conditional routing, or crash-resume semantics.
**Use `Agent.chain`** for a simple linear sequence where text hand-offs between agents are enough.

## Example

```python
from lazybridge import Agent
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    bullets: list[str]

# Agent(...) constructs (no LLM call); ("task") invokes; .text() reads the Envelope.
print(Agent("claude-opus-4-7")("hello").text())

def search(query: str) -> str:
    """Search the web for ``query`` and return the top 3 hits."""
    return "..."

# tools= accepts functions — schema auto-inferred from type hints and docstring.
print(Agent("claude-opus-4-7", tools=[search])("AI news April 2026").text())

# output= activates structured output; .payload is the typed model instance.
resp = Agent("claude-opus-4-7", output=Summary)("summarise LazyBridge")
print(resp.payload.title, resp.payload.bullets)    # .text() would give the JSON dump

# Agents compose — editor treats researcher as a tool via tools=[researcher].
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

