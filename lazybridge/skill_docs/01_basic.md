# LazyBridge — Basic tier
One-shot or tool-calling agents. Text or structured output.
No memory, no pipeline, no HIL. If you need state across calls
or more than one agent, go to Mid.

## Agent

**signature**

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

**rules**

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

**example**

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

**pitfalls**

- Passing ``output=SomeModel`` without tools and then calling ``.text()``
  gives you the JSON dump of the payload, which is rarely what you want.
  Read ``.payload`` instead.
- ``Agent.parallel`` is sugar for deterministic fan-out returning
  ``list[Envelope]``. It is **not** "a different kind of parallelism" —
  if you want the LLM to decide, put the candidates in ``tools=[]``.
- ``verify=`` expects a judge that returns a verdict string starting
  with ``"approved"`` (case-insensitive) to accept. Anything else is
  treated as rejection + feedback.

**see-also**

[tool](tool.md), [envelope](envelope.md),
[chain](chain.md), [agent_parallel](agent-parallel.md),
[as_tool](as-tool.md), [session](session.md),
decision tree: [pick_tier](../decisions/pick-tier.md)

## Tool

**signature**

Tool(
    func: Callable,
    *,
    name: str | None = None,
    description: str | None = None,
    guidance: str | None = None,
    mode: Literal["signature", "llm", "hybrid"] = "signature",
    schema_llm: Any | None = None,
    strict: bool = False,
) -> Tool

Tool.definition() -> ToolDefinition
await Tool.run(**kwargs) -> Any
Tool.run_sync(**kwargs) -> Any   # handles async ``func`` transparently

wrap_tool(obj) -> Tool   # converts functions / Agents / Tools uniformly
build_tool_map(tools: list) -> dict[str, Tool]

**rules**

- Schema generation is automatic from ``func``'s type hints and docstring
  in ``mode="signature"`` (default). Use ``mode="llm"`` to let an LLM
  synthesise the schema, ``mode="hybrid"`` for both.
- ``name`` defaults to ``func.__name__``. Names are API-facing; pick
  stable ones.
- ``strict=True`` enables provider-strict JSON-schema validation on tool
  arguments (Anthropic / OpenAI strict mode).
- ``run`` is async; ``run_sync`` auto-detects coroutine functions and
  drives them to completion so REPL callers (e.g. SupervisorEngine) never
  see a raw coroutine.

**example**

```python
from lazybridge import Tool, Agent

def calculate(expression: str) -> float:
    """Evaluate a basic arithmetic expression and return the result.

    Supports +, -, *, /, parentheses.
    """
    return eval(expression)  # noqa: S307  (trusted inputs only)

# Implicit: pass the function, LazyBridge wraps it.
Agent("claude-opus-4-7", tools=[calculate])("what is 17 * 23?")

# Explicit: override the name or strictness.
calc_tool = Tool(calculate, name="calc", strict=True,
                 description="Evaluate an arithmetic expression.")
Agent("claude-opus-4-7", tools=[calc_tool])("...")

# An Agent is also a Tool — no ceremony.
researcher = Agent("claude-opus-4-7", tools=[search], name="researcher")
orchestrator = Agent("claude-opus-4-7", tools=[researcher])
```

**pitfalls**

- A function with no type hints produces an empty JSON schema and the
  LLM will not know how to call it. Always annotate parameters.
- A docstring is part of the contract the LLM reads. "Returns the
  weather" is weaker than "Returns the current temperature in Celsius
  and a one-word condition (sunny / cloudy / rainy) for ``city``."
- ``strict=True`` rejects optional / defaulted args under some providers;
  if a call fails with "unknown parameter", try ``strict=False``.

**see-also**

[agent](agent.md), [as_tool](as-tool.md),
decision tree: [parallelism](../decisions/parallelism.md)

## Envelope

**signature**

class Envelope(BaseModel, Generic[T]):
    task: str | None = None
    context: str | None = None
    payload: T | None = None
    metadata: EnvelopeMetadata = ...
    error: ErrorInfo | None = None

    @property
    def ok: bool
    def text() -> str
    @classmethod
    def from_task(task: str, context: str | None = None) -> Envelope
    @classmethod
    def error_envelope(exc: Exception, retryable: bool = False) -> Envelope

class EnvelopeMetadata(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    model: str | None = None
    provider: str | None = None
    run_id: str | None = None
    # Aggregation buckets filled when this agent called nested agents as tools
    nested_input_tokens: int = 0
    nested_output_tokens: int = 0
    nested_cost_usd: float = 0.0

class ErrorInfo(BaseModel):
    type: str
    message: str
    retryable: bool = False

**rules**

- ``Envelope`` is the single data type flowing between engines. Every
  engine receives an Envelope and returns an Envelope.
- ``text()`` returns ``payload`` as a string (``str`` verbatim, Pydantic
  models as JSON, other types via ``json.dumps``). Use it when you want a
  plain string regardless of the payload shape.
- ``Envelope[T]`` narrows the payload type for mypy / pyright. Untyped
  ``Envelope`` is equivalent to ``Envelope[Any]`` and stays the default.
- ``ok`` is ``True`` iff ``error is None``. Always check ``ok`` before
  reading ``payload`` in production code.
- ``Envelope.error_envelope(exc)`` is the canonical way for engines to
  convert an exception into an envelope without raising up the stack.

**example**

```python
from lazybridge import Agent
from pydantic import BaseModel

class Article(BaseModel):
    title: str
    body: str

env = Agent("claude-opus-4-7", output=Article)("write a one-paragraph article on bees")

# Branch on success / failure.
if env.ok:
    print(env.payload.title)
    print(env.payload.body)
else:
    print(f"failed ({env.error.type}): {env.error.message}")

# Observability without a Session — metadata is always populated.
m = env.metadata
print(f"cost=${m.cost_usd:.4f}  in={m.input_tokens}  out={m.output_tokens}")

# Typed: the static checker knows env.payload is an Article.
def process(env: "Envelope[Article]") -> str:
    return env.payload.title
```

**pitfalls**

- ``payload`` can legitimately be ``None`` (e.g. when ``error`` is set or
  when the engine produced no content). Use ``env.ok`` or ``env.text()``
  if you want a safe string.
- ``Envelope.from_task(task)`` sets ``payload=task`` for convenience so
  the very first agent in a chain sees the input as both ``task`` and
  ``payload``. Downstream steps see the preceding step's ``payload``.
- ``nested_*`` fields in metadata are plumbed but not always populated
  yet; for accurate cross-agent cost, query ``session.usage_summary()``.

**see-also**

[agent](agent.md), [session](session.md), [sentinels](sentinels.md)
