# LazyBridge — Basic tier
**Use this when** you need a single LLM call — with or without tools, with or without structured output. No setup beyond an API key.

**Move to Mid when** you need memory across calls, shared state, tracing, guardrails, or more than one agent in sequence.

## Agent

**signature**

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

**rules**

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

**example**

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

**pitfalls**

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

## Tool

**signature**

Tool(
    func: Callable,
    *,
    name: str | None = None,
    description: str | None = None,
    mode: Literal["signature", "llm", "hybrid"] = "signature",
    schema_llm: Any | None = None,
    strict: bool = False,
    returns_envelope: bool = False,
) -> Tool

Tool.from_schema(name, description, parameters, func, *, strict=False, returns_envelope=False) -> Tool
Tool.definition() -> ToolDefinition
await Tool.run(**kwargs) -> Any
Tool.run_sync(**kwargs) -> Any   # drives async ``func`` to completion

Agent.as_tool(name=None, description=None, *, verify=None, max_verify=3) -> Tool

# Six paths to a Tool — pick by what you have:
Agent(tools=[fn])                                     # plain Python function
Agent(tools=[Tool(fn, name=..., strict=True)])        # function + override
Agent(tools=[other_agent])                            # sub-agent (auto-wrapped)
Agent(tools=[other_agent.as_tool(verify=judge)])      # sub-agent + judge/retry
Agent(tools=[mcp_server])                             # MCPServer (provider expansion)
Agent(tools=read_docs_tools())                        # external_tools kit (list[Tool])

**rules**

- Schema generation is automatic from ``func``'s type hints and docstring
  in ``mode="signature"`` (default). Use ``mode="llm"`` to let an LLM
  synthesise the schema, ``mode="hybrid"`` for both.
- ``name`` defaults to ``func.__name__``. Names are API-facing; pick
  stable ones.
- ``strict=True`` enables provider-strict JSON-schema validation on tool
  arguments (Anthropic / OpenAI strict mode).
- ``run`` is async; ``run_sync`` auto-detects coroutine functions and
  drives them to completion so synchronous callers (e.g. ``SupervisorEngine``
  REPL) never see a raw coroutine.
- ``Agent(tools=[...])`` accepts callables, ``Tool`` instances, ``Agent``
  instances, and tool providers (objects with ``_is_lazy_tool_provider =
  True`` + ``as_tools()`` method). Everything is normalised to ``Tool``
  at construction; users never call a wrapper directly.

**example**

```python
from lazybridge import Agent, Tool

# 1. Plain function — type hints + docstring drive the schema.
def calculate(expression: str) -> float:
    """Evaluate a basic arithmetic expression and return the result.

    Supports +, -, *, /, parentheses.
    """
    return eval(expression)  # noqa: S307  (trusted inputs only)

Agent("claude-opus-4-7", tools=[calculate])("what is 17 * 23?")

# 2. Function + explicit configuration (override name, strictness, etc.).
calc_tool = Tool(calculate, name="calc", strict=True,
                 description="Evaluate an arithmetic expression.")
Agent("claude-opus-4-7", tools=[calc_tool])("...")

# 3. An Agent is also a Tool — no ceremony.
researcher = Agent("claude-opus-4-7", tools=[search], name="researcher")
orchestrator = Agent("claude-opus-4-7", tools=[researcher])

# 4. Agent + verifier (judge + retry).
judge = Agent("claude-opus-4-7",
              system="Reply 'approved' or 'rejected: <reason>'.")
orchestrator = Agent("claude-opus-4-7", tools=[
    researcher.as_tool(verify=judge, max_verify=2),
])

# 5. MCP server — drop in, framework expands via as_tools().
from lazybridge.ext.mcp import MCP
fs = MCP.stdio("fs", command="npx",
               args=["@modelcontextprotocol/server-filesystem", "."])
Agent("claude-opus-4-7", tools=[fs])

# 6. external_tools kit — factory returns list[Tool].
from lazybridge.external_tools.read_docs import read_docs_tools
Agent("claude-opus-4-7", tools=read_docs_tools())
```

**pitfalls**

- A function with no type hints produces an empty JSON schema and the
  LLM will not know how to call it. Always annotate parameters.
- A docstring is part of the contract the LLM reads. "Returns the
  weather" is weaker than "Returns the current temperature in Celsius
  and a one-word condition (sunny / cloudy / rainy) for ``city``."
- ``strict=True`` rejects optional / defaulted args under some providers;
  if a call fails with "unknown parameter", try ``strict=False``.
- Tool name collisions trigger a ``UserWarning`` — the second
  registration replaces the first. Pick stable, distinct names.
- Pydantic ``BaseModel`` parameters are coerced from the raw LLM dict to
  a model instance before the function is called — you always receive a
  typed object, not a plain dict.

## Native tools (web search, code execution, …)

**signature**

from lazybridge import Agent, NativeTool

Agent("model", native_tools=[NativeTool.WEB_SEARCH, ...]) -> Agent

# Accepted values (NativeTool enum):
#   WEB_SEARCH       — web search (Anthropic, OpenAI, Google)
#   CODE_EXECUTION   — sandboxed Python/JS (Anthropic, OpenAI)
#   FILE_SEARCH      — file search over uploaded content (OpenAI)
#   COMPUTER_USE     — screen control (Anthropic)
#   GOOGLE_SEARCH    — Google grounded search (Google)
#   GOOGLE_MAPS      — Google Maps grounding (Google)

# String aliases also accepted:  native_tools=["web_search", "code_execution"]

**rules**

- Native tools run server-side at the provider. You don't write or host
  them; you opt in by passing the enum.
- Not every tool is supported by every provider. Using
  ``NativeTool.GOOGLE_SEARCH`` on Anthropic raises at ``complete`` time.
  Match the tool to the model's provider.
- Native tools **coexist** with regular ``tools=[...]`` — the model may
  choose to call a native tool, one of your functions, both in the same
  turn, or neither.
- ``native_tools=`` is a shortcut on Agent equivalent to
  ``Agent(engine=LLMEngine(..., native_tools=[...]))``.
- Grounded responses from search tools expose sources via
  ``Envelope.metadata`` (``model``, ``provider``) and, where providers
  return them, via raw ``CompletionResponse.grounding_sources``.

**example**

```python
from lazybridge import Agent, NativeTool

# Web search is one line.
search = Agent("claude-opus-4-7", native_tools=[NativeTool.WEB_SEARCH])
print(search("what happened in AI news April 2026?").text())

# Native + custom tools coexist.
def read_report(path: str) -> str:
    """Read a local markdown file."""
    return open(path).read()

analyst = Agent(
    "claude-opus-4-7",
    native_tools=[NativeTool.WEB_SEARCH, NativeTool.CODE_EXECUTION],
    tools=[read_report],
)
analyst("cross-reference my report.md with current web consensus on the topic").text()

# Strings work too — equivalent to the enum.
Agent("gpt-4o", native_tools=["web_search"])("latest Python release?")
```

**pitfalls**

- Mixing ``NativeTool.GOOGLE_SEARCH`` with an Anthropic model fails at
  provider time, not at Agent construction. Match the enum to the
  provider before you ship.
- Some native tools (``COMPUTER_USE``) require additional setup or beta
  flags on the provider's API. Check the provider's current docs.
- Cost: native tool calls are billed by the provider (search queries,
  code execution time). They appear in ``Envelope.metadata.cost_usd``
  when the provider reports them.

## Function → Tool (schema modes)

**signature**

# Three ways to turn a Python function into an LLM-callable Tool.

Tool(func, *, mode: Literal["signature", "llm", "hybrid"] = "signature",
     schema_llm: Any | None = None, strict: bool = False)

# Mode recap:
#   "signature" — parse type hints + docstring (default). No LLM cost.
#   "llm"       — call an LLM to infer schema from the function body
#                 and docstring.  Needs schema_llm= (an Agent).
#   "hybrid"    — signature first; LLM fills gaps for missing hints.

# No explicit Tool(...) call needed:
Agent(..., tools=[func])                  # function auto-wrapped at construction
Agent(..., tools=[other_agent])           # other_agent.as_tool() called automatically
Agent(..., tools=[mcp_server])            # provider.as_tools() expanded automatically

**rules**

- ``mode="signature"`` is the default and produces a schema from type
  hints + docstring (parameter types, return type, description, tool
  name). No LLM is called. Fast, deterministic, free.
- ``mode="llm"`` calls ``schema_llm`` (a cheap Agent) to synthesise a
  JSON schema from the function source + docstring. Pays in tokens but
  works for functions with incomplete or missing hints.
- ``mode="hybrid"`` starts with ``"signature"`` and falls back to
  ``"llm"`` only for parameters lacking hints. Best of both when your
  codebase is mixed.
- The schema is cached per ``Tool`` instance (first ``.definition()``
  call computes it; subsequent calls reuse).
- ``strict=True`` asks the provider to enforce the schema exactly (no
  extra fields, no coercion). Available on Anthropic + OpenAI strict
  modes; increases reliability at the cost of some flexibility.

**example**

```python
from lazybridge import Agent, Tool

# --- Signature mode (default, no LLM) -----------------------------
def calculate(expression: str) -> float:
    """Evaluate a basic arithmetic expression and return the result.

    Supports +, -, *, /, parentheses.
    """
    return eval(expression)

Agent("claude-opus-4-7", tools=[calculate])   # schema auto-inferred

# --- LLM mode (schema synthesised by a cheap Agent) ---------------
from lazybridge import Agent

tiny = Agent.from_provider("anthropic", tier="cheap", name="schema_bot")

def legacy_func(data, opts=None):
    """Transform the incoming payload per options. data is a dict of
    readings {timestamp: value}; opts controls resampling.
    """
    ...

legacy_tool = Tool(legacy_func, mode="llm", schema_llm=tiny)
Agent("claude-opus-4-7", tools=[legacy_tool])

# --- Hybrid (signature where possible, LLM where missing) ---------
def partial_hint(query: str, opts=None) -> list:
    """Search and return matches. opts is a dict of filters."""
    ...

Agent("claude-opus-4-7",
      tools=[Tool(partial_hint, mode="hybrid", schema_llm=tiny)])
```

**pitfalls**

- ``mode="llm"`` without ``schema_llm=`` silently falls back to
  ``"signature"`` (with warnings). Always pass the schema_llm if you
  pick LLM / hybrid mode.
- Calling ``Tool(func).definition()`` forces the schema computation.
  If ``mode="llm"``, this triggers an LLM call at construction time —
  don't build tools on import if you're latency-sensitive.
- ``strict=True`` is opinionated about JSON schema shape. Tools that
  rely on extra kwargs or variadic args may fail strict validation;
  try without strict first.
- Pydantic ``BaseModel`` parameters work in ``mode="signature"``. The
  schema is inferred from the model's fields; the raw LLM dict is
  coerced into a model instance before your function is called.

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

# Agent(...) constructs; ("task") invokes → always returns an Envelope.
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

- ``Envelope.from_task(task)`` sets ``payload=task`` for convenience so
  the very first agent in a chain sees the input as both ``task`` and
  ``payload``. Downstream steps see the preceding step's ``payload``.
- ``nested_*`` fields in metadata are plumbed but not always populated;
  for accurate cross-agent cost, query ``session.usage_summary()``.
