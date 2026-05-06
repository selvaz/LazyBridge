# LazyBridge — Mid tier
**Use this when** you need conversation memory, shared key-value state, request/response tracing, guardrails, linear multi-agent chains, a simple human approval gate, or to wire in an MCP server as a tool catalogue.

**Move to Full when** your pipeline has conditional branching, typed hand-offs between steps, or crash-resume requirements.

## Memory

**signature**

Memory(
    *,
    strategy: Literal["auto", "sliding", "summary", "none"] = "auto",
    max_tokens: int | None = 4000,         # token budget that triggers compression
    max_turns: int | None = 1000,          # hard cap on retained turns (memory backstop)
    store: Store | None = None,            # reserved — durable persistence (1.1+)
    summarizer: Agent | Callable | None = None,
    summarizer_timeout: float | None = 30.0,  # deadline applied to async summarisers
) -> Memory

memory.add(user: str, assistant: str, *, tokens: int = 0) -> None
memory.messages() -> list[Message]
memory.text() -> str           # current view as plain text (live read)
memory.clear() -> None

Usage: Agent("model", memory=Memory("auto"))
       Agent("model", sources=[mem])     # share live view across agents

**rules**

- ``auto`` — sliding window plus summary of older turns once
  ``max_tokens`` is exceeded; default. Good for general chat.
- ``sliding`` — compress by dropping oldest turns whenever > 10 turns are
  kept. Does NOT require ``max_tokens``; works with ``max_tokens=None``.
- ``summary`` — compress whenever > 10 turns are kept.
  Uses ``summarizer=`` if provided; otherwise falls back
  to keyword extraction (a rough but loss-aware fallback — never a
  silent no-op).
- ``none`` — never compress; ``max_turns`` is the only backstop.
- Failed structured-output retries (internal ``_validate_and_retry``
  loops) pass ``memory=None`` so correction turns are never stored as
  real conversation history.
- ``Memory`` is per-agent by default. Share across agents by passing
  the same instance to each ``memory=`` or via ``sources=[mem]``.
- ``text()`` is live — every call re-materialises the current view.
  Do not snapshot and cache it.
- ``summarizer_timeout`` only enforces a deadline when the summariser
  returns a coroutine / awaitable. Sync summarisers cannot be
  cancelled mid-call; on timeout the keyword fallback runs.
- Compression runs OUTSIDE the internal lock — concurrent ``add()``
  calls keep progressing while a slow summariser is in flight.

**example**

```python
from lazybridge import Agent, Memory

mem = Memory(strategy="auto", max_tokens=3000)
chat = Agent("claude-opus-4-7", memory=mem, name="chat")

chat("hi, I'm Marco")
chat("what's my name?")         # "Marco"
print(mem.text())               # current compressed view

# LLM-summarised memory with explicit fallback timeout.
summariser = Agent("claude-haiku-4-5-20251001",
                   system="Summarize conversations concisely.")
mem = Memory(strategy="summary", summarizer=summariser, summarizer_timeout=15.0)

# Share memory across two agents — the judge reads the live history.
judge = Agent("claude-opus-4-7", name="judge",
              sources=[mem],
              system="Grade the assistant's last reply on helpfulness 1-5.")
judge("grade the last turn")
```

**pitfalls**

- ``Memory(strategy="summary")`` without a ``summarizer=`` agent uses
  the keyword-extraction fallback — bounded, but lossy. Pass a cheap
  agent for production-quality summaries.
- ``memory.clear()`` wipes everything including the in-process summary;
  it does not persist across restarts. For durable memory use ``Store``.
- ``max_turns`` is a hard backstop, not the primary compression knob.
  When it fires you get a one-shot warning — that's the signal to
  switch from ``strategy="none"`` to ``"auto"``.
- Setting ``summarizer_timeout=None`` restores the legacy unbounded
  behaviour. Do this only if you're confident your summariser is fast
  and reliable.

## Store

**signature**

Store(db: str | None = None) -> Store
  # db=None   → in-memory (lost on exit)
  # db="file" → SQLite WAL-mode, thread-safe, persistent

store.write(key: str, value: Any, *, agent_id: str | None = None) -> None
store.read(key: str, default: Any = None) -> Any
store.read_entry(key: str) -> StoreEntry | None
store.read_all() -> dict[str, Any]
store.keys() -> list[str]
store.delete(key: str) -> None
store.clear() -> None
store.to_text(keys: list[str] | None = None) -> str   # for sources=

StoreEntry = dataclass(key, value, written_at, agent_id)

**rules**

- Values are JSON-encoded on write (via ``json.dumps(default=str)``),
  so non-JSON types are stringified. Prefer primitives + Pydantic models
  (use ``.model_dump()`` before writing).
- ``to_text()`` renders the store as ``key: <json>`` lines, designed for
  ``Agent(sources=[store])`` live-view injection.
- Store is thread-safe via a lock (in-memory) or SQLite WAL mode + busy
  timeout (persistent). Safe to share across concurrent agents.
- Store is not transactional; each write commits immediately.
- ``Store()`` (in-memory) returns a **deep copy** from ``read()`` and
  stores a deep copy on ``write()``, matching the SQLite path's
  copy-on-write semantics. Do not rely on reference identity from
  ``store.read()`` — mutating the returned value does not affect the store.

**example**

```python
from lazybridge import Agent, Store, Plan, Step

store = Store(db="research.sqlite")

# Plan step writes a result into the store automatically.
plan = Plan(
    Step(researcher, name="search", writes="hits"),
    Step(writer,     name="write"),
)
Agent.from_engine(plan)("AI trends")
print(store.read("hits"))

# Agent with sources= sees the live store on every call.
monitor = Agent("claude-opus-4-7", name="monitor", sources=[store],
                system="Report what's currently in the blackboard.")
print(monitor("status?").text())
```

**pitfalls**

- ``Store(db=":memory:")`` is NOT the same as ``Store()`` — the former
  opens an in-memory SQLite (connection-scoped), the latter uses a
  Python dict. Use ``Store()`` unless you specifically need SQLite
  semantics in-process.
- Large binary blobs go through JSON serialisation; for files use a
  filesystem path as the value and read the file when you need it.
- ``store.to_text()`` can be expensive for stores with thousands of keys;
  pass ``keys=[...]`` to limit the slice.

## Session & tracing

**signature**

Session(
    *,
    db: str | None = None,            # None = in-memory SQLite (per session_id)
    exporters: list[EventExporter] = None,
    redact: Callable[[dict], dict] | None = None,
    redact_on_error: Literal["fallback", "strict"] = "strict",
    console: bool = False,            # add a ConsoleExporter
    # Batched-writer (opt-in) — submit events from the hot path,
    # let a background thread INSERT in batches.
    batched: bool = False,
    batch_size: int = 100,
    batch_interval: float = 1.0,
    max_queue_size: int = 10_000,
    on_full: Literal["drop", "block", "hybrid"] = "hybrid",
    critical_events: frozenset[str] | None = None,  # None = framework default
) -> Session

session.emit(event_type: EventType, payload: dict, *, run_id: str = None) -> None
session.add_exporter(exporter: EventExporter) -> None
session.remove_exporter(exporter: EventExporter) -> None
session.flush(timeout: float = 5.0) -> None        # drain batched writer
session.close() -> None                            # release SQLite + flush exporters
session.usage_summary() -> {"total": {...}, "by_agent": {...}, "by_run": {...}}

session.events: EventLog          # session.events.query(...) for raw rows
session.graph:  GraphSchema       # auto-populated when Agents register

EventType (StrEnum):
  AGENT_START  AGENT_FINISH
  LOOP_STEP
  MODEL_REQUEST  MODEL_RESPONSE
  TOOL_CALL  TOOL_RESULT  TOOL_ERROR
  HIL_DECISION

# Agent("model", verbose=True) creates a private Session(console=True).

**rules**

- Every engine emits events with the same enum. Hand an Agent a
  ``session=`` and you get a full per-run trace.
- ``redact`` runs on every payload. ``redact_on_error="strict"``
  (default) drops the event when the redactor raises or returns a
  non-dict — fail-closed. ``"fallback"`` warns once, then records the
  unredacted payload.
- Nested Agents (Agent A has Agent B as a tool) inherit the outer
  session. All events flow to one EventLog so ``usage_summary()`` can
  aggregate cost across the whole tree.
- ``batched=True`` makes ``emit`` non-blocking. Saturation policy
  (``on_full=``):
    * ``"hybrid"`` — block on critical events (``AGENT_*`` /
      ``TOOL_*`` / ``HIL_DECISION``), drop ``LOOP_STEP`` /
      ``MODEL_REQUEST`` / ``MODEL_RESPONSE``. **Default.**
    * ``"block"`` — back-pressure unconditionally.
    * ``"drop"``  — drop on saturation with a doubling-interval warning.
- ``critical_events=`` overrides the hybrid set. Pass an empty set to
  get drop-all-on-saturation behaviour while keeping hybrid as the policy.
- Exporters fire in registration order. A failing exporter warns once
  per instance; subsequent failures from the same exporter are
  suppressed.

**example**

```python
from lazybridge import Agent, Session, ConsoleExporter, JsonFileExporter
from lazybridge.session import EventType
from lazybridge.ext.otel import OTelExporter

# Dev — stdout tracing with one flag.
sess = Session(console=True)
Agent("claude-opus-4-7", name="chat", session=sess)("hello")

# Prod — multi-sink observability with batched writer.
sess = Session(
    db="events.sqlite",
    batched=True,
    on_full="hybrid",                 # default; explicit for clarity
    exporters=[
        JsonFileExporter(path="events.jsonl"),
        OTelExporter(endpoint="http://otelcol:4318"),
    ],
    redact=lambda p: {**p, "task": _mask_pii(p.get("task", ""))},
)
agents = [researcher, writer]
pipeline = Agent.chain(*agents, session=sess)
pipeline("summarise AI trends")

# Cost / token roll-up across the whole tree.
print(sess.usage_summary()["total"]["cost_usd"])

# Drain the writer before reading the log (or use a context manager).
sess.flush()
print(sess.events.query(event_type=EventType.TOOL_ERROR))

# Topology for a UI.
print(sess.graph.to_json())
```

**pitfalls**

- ``Session(db=":memory:")`` behaves like ``Session()`` (in-memory).
  Use a filename to persist.
- Exporter failures warn ONCE per exporter instance. If a third
  failure mode shows up, you'll only see the first — wrap in
  ``CallbackExporter(fn=lambda e: print(e))`` while debugging.
- ``Agent(verbose=True)`` creates a **new** Session for that agent.
  If you also pass ``session=another``, ``verbose`` is ignored.
- With ``batched=True`` reads via ``session.events.query()`` may be
  stale until ``session.flush()`` (or ``close()``) drains the writer.
- ``on_full="drop"`` was the pre-1.0.x default and is still available;
  the 1.0.x release flipped the default to ``"hybrid"`` so an
  ``AGENT_FINISH`` or ``TOOL_ERROR`` is never silently lost.

## Guards

**signature**

class Guard:
    async def acheck_input(self, text: str) -> GuardAction
    async def acheck_output(self, text: str) -> GuardAction

GuardAction(allowed: bool = True, message: str = None, modified_text: str = None,
            metadata: dict = {})

ContentGuard(input_fn: Callable[[str], GuardAction] = None,
             output_fn: Callable[[str], GuardAction] = None)
GuardChain(*guards: Guard)                 # first blocker wins
LLMGuard(judge: Agent, policy: str, *, timeout: float | None = 60.0)
  # LLM-as-judge; timeout applies to BOTH sync and async paths.
  # Sync path: daemon thread + join(timeout=). Async path: asyncio.wait_for.
  # On timeout the guard fails closed (blocked). timeout=None → unbounded.

class GuardError(Exception)                # raised by some integrations

Usage: Agent("model", guard=GuardChain(my_filter, LLMGuard(judge, "no PII")))

**rules**

- ``acheck_input`` runs BEFORE the engine. If ``allowed=False`` the
  engine is never invoked and an error Envelope is returned.
- ``acheck_output`` runs AFTER the engine on ``Envelope.text()``. If
  blocked, the output is replaced with an error Envelope (type
  ``GuardBlocked``).
- ``modified_text`` lets a guard rewrite its input — input rewrites
  become the engine's task; output rewrites replace the payload string.
- ``GuardChain`` short-circuits on the first ``allowed=False``.
- ``Agent.stream()`` calls ``acheck_input`` before emitting the first
  token. A blocked task raises ``ValueError`` instead of silently
  streaming — guards are enforced on the streaming path too.

**example**

```python
from lazybridge import Agent, ContentGuard, GuardChain, LLMGuard, GuardAction
import re

# Cheap regex guard.
def no_emails(text: str) -> GuardAction:
    if re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text):
        return GuardAction(allowed=False, message="Remove email addresses first.")
    return GuardAction(allowed=True)

# LLM-backed policy guard.
judge = Agent("claude-opus-4-7", name="judge",
              system='Respond "approved" or "rejected: <reason>".')

guard = GuardChain(
    ContentGuard(input_fn=no_emails),
    LLMGuard(judge, policy="Reject outputs that contain medical advice."),
)

bot = Agent("claude-opus-4-7", guard=guard, name="bot")
env = bot("my email is foo@bar.com, what's the weather?")
assert not env.ok                        # blocked by the regex guard
print(env.error.message)
```

**pitfalls**

- A guard that raises instead of returning ``GuardAction`` aborts the
  run. Return ``GuardAction(allowed=False, message=str(e))`` on error
  to keep pipelines resilient.
- Compose guards: put a cheap regex ``ContentGuard`` first; fall back to
  ``LLMGuard`` only for what regex can't handle. Saves tokens.
- Guards see ``Envelope.text()``, not the typed payload. If you're using
  structured output, the guard operates on the JSON serialisation.

## Agent.chain

**signature**

Agent.chain(*agents: Agent, name: str = "chain", **kwargs) -> Agent

Under the hood: Plan(*[Step(target=a, name=a.name) for a in agents]).
Sequential. Each agent's output becomes the next agent's task
(``from_prev`` default). Result: the last agent's Envelope.

Alternatives:
  Plan(Step(a), Step(b))                    # same thing, more explicit
  Plan(Step(a, name="a"),
       Step(b, name="b", task=from_start))  # b gets the ORIGINAL task, not a's output

**rules**

- ``Agent.chain`` is sugar for a linear ``Plan``. Use it when you have
  a straight line and no need for typed hand-offs or routing.
- Memory / session / guards on the chain wrapper apply at the outer
  boundary; individual agents keep their own.
- For fan-out on the same task see ``Agent.parallel``. For typed /
  conditional flows see ``Plan``.

**example**

```python
from lazybridge import Agent, Memory

researcher = Agent("claude-opus-4-7", name="researcher", tools=[search])
editor     = Agent("claude-opus-4-7", name="editor")
writer     = Agent("claude-opus-4-7", name="writer")

# Each agent's output becomes the next agent's task (from_prev default).
# Memory("auto") keeps the running transcript in the chain's context window.
agents = [researcher, editor, writer]
pipeline = Agent.chain(*agents, memory=Memory("auto"))   # construction
print(pipeline("AI trends April 2026").text())            # invocation → Envelope → text
```

**pitfalls**

- ``Agent.chain`` does not preserve typed outputs between steps — the
  next step sees the previous step's ``Envelope.text()``. If step N
  produces a Pydantic model and step N+1 needs it as a model, use
  ``Plan`` instead so you can declare ``output=``.
- The outer chain Agent has its own name ("chain" by default); set
  ``name="…"`` if you want it to appear distinctly in ``Session.graph``.

## Agent.as_tool

**signature**

agent.as_tool(
    name: str | None = None,
    description: str | None = None,
    *,
    verify: Agent | Callable[[str], Any] | None = None,
    max_verify: int = 3,
) -> Tool

# Tool schema: (task: str) -> str

Usage: Agent("model", tools=[researcher.as_tool()])
       Agent("model", tools=[researcher])   # implicit — Agent auto-wraps via as_tool()

**rules**

- ``as_tool()`` with no arguments produces a Tool named after the agent.
  Passing ``name`` / ``description`` overrides them.
- ``verify=`` turns the tool into a judge-gated call: every invocation
  runs up to ``max_verify`` times against the judge, retrying with the
  judge's feedback injected into the task. This is the "Option B"
  placement — the judge sits at the tool-call boundary.
- Passing an ``Agent`` directly to ``tools=[...]`` is equivalent to
  passing ``agent.as_tool()``.
- Nested agents inherit the outer session and register an ``as_tool``
  edge in the graph automatically (see Agent docs).

**example**

```python
from lazybridge import Agent

researcher = Agent("claude-opus-4-7", name="researcher", tools=[search])
judge      = Agent("claude-opus-4-7", name="judge",
                   system='Respond "approved" or "rejected: <reason>".')

# Implicit: pass the agent, LazyBridge wraps it.
orchestrator = Agent("claude-opus-4-7",
                     tools=[researcher])   # equivalent to researcher.as_tool()

# Explicit + verified: the judge gates every research call.
orchestrator = Agent("claude-opus-4-7",
                     tools=[researcher.as_tool(
                         name="research",
                         description="Find 3 high-quality sources for a topic.",
                         verify=judge, max_verify=2,
                     )])
```

**pitfalls**

- A misplaced ``verify=`` can cause a feedback loop if the judge is too
  strict; ``max_verify=2`` is a good default ceiling.
- Long nested chains (``Agent → Agent → Agent``) should share one
  ``Session`` — pass ``session=sess`` on the outer agent only and the
  inner ones will inherit it, so ``usage_summary()`` aggregates
  everything into one view.
- ``as_tool()``'s default schema is ``(task: str) -> str`` regardless of
  the wrapped agent's ``output=``. If you need a typed payload in the
  caller, orchestrate via ``Plan`` with ``Step(output=Model)`` instead.

## Agent.parallel

**signature**

Agent.parallel(
    *agents: Agent,
    concurrency_limit: int | None = None,   # semaphore
    step_timeout: float | None = None,      # per-agent wait_for
    **kwargs,                                # name, description, session
) -> _ParallelAgent

parallel_agent(task) -> list[Envelope]   # one entry per input agent, order preserved

**rules**

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

**example**

```python
from lazybridge import Agent

us   = Agent("claude-opus-4-7", name="us",   tools=[search_us])
eu   = Agent("claude-opus-4-7", name="eu",   tools=[search_eu])
asia = Agent("claude-opus-4-7", name="asia", tools=[search_asia])

# All three receive the same task; run concurrently; results arrive in input order.
# Use this for deterministic fan-out — not LLM-directed dispatch (use tools=[] for that).
agents = [us, eu, asia]
results = Agent.parallel(*agents,
                          concurrency_limit=3,   # cap simultaneous in-flight calls
                          step_timeout=30.0)("AI policy news")

for env in results:   # list[Envelope], one per agent
    print(env.metadata.model, env.text()[:100])
```

**pitfalls**

- The helper returns ``list[Envelope]``, not an ``Envelope``.  If you
  need a single aggregated answer, feed the list into a summariser
  agent as a follow-up.
- ``concurrency_limit=`` caps simultaneous in-flight calls; default
  (``None``) fires everything at once. Use a cap when you're rate-
  limit sensitive.
- ``step_timeout=`` wraps each per-agent call in ``asyncio.wait_for``.
  Timeouts return an error Envelope in the slot, preserving the
  positional contract.

## HumanEngine (ext.hil)

**signature**

HumanEngine(
    *,
    ui: Literal["terminal", "web"] | _UIProtocol = "terminal",
    timeout: float | None = None,
    default: str | None = None,
) -> Engine

Usage: Agent(engine=HumanEngine(), tools=[...], output=Pydantic)

# When output= is a Pydantic model, the terminal UI prompts field-by-field.

**rules**

- ``HumanEngine`` prompts the human for input and returns it as an
  Envelope. It implements the same ``Engine`` protocol as ``LLMEngine``,
  so ``Agent(engine=HumanEngine())`` is a drop-in replacement.
- ``output=SomeModel`` switches to per-field prompting (terminal UI).
- ``timeout`` triggers ``default`` if set, else raises ``TimeoutError``.
- Tool invocation is NOT handled by HumanEngine — the human types a
  raw string, they don't call tools interactively. If you want the
  human to call tools, use ``SupervisorEngine``.

**example**

```python
from lazybridge import Agent
from lazybridge.ext.hil import HumanEngine
from pydantic import BaseModel

class Review(BaseModel):
    approved: bool
    comment: str
    rating: int

reviewer = Agent(
    engine=HumanEngine(timeout=120, default="no comment"),
    output=Review,
    name="reviewer",
)

# In a pipeline: draft → review → finalise.
pipeline = Agent.chain(drafter, reviewer, finaliser)
pipeline("draft the release notes")
```

**pitfalls**

- The terminal UI blocks the current process. In a web app, supply a
  custom ``ui=`` adapter implementing ``prompt(task, *, tools,
  output_type) -> str``.
- ``timeout=`` uses the event loop, not signals; it works in async
  contexts but may hang in tightly-blocking sync nests.

## EvalSuite (ext.evals)

**signature**

EvalCase(
    input: str,
    check: Callable[[str], bool] | Callable[[str, Any], bool],
    expected: Any = None,
    description: str = "",
)

EvalSuite(*cases: EvalCase)
suite.run(agent) -> EvalReport
await suite.arun(agent) -> EvalReport

EvalReport(results: list[EvalResult])
  .total, .passed, .failed, .errors

# Built-in checks
exact_match(expected: str) -> Callable
contains(substring: str) -> Callable
not_contains(substring: str) -> Callable
min_length(n: int) -> Callable
max_length(n: int) -> Callable
llm_judge(agent: Agent, criteria: str) -> Callable   # cheap Agent as judge

**rules**

- ``EvalSuite`` feeds each ``EvalCase.input`` to the agent, captures
  the output text, runs ``check(output)``. ``expected`` is metadata
  only — checks are closed over their expected values.
- ``check`` can return ``bool`` or raise. Raising counts as an error,
  not a failure.
- ``llm_judge`` accepts an Agent and a string policy; the judge
  evaluates output and must respond with ``"approved"`` to pass.

**example**

```python
from lazybridge import Agent
from lazybridge.ext.evals import EvalCase, EvalSuite, contains, llm_judge

bot   = Agent("claude-opus-4-7", system="You are a helpful assistant.")
judge = Agent("claude-opus-4-7", name="judge",
              system='Respond "approved" or "rejected: <reason>".')

suite = EvalSuite(
    EvalCase("What's the capital of France?",
             check=contains("Paris")),
    EvalCase("Write a poem about bees.",
             check=llm_judge(judge,
                 "Output must be a poem of at least 4 lines mentioning bees.")),
    EvalCase("hello",
             check=lambda out: len(out) < 500,
             description="brevity check"),
)

report = suite.run(bot)
print(report)                      # "2/3 passed (66%)"
assert report.passed == report.total, [r.case.input for r in report.results if not r.passed]
```

**pitfalls**

- ``llm_judge`` costs tokens on every run; use a ``cheap`` tier agent.
- Evals test the agent's text output (``Envelope.text()``), not the
  typed payload. If you're evaluating a structured-output agent, the
  check sees the JSON serialisation.
- ``EvalSuite.run`` is synchronous; use ``arun`` in async test harnesses.

## MCP integration (ext.mcp)

**signature**

# Status: alpha (lazybridge.ext.mcp).  Install: pip install lazybridge[mcp].

from lazybridge.ext.mcp import MCP, MCPServer

MCP.stdio(
    name: str,
    *,
    command: str,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    namespace: bool = True,
    prefix: str | None = None,
    allow: Iterable[str] | None = None,
    deny: Iterable[str] | None = None,
    cache_tools_ttl: float | None = 60.0,    # tool-list cache lifetime
) -> MCPServer

MCP.http(
    name: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    namespace: bool = True,
    prefix: str | None = None,
    allow: Iterable[str] | None = None,
    deny: Iterable[str] | None = None,
    cache_tools_ttl: float | None = 60.0,
) -> MCPServer

MCP.from_transport(
    name: str,
    transport: _Transport,
    *,
    namespace: bool = True,
    prefix: str | None = None,
    allow: Iterable[str] | None = None,
    deny: Iterable[str] | None = None,
    cache_tools_ttl: float | None = 60.0,
) -> MCPServer

# MCPServer behaves like a tool provider — drop into Agent(tools=[...])
# and it expands into one Tool per MCP tool.
server.invalidate_tools_cache() -> None
async with server:        # explicit lifecycle: connect + close
    ...

**rules**

- An ``MCPServer`` is a *tool provider*; pass it directly to
  ``Agent(tools=[server])``.  Agent expansion calls
  ``server.as_tools()`` to expand it into one ``Tool`` per MCP tool.
  No separate ``MCPEngine`` / ``MCPProvider`` exists.
- The transport connects **lazily** on the first ``as_tools()`` call,
  which is normally Agent construction time.  Connection failures
  surface there — fail-fast.
- Default tool naming: ``"<server-name>.<mcp-tool-name>"``.  Pass
  ``namespace=False`` to keep raw names, or ``prefix="..."`` to
  override.
- ``allow`` / ``deny`` use shell-style globs (``fnmatch``) against the
  full namespaced name.  ``"github.delete_*"``, not regex.
- The discovered-tools cache lives ``cache_tools_ttl`` seconds
  (default 60).  An MCP server that hot-loads or unloads tools is
  reflected on the next call past the TTL.  Pass
  ``cache_tools_ttl=None`` to disable expiry, or call
  ``server.invalidate_tools_cache()`` on an out-of-band signal.
- Closure is **terminal**.  After ``aclose()`` (or exiting an
  ``async with`` block) the server cannot be reconnected — construct
  a new one if you need to.
- The MCP SDK is an optional dependency.  Importing
  ``lazybridge.ext.mcp`` is cheap; constructing an
  ``MCP.stdio(...)`` / ``MCP.http(...)`` is when the SDK gets imported
  and raises a clean ``ImportError`` if missing.

**example**

```python
from lazybridge import Agent
from lazybridge.ext.mcp import MCP

# 1) Spawn a stdio MCP server (subprocess) and use its tools.
fs = MCP.stdio(
    "fs",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp/project"],
)
agent = Agent("claude-opus-4-7", tools=[fs])
agent("Read README.md and summarise the install steps")

# 2) Mix MCP with custom tools and other agents.
def estimate_cost(plan: str) -> float:
    """Estimate the cost in USD of executing ``plan``."""
    return 0.0

planner = Agent(
    "claude-opus-4-7",
    tools=[fs, estimate_cost],
    name="planner",
)

# 3) Allow / deny lists keep dangerous tools out of reach.
fs_safe = MCP.stdio(
    "fs",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp/project"],
    allow=["fs.list_*", "fs.read_*"],
    deny=["fs.delete_*"],
)

# 4) Refresh the tool list explicitly when an upstream plugin is
#    installed mid-process.
fs.invalidate_tools_cache()

# 5) Explicit lifecycle (rare; the transport otherwise lives until
#    process exit).
async with MCP.stdio("fs", command="...") as fs:
    agent = Agent("claude-opus-4-7", tools=[fs])
    await agent.run("...")
```

**pitfalls**

- Tool-name collisions across servers are real.  Default namespacing
  prevents them; don't disable it casually.
- ``allow`` / ``deny`` patterns match the **namespaced** name; write
  ``"github.delete_*"``, not ``"delete_*"``.
- Lazy connect surfaces transport errors at ``Agent(tools=[server])``
  time, not at first user query.  If the underlying subprocess won't
  start, you'll see the error during agent construction.
- ``cache_tools_ttl=None`` (legacy behaviour) caches forever — fine
  for static MCP servers, dangerous for hot-loaded ones.
- An MCP tool's JSON Schema is published by the server; LazyBridge
  uses it directly via ``Tool.from_schema``.  If the schema is
  malformed, the model will fail the tool call with the new
  ``ToolArgumentParseError`` shape rather than silently coerce.
