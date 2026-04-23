# LazyBridge — Mid tier
Realistic apps. Conversation memory, shared state, console/verbose
logging, input/output guardrails, simple chain or parallel fan-out,
one agent re-used as a tool, basic human-in-the-loop, evals.
No explicit DAG — for that, go to Full.

## Memory

**signature**

Memory(
    strategy: Literal["auto", "sliding", "summary", "none"] = "auto",
    max_tokens: int = 4000,
    window: int = 10,            # sliding: last N turns kept raw
    summarizer: Agent | None = None,  # cheap agent used when strategy="summary"
) -> Memory

memory.add(user: str, assistant: str, tokens: int = 0) -> None
memory.messages() -> list[Message]
memory.text() -> str              # current view as plain text (live read)
memory.clear() -> None

Usage: Agent("model", memory=Memory("auto"))

**rules**

- ``auto`` — sliding window plus summary of older turns once ``max_tokens``
  is exceeded; default. Good for general chat.
- ``sliding`` — keep last ``window`` turns verbatim, drop the rest.
  Lossy but cheap.
- ``summary`` — compress everything with ``summarizer`` on each overflow.
  Requires passing a cheap Agent.
- ``none`` — do not compress; raises memory over time.
- ``Memory`` is per-agent by default. To share memory across agents, pass
  the same instance to each agent's ``memory=`` or via ``sources=[mem]``.
- ``text()`` is live — every call re-materialises the current view. Do
  not snapshot and cache it.

**example**

```python
from lazybridge import Agent, Memory

mem = Memory("auto", max_tokens=3000)
chat = Agent("claude-opus-4-7", memory=mem, name="chat")

chat("hi, I'm Marco")
chat("what's my name?")         # "Marco"
print(mem.text())               # current compressed view

# Share memory across two agents — the judge reads the live history.
judge = Agent("claude-opus-4-7", name="judge",
              sources=[mem],
              system="Grade the assistant's last reply on helpfulness 1-5.")
judge("grade the last turn")
```

**pitfalls**

- ``Memory(strategy="summary")`` without a ``summarizer=`` agent falls
  back to a no-op and grows unboundedly.
- ``memory.clear()`` wipes everything including the in-process summary;
  it does not persist across restarts. For durable memory use ``Store``.

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
    db: str | None = None,            # None = in-memory SQLite
    exporters: list[EventExporter] = None,
    redact: Callable[[dict], dict] | None = None,
    console: bool = False,             # install a ConsoleExporter for stdout tracing
) -> Session

session.emit(event_type: EventType, payload: dict, *, run_id: str = None) -> None
session.add_exporter(exporter: EventExporter) -> None
session.remove_exporter(exporter: EventExporter) -> None
session.usage_summary() -> {"total": {...}, "by_agent": {...}, "by_run": {...}}

session.events: EventLog
session.graph:  GraphSchema          # auto-populated when Agents register

EventLog.record(event_type, payload, *, run_id) -> None
EventLog.query(*, run_id=None, event_type=None) -> list[dict]

EventType (StrEnum):
  AGENT_START  AGENT_FINISH
  LOOP_STEP
  MODEL_REQUEST  MODEL_RESPONSE
  TOOL_CALL  TOOL_RESULT  TOOL_ERROR

Shortcut: Agent("model", verbose=True) creates a private Session(console=True).

**rules**

- Every engine emits events with the same 8-type enum. Hand an Agent
  a ``session=`` and you get a full per-run trace.
- ``redact`` is called on every payload before recording / exporting;
  use it for PII scrubbing.
- Nested Agents (Agent A has Agent B as a tool) inherit the outer
  session. All events flow to one EventLog so ``usage_summary()`` can
  aggregate cost across the whole tree.
- Exporters fire in registration order on every emit. Exceptions raised
  by one exporter do not block others.

**example**

```python
from lazybridge import Agent, Session, ConsoleExporter, JsonFileExporter

# Dev — stdout tracing with one flag.
sess = Session(console=True)
Agent("claude-opus-4-7", name="chat", session=sess)("hello")

# Prod — multi-sink observability.
sess = Session(
    db="events.sqlite",
    exporters=[
        JsonFileExporter("events.jsonl"),
        ConsoleExporter(),
    ],
    redact=lambda p: {**p, "task": _mask_pii(p.get("task", ""))},
)
pipeline = Agent.chain(researcher, writer, session=sess)
pipeline("summarise AI trends")

# Observability summary.
summary = sess.usage_summary()
print(summary["total"]["cost_usd"])
print(summary["by_agent"]["researcher"]["input_tokens"])

# Topology for a UI.
print(sess.graph.to_json())
```

**pitfalls**

- ``Session(db=":memory:")`` behaves like ``Session()`` (in-memory).
  Use a filename to persist.
- Exporter failures are caught silently. If an exporter looks like it's
  doing nothing, wrap it in ``CallbackExporter(lambda e: print(e))`` to
  see what's arriving.
- ``Agent(verbose=True)`` creates a **new** Session for that agent; if
  you also pass ``session=another``, ``verbose`` is ignored (the
  explicit session wins).

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
LLMGuard(judge: Agent, policy: str)        # LLM-as-judge

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

pipeline = Agent.chain(researcher, editor, writer,
                        memory=Memory("auto"))
print(pipeline("AI trends April 2026").text())
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
       Agent("model", tools=[researcher])   # implicit — wrap_tool auto-calls as_tool()

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

us   = Agent("claude-opus-4-7", name="us", tools=[search_us])
eu   = Agent("claude-opus-4-7", name="eu", tools=[search_eu])
asia = Agent("claude-opus-4-7", name="asia", tools=[search_asia])

results = Agent.parallel(us, eu, asia,
                          concurrency_limit=3,
                          step_timeout=30.0)("AI policy news")

for env in results:
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

## HumanEngine

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
from lazybridge import Agent, HumanEngine
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

## EvalSuite

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
from lazybridge import Agent, EvalCase, EvalSuite, contains, llm_judge

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
