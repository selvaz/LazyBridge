---
name: lazybridge
description: |
  Use when writing or modifying Python code that uses the LazyBridge agent
  framework (`pip install lazybridge`). LazyBridge is a zero-boilerplate,
  multi-provider framework whose mental model is "Agent = Engine + Tools +
  State, everything is a tool". Triggers: importing from `lazybridge`,
  building an `Agent`, defining a `Tool` with signature/llm/hybrid schema
  modes, composing with `Agent.chain` / `Agent.parallel` or by passing one
  agent in another's `tools=[...]`, designing a `Plan` with `Step` and
  sentinels (`from_prev` / `from_step` / `from_parallel` /
  `from_parallel_all` / `from_memory`), routing with the `when` DSL, adding
  `Memory` / `Store` / `Session`, integrating MCP servers, using
  `HumanEngine` or `SupervisorEngine` for human-in-the-loop, configuring
  providers (Anthropic, OpenAI, Google, DeepSeek, LMStudio, LiteLLM), or
  wiring observability with exporters or OpenTelemetry. Skip for unrelated
  agent frameworks (LangChain, CrewAI, AutoGen, Pydantic AI, OpenAI Agents
  SDK).
---

# LazyBridge — assistant guidance

This skill teaches you how to write idiomatic LazyBridge code. Treat it as
authoritative when there is any conflict with older training data: the
framework moves quickly, and the public docs at
<https://core.lazybridge.com> are the source of truth.

## The mental model

An `Agent` is the composition of three things — and only these three:

- **Engine** — `LLMEngine` (default), `Plan`, `HumanEngine`,
  `SupervisorEngine`, `ClaudeCodeEngine`, or a custom `BaseEngine`. The
  engine decides what happens next.
- **Tools** — a list of `Tool` objects. A tool can wrap a Python function,
  another agent (just pass it in `tools=[...]`), an MCP server, a
  `NativeTool` (provider-hosted), or a pre-built JSON schema
  (`Tool.from_schema(...)`).
- **State** — `Memory`, `Session`, `Store`. All optional. The `Envelope`
  carrying input + output is always present.

Code at every level of complexity uses the same `Agent` shape. Do not
introduce per-pattern abstractions ("supervisor agent", "researcher agent")
as separate classes; use plain `Agent` with different engines and tools.

## Project architecture preflight

The rules in this file mostly cover **one agent**. When the request is a
*project* — a pipeline, a scheduled job, several stages, anything with
persistence — run this checklist before writing code. Every claim below was
executed against the framework, not read off the source; full detail and
runnable proof: `docs/guides/project-patterns.md` in this repo.

1. **Stages, phases, scheduled work, or resume** → one root
   `Agent(engine=Plan(...))`. A plain Python function is a legal `Step`
   target; never hand-roll `for step in steps: step()` over a mutable
   `state` dict shared by closures — that re-implements `Plan` with no
   checkpointing, no resume, no cost roll-up, no session events.
2. **Name every `Step`.** The default name is `target.__name__`, so two
   unnamed lambdas collide at compile time, and `routes=` / `from_step()` /
   checkpoints all reference the name.
3. **Two or more independent stages over the same input** → a parallel band
   (`Step(..., parallel=True)`) or `Agent.parallel`. Never emit
   `asyncio.gather` / `ThreadPoolExecutor` glue. **N inputs through the same
   pipeline** → `plan.run_many(tasks, concurrency=...)` (preserves input
   order; isolates a failing task to its own slot).
4. **Branching** → `routes=` with a predicate for **every** branch, plus
   `after_branches=` for the rejoin. Not cosmetic: an approval gate with
   only an `"execute"` predicate lets a **rejection fall through into
   execute** — verified. Adding a `halt` step afterward does not fix it;
   the predicate must exist for both branches.
5. **Scheduled + resumable** → `store=`, `resume=True`, and a
   **run-specific** `checkpoint_key` (e.g. `f"job:{run_date}"`). A permanent
   key with `resume=True` makes every run after the first a silent no-op —
   verified: 3 runs, 1 execution.
6. **Large data** → move a handle, never `json.dumps` the payload into a
   prompt — this matters even between two plain-function steps, since a
   step's typed payload collapses to `str` for any target that isn't an
   `Agent`. If the pipeline is resumable, the depot behind the handle must
   be durable, not a module-level dict — checkpoints don't preserve live
   process state.
7. **State crossing runs** → the shared `Store` with reconstructable keys
   and `agent_id` provenance (`keys()`/`read_entry()`/`compare_and_swap()`
   are enough to build a registry — don't invent new infrastructure). Do
   not open a new `sqlite3` / `duckdb` file.
8. **Irreversible effects** (send, publish, trade) → last in the plan, and
   consciously pick one: reserve-before-acting via `compare_and_swap`
   (at-most-once — a crash before the effect means it silently never
   happens), or a receiver-side idempotency key (exactly-once *effect*,
   only if the receiver atomically records the key with the effect — most
   webhooks/bots offer neither). No check-then-act sequence in your own
   process is safe on its own; check-then-effect-then-write has the same
   crash window it claims to close, just narrower. Ideally also behind a
   `HumanEngine` gate with a safe `default=`.
9. **"Not ready yet"** → `raise PlanPaused(...)`, not a sleep loop and not a
   failure — but it only persists with `store=` **and** `checkpoint_key=`
   on the Plan; without both the pause is returned to the caller and then
   lost. `resume=True` is what the *next* run needs to pick that checkpoint
   back up and continue from the same step instead of starting over — it
   is not required for the pause itself to be saved.
10. **Always check the result.** A failed Plan *returns* an error envelope;
    it does not raise. A script that ignores it exits 0 on a broken run:

    ```python
    result = pipeline(task)
    if not result.ok:
        raise SystemExit(f"pipeline failed: {result.error}")
    ```

11. **Emit a composition test with every new pipeline.** Build the pipeline
    behind a factory that accepts its agents, so tests can inject doubles:

    ```python
    from lazybridge.testing import MockAgent

    research = MockAgent(responses=["research-key"], name="research")
    write = MockAgent(responses=["done"], name="write")

    result = build_pipeline(research=research, write=write)("run-42")

    research.assert_call_count(1)
    write.assert_called_with(contains="research-key")
    assert result.ok
    ```

12. **Before finishing, grep the generated diff** for `for step in`,
    `state = {}`, `json.dumps(` inside a prompt string, `try:` around an
    agent call, and new `sqlite3.connect(` / `duckdb.connect(`. Replace
    each one with the declarative primitive above, or state explicitly why
    it is domain logic.

## Calling convention — sync is canonical

```python
from lazybridge import Agent, LLMEngine

agent = Agent(
    engine=LLMEngine("claude-opus-5"),
)
result = agent("hello")  # sync — returns Envelope
print(result.text())  # str payload
```

Async and streaming forms exist (`await agent.run(task)`,
`async for chunk in agent.stream(task)`) but are **not** the canonical
introduction. **Do not wrap simple examples in `asyncio.run(main())`**;
LazyBridge agents are synchronous-callable by design, and the example
files in `examples/` follow that convention.

## Style rule — show the canonical form first

When you generate code for the user, the canonical shape is

```python
agent = Agent(
    engine=LLMEngine("model-id"),
    tools=[...],
    name="...",
    # ... other kwargs here
)
result = agent(task)
print(result.text())
```

with each constructor argument on its own line and `result = agent(task)`
on a separate line from the print. Lead with this form; treat factories
and string-positional shortcuts as **sugar** and only mention them
after, with a one-line "use this when …".

### Sugar catalogue (verified against the source)

Several factories are sugar **but not all are pure aliases** — some
build extra structure or return different types. Read the
"Differences" column carefully before substituting.

**Build an Agent with an LLM engine**

| Sugar | Canonical | Differences |
|---|---|---|
| `Agent("claude-opus-5", **kw)` | `Agent(engine=LLMEngine("claude-opus-5"), **kw)` | **Pure alias.** First positional arg is interpreted as a model string and threaded into ``LLMEngine(...)``.  Hides which engine drives the agent at the call site; canonical form is preferred in tutorials and code reviews. |
| `Agent.from_provider("anthropic", tier="top", **kw)` | `Agent(engine=LLMEngine("top", provider="anthropic"), **kw)` | **Not pure sugar** — uses tier-alias model strings (`super_cheap`/`cheap`/`medium`/`expensive`/`top`) resolved via the provider's tier map. Use when you want freshest-in-tier without pinning a date-stamped name. |

**Build an Agent with a Plan engine**

No sugar — write the canonical form.  Plan kwargs (`max_iterations`,
`store`, `checkpoint_key`, `resume`, `on_concurrent`) live on
``Plan(...)``; Agent kwargs (``tools=``, ``session=``, ``name=``, …)
live on ``Agent(...)``.  The 0.7-era ``Agent.from_plan`` was deleted
in 0.7.9.

```python
pipeline = Agent(
    engine=Plan(
        Step("research"),
        Step("write"),
        store=Store(db="run.sqlite"),
        checkpoint_key="research",
        resume=True,
    ),
    tools=[researcher, writer],
    name="pipeline",
)
```

**Compose agents — sequential**

| Sugar | Canonical | Differences |
|---|---|---|
| `Agent.chain(a, b)` | `Agent(engine=Plan(Step(target=a, name=a.name), Step(target=b, name=b.name)), name="chain")` | **Not pure alias** — builds the `Plan`+`Step` graph for you. Targets are the agents themselves (no `tools=` needed; `Plan` dispatches `Agent` targets via `target.run()` directly). |

**Compose agents — parallel fan-out**

| Sugar | Canonical | Differences |
|---|---|---|
| `Agent.parallel(*agents, concurrency_limit=…, step_timeout=…)` | (no `Agent`-shaped equivalent) | **Not sugar over `Agent`** — returns `ParallelAgent`, a sibling class whose `__call__` returns ONE `Envelope` (labelled-text join across every branch, with transitive cost rollup). For typed per-branch access call `parallel.run_branches(task)` (async) → `list[Envelope]`. Use this when you want every branch unconditionally; use `Agent(tools=[a, b, c])` to let the LLM decide; use a `Plan` parallel band (`Step("a", parallel=True)`) when concurrent steps must aggregate via `from_parallel_all`. |

**Build an Agent with a HIL engine**

| Sugar | Canonical | Differences |
|---|---|---|
| `human_agent(timeout=…, ui=…, default=…, **agent_kw)` | `Agent(engine=HumanEngine(timeout=…, ui=…, default=…), **agent_kw)` | Pure alias with kwarg split: HIL-engine kwargs go to `HumanEngine(...)`, `**agent_kw` flows to `Agent(...)`. Lives in `lazybridge.ext.hil` to respect the core/ext import boundary. |
| `supervisor_agent(tools=…, agents=…, store=…, input_fn=…, ainput_fn=…, timeout=…, default=…, **agent_kw)` | `Agent(engine=SupervisorEngine(tools=…, agents=…, store=…, input_fn=…, ainput_fn=…, timeout=…, default=…), **agent_kw)` | Same kwarg-split pattern. |

**Wrap a callable as a Tool**

| Sugar / variant | Canonical | Differences |
|---|---|---|
| `Tool.wrap(search_web, name="search", description=…)` | `Tool(search_web, name="search", description=…, mode="signature")` | **Not pure alias.** Multi-input dispatcher classmethod (callable → Tool, Agent → `as_tool`, Tool → passthrough/clone) — same idiom as `dict.fromkeys` / `Path.cwd`. Both default to `mode="signature"` since 0.7.9 (the `"auto"` graceful-fallback ladder was removed — opt into LLM enrichment by passing `mode="hybrid"` or `mode="llm"` plus `schema_llm=`). |
| `tool(search_web, …)` (lowercase) | `Tool.wrap(search_web, …)` | Backwards-compat alias for `Tool.wrap`; kept indefinitely so existing imports work. New code should prefer the classmethod. |
| `Tool.from_schema(name, description, parameters, func, strict=…, returns_envelope=…)` | (no callable-introspection canonical) | **Not sugar over `Tool(callable, …)`** — this is the canonical form when the JSON Schema is already known (MCP, OpenAPI bridges, third-party registries). Bypasses the schema builder. |

**Wrap an Agent as a Tool**

| Sugar | Canonical | Differences |
|---|---|---|
| `tools=[other_agent]` (in another agent) | (this is itself the canonical) | The agent's `name=` becomes the surface tool name. |
| `researcher.as_tool("deep_research")` | A `Tool` whose `func` calls `researcher.run` | **Not pure alias.** Use to **rename** (different surface name than `researcher.name`) or to attach a `verify=` / `max_verify=` judge-and-retry loop — a feature `tools=[researcher]` does **not** expose. |
| `Tool.wrap(researcher, name="deep_research")` | Identical to `researcher.as_tool("deep_research")` | Pure alias of `as_tool` for agent-like inputs (also dispatches callables and Tools through the same factory). |

**Call an Agent**

| Form | When |
|---|---|
| `result = agent(task)` (sync) | **Canonical entry point.** `__call__` auto-detects an event loop. |
| `result = await agent.run(task)` | Inside an existing `async def` caller. |
| `async for chunk in agent.stream(task):` | Incremental tokens / events. |

Default model in examples: `claude-opus-5`. When a user is learning,
err on the side of the longer canonical form — even if a one-liner
works, the canonical version teaches the shape they will need at every
later rung.

**Default-model fallback** — when `claude-opus-5` is sunset (or any
date-pinned model id stops resolving), reach for the **tier alias**
path instead of guessing the next model id:

```python
# Always-current "best in tier" — tracks the provider's lineup
agent = Agent.from_provider("anthropic", tier="top")
```

Tier strings: `super_cheap` / `cheap` / `medium` / `expensive` / `top`.
This is the **only** non-pure-alias `from_*` factory left after 0.7.9;
the deleted ones (`from_model` / `from_engine` / `from_chain` /
`from_plan` / `from_parallel`) were just renames of the canonical
`Agent(engine=...)` ctor and are gone for good.

Full reference with worked examples for each row:
<https://core.lazybridge.com/concepts/canonical-vs-sugar/>.

## Canonical patterns

### Single agent

```python
from lazybridge import Agent, LLMEngine

agent = Agent(
    engine=LLMEngine("claude-opus-5"),
)
result = agent("hello")
print(result.text())
```

`LLMEngine("claude-opus-5")` is what makes this an LLM-driven agent.
Configure the engine in place — `LLMEngine("claude-opus-5", system=
"...", max_turns=10, thinking=True, ...)` — instead of reaching for
factory variants. Default model is `claude-opus-5`.

### Agent with a tool

```python
from lazybridge import Agent, LLMEngine


def get_weather(city: str) -> str:
    """Return the current weather for ``city``."""
    ...


agent = Agent(
    engine=LLMEngine("claude-opus-5"),
    tools=[get_weather],
)
result = agent("Weather in Paris?")
print(result.text())
```

Do **not** write a JSON schema by hand. LazyBridge infers it from the
signature, type hints, and docstring (`mode="signature"`, the default).
For legacy callables you can't annotate, switch the mode to `"llm"` or
`"hybrid"` via `Tool(callable, mode="llm")`.

### Structured output

```python
from pydantic import BaseModel
from lazybridge import Agent, LLMEngine


class Summary(BaseModel):
    headline: str
    bullets: list[str]


agent = Agent(
    engine=LLMEngine("claude-opus-5"),
    output=Summary,
)
result = agent("Summarise the news")
print(result.payload.headline)  # read .payload, not .text()
```

### Sequential / parallel composition

The canonical sequential form is a `Plan` of named steps — same shape
you'll use for routing, parallel bands, and checkpoints later, so the
mental model stays uniform as the workflow grows:

```python
from lazybridge import Agent, Plan, Step

pipeline = Agent(
    engine=Plan(Step("research"), Step("write")),
    tools=[researcher, writer],
)
```

For a *purely* linear handoff with no other plan features,
`Agent.chain(researcher, writer)` is sugar for exactly the form above —
reach for it when you want a one-liner.

For **scripted** fan-out, use `Agent.parallel(a, b, c)`. The runner's
`__call__` returns ONE `Envelope` whose `.text()` is a labelled-text
join across every branch — same shape as `Plan`'s `from_parallel_all`
aggregator, with transitive cost rollup in `metadata.nested_*` and
first-error short-circuit in `.error`. For typed per-branch access
call `parallel.run_branches(task)` (async) → `list[Envelope]`. Use a
`Plan` parallel band (`Step("a", parallel=True)`) when concurrent steps
need to aggregate via `from_parallel_all`, and put the candidates in
`tools=[...]` when you want the **LLM** to decide which sub-agent to
dispatch instead of running all of them.

### Agent as tool (supervisor / hierarchical)

```python
from lazybridge import Agent, LLMEngine

researcher = Agent(
    engine=LLMEngine("claude-opus-5"),
    name="research",
    tools=[search],
)
supervisor = Agent(
    engine=LLMEngine("claude-opus-5"),
    tools=[researcher],
)
```

The researcher's `name=` becomes the tool name the supervisor sees.
Use `researcher.as_tool("alias")` only when you need a surface name
different from the agent's own `name=`. Prefer this over building
bespoke multi-agent orchestration glue.

### Dynamic graph (AgentPool + conclude)

For multi-agent graphs where agents delegate to **each other** by name and
the topology is decided by the LLM at runtime (not a fixed DAG), use
`AgentPool` together with `conclude`:

```python
from lazybridge import Agent, AgentPool, LLMEngine, conclude

pool = AgentPool()  # registry exposed as a single `route(agent_name, task)` tool
alice = Agent(engine=LLMEngine("claude-opus-5"), name="alice", tools=[pool.as_tool(), conclude])
bob = Agent(engine=LLMEngine("claude-opus-5"), name="bob", tools=[pool.as_tool(), conclude])
pool.register(alice, bob)  # register AFTER construction (breaks the cycle)
result = alice.run("...")  # alice may route("bob", …); any agent may conclude(…)
```

- `pool.as_tool()` is an ordinary `Tool` named `route` — the engine does not
  special-case it. Pass `pool.as_tool("ask_team")` to rename it so one agent
  can hold several pools (e.g. `tools=[team.as_tool("ask_team"),
  peers.as_tool("ask_peer")]`) — handy for layered/nested routing.
  `AgentPool(max_depth=…)` bounds recursion so cycles can't blow the stack.
- `conclude("answer")` is a non-local exit: raised anywhere in the nested tree,
  it unwinds the **whole** chain and returns straight to the top-level
  `run()` — internally a `ConcludeSignal` (a `BaseException`) caught only there.
- Pair with `LLMEngine(max_tool_calls_per_turn=1)` to keep the graph on a
  single non-branching path (distinct from `max_parallel_tools`, which only
  bounds concurrency, not the number of calls executed).

### Deterministic plan

```python
from lazybridge import Agent, Plan, Step

pipeline = Agent(
    engine=Plan(
        Step("research"),
        Step("write"),
    ),
    tools=[researcher, writer],
)
print(pipeline("Topic: AI agents 2026").text())
```

`Step("name")` references a sub-agent by its `name=`. Plans are validated
at construction — forward references, duplicate names, and unknown
targets raise `PlanCompileError` before any LLM call. `Agent(engine=Plan(*steps))`
is sugar for the explicit form above.

### Sentinels — wiring data between steps

| Sentinel | Resolves to |
|---|---|
| `from_prev` | The previous step's payload (default if you write nothing) |
| `from_start` | The plan's initial input |
| `from_step("name")` | The named step's output |
| `from_parallel("name")` | The named branch's output (single branch) |
| `from_parallel_all("name")` | All consecutive parallel siblings, aggregated as labelled text |
| `from_memory("name")` | The agent's live conversation history |
| `from_agent("name")` | The agent's last stored output from the cross-run `Store` |

```python
Step("write", task=from_prev, context=from_step("research"))
```

### Routing

```python
from lazybridge import when

Step(
    "triage",
    routes={
        "legal": when.field("category").equals("legal"),
        "technical": when.field("category").equals("technical"),
    },
)
```

Or let an LLM decide via a structured field on the step's `output=`:
`Step("triage", routes_by="category")`.

### Checkpoint + resume

```python
from lazybridge import Agent, Plan, Step, Store

store = Store(db="runs.db")

pipeline = Agent(
    engine=Plan(*steps, store=store, checkpoint_key="ticket-42"),
    tools=[...],
)
# crash...
resumed = Agent(
    engine=Plan(*steps, store=store, checkpoint_key="ticket-42", resume=True),
    tools=[...],
)
```

Concurrent forks on the same key are protected by compare-and-swap; pass
`on_concurrent="fail"` (default) or `"queue"`.

### Dynamic replan loop — ReplanEngine

Use `ReplanEngine` when the pipeline shape is decided at runtime by a planner
agent.  `Plan` is for fixed DAGs; `ReplanEngine` is for adaptive multi-round
work where each round's tasks depend on the previous round's results.

```python
from lazybridge import Agent, LLMEngine, ReplanEngine, Store
from lazybridge.engines.replan import PlanRound, ReplanTask

planner = Agent(
    engine=LLMEngine("claude-opus-5", system="You are a task planner."),
    output=PlanRound,
    name="planner",
)

guardian = Agent(
    engine=ReplanEngine(
        store=Store(db="project.sqlite"),
        checkpoint_key="my-project",
        resume=True,
    ),
    tools=[planner, analyst, coder],
    name="guardian",
)

guardian("refactor the auth module")  # first session
guardian("continue")  # resumes from last checkpoint
```

`ReplanTask` uses `tool + kwargs` so dispatch is `tool_map[task.tool].run(**task.kwargs)` —
no special-casing for pools, agents, or functions.  The planner receives the
available tool schemas and accumulated history dynamically so its system prompt
needs no hardcoded worker names.  Same checkpoint/resume semantics as `Plan`.

### Claude Code as the model loop — ClaudeCodeEngine

`ClaudeCodeEngine` is a standard `Engine` that runs the model/tool loop
through the locally authenticated Claude Code runtime (Claude Agent SDK)
instead of a raw provider API call — same `Agent`/`Memory`/`Session`/`tools=`
surface as `LLMEngine`, no separate integration code. Requires the optional
`lazybridge[claude-code]` extra.

```python
from lazybridge import Agent, ClaudeCodeEngine

agent = Agent(
    name="research",
    engine=ClaudeCodeEngine(model="sonnet", cwd="C:/work/project"),
    tools=[search, specialist_agent],
)
```

`cwd`/`file_roots` opt into read-only `Read`/`Glob`/`Grep` scoped to those
directories; `web=True` (default) enables `WebSearch`/`WebFetch`. `Bash`,
`Write`, `Edit`, and native Claude Code subagents are never exposed. Full
setup (Claude Code sign-in, install, config knobs, session modes,
troubleshooting): [Claude Code Engine guide](../../docs/guides/full/claude-code-engine.md).

A `CodexEngine` provides the same `Engine` contract through the locally
authenticated Codex App Server. Configure either engine explicitly with
`CodingAgentConfig.reviewer()` for a fail-closed read-only reviewer or
`CodingAgentConfig.writer(approval_gate)` for a workspace writer whose native
and LazyBridge tool actions pass through one `ApprovalRequest` /
`ApprovalDecision` callback. Use `ClaudeCodePolicy` and `CodexPolicy` inside a
custom `CodingAgentConfig` when the native permission mode, sandbox, allowlist,
denylist, setting sources, or dynamic-tool policy needs to be set individually.
Implement a custom `ApprovalGate` callback for production or use
`TerminalApprovalGate` during local development.
See [Coding-agent permissions](../../docs/guides/full/coding-agent-config.md).

### Human-in-the-loop

```python
from lazybridge import Agent
from lazybridge.ext.hil import HumanEngine, SupervisorEngine

approval = Agent(engine=HumanEngine(timeout=300), name="approve")
repl = Agent(engine=SupervisorEngine(tools=[...]), name="repl")
```

`human_agent(timeout=300, name="approve")` and
`supervisor_agent(tools=[...], name="repl")` are sugar for the two lines
above. Both forms return regular `Agent` objects — drop them into a
`Plan`'s `tools=[...]` and reference them from a `Step` like any other
agent.

### MCP

```python
from lazybridge import Agent, LLMEngine
from lazytools.connectors.mcp import MCP  # pip install lazytoolkit (import name: lazytools)

# command / args / allow are keyword-only on MCP.stdio.
# allow= (or deny=) is REQUIRED since 0.7.9 — deny-by-default for
# both stdio and http; omitting them raises ValueError.
fs = MCP.stdio(
    "fs",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    allow=["fs.read_*", "fs.list_*"],
)
http = MCP.http(
    "docs",
    "https://example.com/mcp",
    allow=["docs.search_*"],
)

agent = Agent(
    engine=LLMEngine("claude-opus-5"),
    tools=[fs, http],
)
```

`MCPServer` is a `ToolProvider` — pass it directly into `tools=[...]`.
Tool names are namespaced as `"<server>.<tool>"`.

### Sessions and exporters

```python
from lazybridge import Agent, JsonFileExporter, LLMEngine, Session

session = Session(exporters=[JsonFileExporter("events.jsonl")])
agent = Agent(
    engine=LLMEngine("claude-opus-5"),
    session=session,
)
```

For OpenTelemetry, install `lazybridge[otel]` and add `OTelExporter(...)`
to the same list. **Inheritance depends on how the sub-agent is nested,
verified:** an agent passed via `tools=[...]` (and `fallback=`/`verify=`)
inherits the parent's session automatically. An agent used **directly as a
`Plan` `Step` target does not** — pass `session=` explicitly when building
it, or its events (and those of anything it calls) never reach the log,
making a multi-step pipeline look silently incomplete in its own trace.

## Anti-patterns to avoid

- **Wrapping simple examples in `asyncio.run(main())`**. The canonical
  call shape is `agent(task)`. Reach for `await agent.run(task)` only
  inside an existing async caller.
- **Defining a JSON tool schema by hand** when a Python function
  exists. The signature path is the default and covers >95% of real
  callables.
- **Hiding the engine behind sugar.** `Agent(...)`, the
  string-positional `Agent("claude-opus-5")`, and `Agent(engine=...)`
  all save a line of code at the cost of hiding which engine the agent
  actually runs. Lead with `Agent(engine=LLMEngine("..."), ...)`,
  especially in tutorials and code reviews.
- **Wrapping every helper in its own sub-agent.** Sub-agents are not
  free — use them when the responsibility is genuinely distinct.
- **Reaching for a `Plan` when one `Agent` with a few tools would do.**
  Pick the lowest rung on the
  [progressive complexity ladder](https://core.lazybridge.com/concepts/progressive-complexity/)
  that solves the problem.
- **Passing the same agent twice via `agent.as_tool(...)` for both
  positional and tool use** when the agent's own `name=` is already
  unique. `tools=[other_agent]` works; `as_tool("alias")` is only for
  renaming.
- **Holding state in free-form text passed between agents.** Use a typed
  `output=PydanticModel` or write to a `Store`.
- **Importing private names** (`_`-prefixed) or anything from
  `lazybridge.core.*` directly. The public surface is `lazybridge.*` and
  `lazybridge.ext.*` only.
- **Reaching for a deleted-in-0.7.9 factory.** `Agent.from_model` /
  `Agent.from_engine` / `Agent.from_chain` / `Agent.from_plan` /
  `Agent.from_parallel` were pure-alias renames of the canonical
  `Agent(engine=...)` ctor and are gone in 0.7.9. The `Agent.from_*`
  shape that survives is **only** `Agent.from_provider(provider, tier=...)`,
  which is non-trivial (resolves a tier alias to the provider's current
  model). Use the canonical ctor for everything else.
- **Iterating `Agent.parallel(...)("task")` as a list.** Since 0.7.9 the
  call returns ONE `Envelope` (joined branches in `.text()`); for the
  typed list, call `parallel.run_branches(task)` (async).
- **Passing `runtime=` / `resilience=` / `observability=` / a config
  object to `Agent`.** The three wrapper-of-flat-kwargs configs were
  deleted in 0.7.9 along with the `_UNSET` precedence game. Pass flat
  kwargs (`timeout=...`, `max_retries=...`, `session=...`, `name=...`)
  directly, or share a fleet default via `**PROD_DEFAULTS`.
- **Wrapping an agent call in `try` / `except` to reach a backup**, when
  the failure arrives as a *returned error envelope* — which includes a
  `Plan` step that raises. Use `fallback=` instead. It does not reliably
  fire on a **tool** raising inside an `LLMEngine`'s tool loop (that
  becomes a tool result the model sees and may work around, not an error
  envelope) — if a tool failure must abort the run, return a typed failure
  and check it, don't rely on the exception to escalate. It also does not
  fire when the agent's **own engine raises directly** rather than
  returning an error envelope (a custom engine, most commonly) — keep
  `try`/`except` around that call site; do not remove it in favor of
  `fallback=` on the strength of this rule alone.
- **A gate with a predicate for only one branch.** `routes={"execute": ...}`
  with nothing for the rejection path executes anyway when the predicate is
  false — routing falls through linearly. Every branch needs its own
  predicate, always, especially on an approval gate.
- **An agent used directly as a `Step` target does not inherit the
  parent's `Session`** — only agents passed via `tools=[...]` do. Pass
  `session=` explicitly when building a sub-agent for a `Step`, or its
  events silently vanish from the run's log.

## Where to read more

- Full mental model: <https://core.lazybridge.com/concepts/mental-model/>
- Composition rule: <https://core.lazybridge.com/concepts/everything-is-a-tool/>
- The 12-rung complexity ladder: <https://core.lazybridge.com/concepts/progressive-complexity/>
- Per-concept guides: <https://core.lazybridge.com/guides/>
- Runnable recipes: <https://core.lazybridge.com/recipes/>
- API reference: <https://core.lazybridge.com/reference/>
- Errors → fixes: <https://core.lazybridge.com/errors/>
