# Canonical vs sugar — full reference

LazyBridge ships several factory functions and classmethod shortcuts
that exist for ergonomic reasons. Each is sugar over a more explicit
**canonical** form built from the framework's primitives (`Agent`,
`LLMEngine`, `Plan`, `Step`, `HumanEngine`, `Tool`, …).

Knowing the canonical form behind every sugar is useful because:

- It teaches the framework's actual mental model (Engine + Tools + State).
- **Not every sugar is a pure alias** — some build extra structure or
  return different types. This page calls those out so you know what
  the sugar buys you and what (if anything) it costs.
- Tutorials and code reviews should lead with the canonical form so
  the engine choice is visible at the call site.

The shape used in every "Canonical" block below is the same one
`examples/` uses: each constructor argument on its own line and
`result = agent(task)` on a separate line from the `print`.

---

## 1. Build an Agent with an LLM engine

```python
# Canonical
from lazybridge import Agent, LLMEngine

agent = Agent(
    engine=LLMEngine("claude-haiku-4-5"),
    tools=[search],
    name="research",
)
```

| Sugar | Expands to | Differences |
|---|---|---|
| `Agent("claude-haiku-4-5", tools=[search], name="research")` | The canonical form above | **Pure alias.** The first positional argument is interpreted as a model string and threaded through to `LLMEngine(...)` internally. Hides which engine drives the agent at the call site. |
| `Agent.from_provider("anthropic", tier="top", tools=[search], name="research")` | `Agent(engine=LLMEngine("top", provider="anthropic"), tools=[search], name="research")` | **Not pure sugar.** Builds an `LLMEngine` whose model string is a **tier alias** (`super_cheap` / `cheap` / `medium` / `expensive` / `top`); each provider class maps the alias to its current lineup. Use when you want "freshest model in tier X" without pinning a date-stamped name. |

---

## 2. Build an Agent with a Plan engine

```python
# Canonical
from lazybridge import Agent, Plan, Step, Store

pipeline = Agent(
    engine=Plan(
        Step("research"),
        Step("write"),
        store=Store(db="run.sqlite"),
        checkpoint_key="research",
        resume=True,
    ),
    tools=[researcher, writer],
)
```

No sugar — write the canonical form. Plan's kwargs (`max_iterations`,
`store`, `checkpoint_key`, `resume`, `on_concurrent`) live on `Plan(...)`;
Agent's kwargs (`tools=`, `session=`, `name=`, …) live on `Agent(...)`.

---

## 3. Compose agents — sequential

```python
# Canonical Pattern A — Step.target is the agent itself, no tools= needed
from lazybridge import Agent, Plan, Step

pipeline = Agent(
    engine=Plan(
        Step(target=researcher, name=researcher.name),
        Step(target=writer,     name=writer.name),
    ),
    name="chain",
)

# Canonical Pattern B — Step references by name, agents in tools=
pipeline = Agent(
    engine=Plan(Step("research"), Step("write")),
    tools=[researcher, writer],
)
```

Both Patterns are canonical. Pattern A is what `Agent.chain` produces
internally — no `tools=` needed because `Plan` dispatches `Agent`
targets via `target.run()` directly. Pattern B is more readable when
many agents share a single tool-map at the top level.

| Sugar | Expands to | Differences |
|---|---|---|
| `Agent.chain(researcher, writer)` | Pattern A above, with `name="chain"` | **Not a pure alias** — it constructs the `Plan` + `Step` graph for you, but the result is structurally identical to canonical Pattern A. Targets are agents (not name strings); no `tools=` needed. |

---

## 4. Compose agents — parallel fan-out

```python
# Canonical (no Agent-shaped equivalent — this IS the canonical form)
from lazybridge import Agent

multi = Agent.parallel(researcher_a, researcher_b, researcher_c)
env = multi("Same task for everyone")   # -> Envelope (labelled-text join in .text())
# For typed per-branch list[Envelope]: branches = await multi.run_branches(task)
```

| Sugar | Expands to | Differences |
|---|---|---|
| `Agent.parallel(*agents, concurrency_limit=None, step_timeout=None)` | (no `Agent`-shaped equivalent) | **Not sugar over `Agent`.** Returns `ParallelAgent`, a sibling class whose `__call__` produces ONE `Envelope` whose `payload` is the labelled-text join of every branch (`[name]\n<output>`) — same shape as `Plan`'s `from_parallel_all` aggregator, with transitive cost rollup and first-error short-circuit.  For typed per-branch access (`list[Envelope]`) call `parallel.run_branches(task)` (async).  Use this when you want every branch unconditionally; to let the **LLM** decide which branches to invoke, use `Agent(tools=[a, b, c])` instead; to run concurrent steps that **aggregate** via `from_parallel_all`, use a `Plan` parallel band (`Step("a", parallel=True)`). |

---

## 5. Build an Agent with a HIL engine

```python
# Canonical
from lazybridge import Agent
from lazybridge.ext.hil import HumanEngine, SupervisorEngine

approval = Agent(
    engine=HumanEngine(timeout=300, ui="terminal", default="approve"),
    name="approve",
)
repl = Agent(
    engine=SupervisorEngine(tools=[search], agents=[researcher]),
    name="ops-supervisor",
    session=sess,
)
```

| Sugar | Expands to | Differences |
|---|---|---|
| `human_agent(timeout=300, ui="terminal", default="approve", name="approve")` | `Agent(engine=HumanEngine(timeout=300, ui="terminal", default="approve"), name="approve")` | **Pure alias** with a kwarg split: HIL-engine kwargs go to `HumanEngine(...)`, remaining `**agent_kwargs` flow to `Agent(...)`. Lives in `lazybridge.ext.hil` (not on `Agent`) so the core package doesn't have to import the ext-side engine. |
| `supervisor_agent(tools=[search], agents=[researcher], session=sess, name="ops-supervisor")` | `Agent(engine=SupervisorEngine(tools=[search], agents=[researcher]), session=sess, name="ops-supervisor")` | Same kwarg-split pattern as `human_agent`; same import-boundary rationale. |

---

## 6. Wrap a callable as a Tool

```python
# Canonical — explicit ``tool()`` factory pins the LLM-visible name
# even if the function is renamed, keeping tool-maps and plan
# references stable across refactors.  This is the form the framework
# docstring (lazybridge/__init__.py) flags as canonical.
from lazybridge import tool

agent = Agent(
    engine=LLMEngine("claude-haiku-4-5"),
    tools=[tool(search_web, name="search_web")],
)

# Sugar — bare callable.  Backward-compatible; auto-wrapped with
# ``Tool(search_web, name=search_web.__name__)``.  Convenient for
# one-shot scripts; prefer the explicit form in production.
agent = Agent(
    engine=LLMEngine("claude-haiku-4-5"),
    tools=[search_web],
)

# Advanced — direct ``Tool`` constructor when you need ``mode=`` /
# ``strict=`` / ``schema_llm=`` / a custom name.
from lazybridge import Tool

search = Tool(
    search_web,
    name="search",
    description="Search the web for the query.",
    mode="signature",
)
agent = Agent(engine=LLMEngine("claude-haiku-4-5"), tools=[search])
```

| Sugar | Expands to | Differences |
|---|---|---|
| `tool(search_web, name="search", description="Search the web.")` | `Tool(search_web, name="search", description="Search the web.")` | **Not pure alias.** Multi-input dispatcher: callables → `Tool(...)`, Agents → `agent.as_tool(...)`, existing Tools → passthrough or clone-with-overrides. Both default to `mode="signature"` since 0.7.9 (the `"auto"` graceful-fallback ladder was removed — opt into LLM enrichment by passing `mode="hybrid"` or `mode="llm"` plus `schema_llm=`). |
| `Tool.from_schema(name, description, parameters, func, strict=False, returns_envelope=False)` | (no callable-introspection canonical — this IS the canonical for pre-built schemas) | **Not sugar over `Tool(callable, …)`.** Used when the JSON Schema is already known (MCP servers, OpenAPI bridges, third-party tool registries). Bypasses the schema builder and sets `_definition` directly. |

---

## 7. Wrap an Agent as a Tool

```python
# Canonical — the agent's own name= becomes the tool name
researcher = Agent(
    engine=LLMEngine("claude-haiku-4-5"),
    name="research",
    tools=[search],
)
orchestrator = Agent(
    engine=LLMEngine("claude-haiku-4-5"),
    tools=[researcher],          # <-- pass the agent directly
)
```

| Sugar | Expands to | Differences |
|---|---|---|
| `researcher.as_tool("deep_research")` | A `Tool` whose `func` calls `researcher.run` and whose `name` is the alias | **Not pure alias.** Use when you want a **different surface name** than the agent's own `name=`, or when wrapping a duck-typed agent that doesn't have an explicit `name=`. Also takes `verify=` / `max_verify=` to wrap the call in a judge/retry loop — a feature `tools=[researcher]` does **not** expose. |
| `tool(researcher, name="deep_research")` | Equivalent to `researcher.as_tool("deep_research")` | **Pure alias** of `as_tool` for agent-like inputs. Useful when you're building a tool list programmatically (single dispatcher for callables, agents, and Tools). |

---

## 8. Call an Agent

```python
# Canonical (sync) — what every runnable example in examples/ uses
result = agent(task)
print(result.text())
```

| Form | When |
|---|---|
| `agent(task)` (sync) | **Canonical entry point.** `__call__` detects whether an event loop is already running and either runs `asyncio.run` or schedules a coroutine with caller contextvars. |
| `await agent.run(task)` | When you're already inside an `async def` caller. Same semantics as `agent(task)`, no event-loop detection. |
| `async for chunk in agent.stream(task)` | When you want incremental tokens / events instead of one final envelope. |

---

## Summary — when sugar is worth it

| Situation | Reach for sugar |
|---|---|
| Tutorials, code reviews, the example you ship in the README | **No.** Canonical form makes the engine choice visible. |
| Internal one-liners when the engine choice is uninteresting (`Agent(engine=Plan(...))` for a 3-step pipeline, `human_agent(timeout=60)` for a one-shot gate) | **Yes.** |
| Production code with structured config (the agent is built once, configured via `runtime=` / `resilience=` / `observability=`) | **No.** Canonical form composes more cleanly with config objects. |
| When you're using a tier alias (`Agent.from_provider("anthropic", tier="top")`) | **Yes.** This is the canonical way to pin a tier without a date-stamped model name. |
| Scripted fan-out (`Agent.parallel(...)`) | **Yes.** This *is* the canonical form — there is no `Agent`-shaped equivalent. |

## See also

- [Mental model](mental-model.md) — Agent = Engine + Tools + State, the
  decomposition every form on this page slots into.
- [Everything is a tool](everything-is-a-tool.md) — why so many forms
  collapse into `tools=[...]`.
- [Progressive complexity](progressive-complexity.md) — every rung
  uses the canonical form first, with sugar called out where it
  shortens the example without hiding the engine.
