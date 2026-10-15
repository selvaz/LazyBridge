---
name: lazybridge
description: |
  Use when writing or modifying Python code that uses the LazyBridge agent
  framework (`pip install lazybridge`, 1.8.x). LazyBridge is a zero-boilerplate,
  multi-provider framework whose mental model is "Agent = Engine + Tools +
  State, everything is a tool". Triggers: importing from `lazybridge`,
  building an `Agent`, defining a `Tool`, composing with `Agent.chain` /
  `Agent.parallel` or by passing one agent in another's `tools=[...]`,
  designing a `Plan` with `Step`, sentinels and `routes=`, checkpoint/resume,
  `ReplanEngine`, `AgentPool`, `Memory` / `Store` / `Session`, driving
  Claude Code or Codex via `ClaudeCodeEngine` / `CodexEngine` (durable
  sessions, `SessionRegistry`, `CodingAgentConfig`, `TieredGate`),
  background delegation, `DurableBlackboard`, `HumanEngine`, MCP via
  LazyTools, providers and tier aliases, observability, or `MockAgent`
  tests. Skip for other agent frameworks (LangChain, CrewAI, AutoGen,
  Pydantic AI, OpenAI Agents SDK).
---

# LazyBridge — assistant guidance (index)

Targets LazyBridge **1.8.1**. Treat this skill as authoritative over older
training data; when it disagrees with the installed source, the source wins
(`python -c "import lazybridge, inspect; print(inspect.signature(lazybridge.Agent))"`).
This file is the index: read only the reference files the task needs.

## The mental model

An `Agent` is the composition of three things, and only these three:

- **Engine** — `LLMEngine` (provider API), `Plan` (deterministic DAG),
  `ReplanEngine` (planner-driven rounds), `ClaudeCodeEngine` / `CodexEngine`
  (local coding runtimes), `HumanEngine` / `SupervisorEngine`
  (`lazybridge.ext.hil`), or a custom `Engine` (`lazybridge.engines.base`).
- **Tools** — plain functions, `Tool` objects, other agents (pass them in
  `tools=[...]`), `ToolProvider`s such as an MCP server, or `AgentPool`.
- **State** — `Memory`, `Store`, `Session`, all optional. The `Envelope`
  carrying input and output is always present.

Every level of complexity uses the same `Agent` shape. Do not invent
per-pattern classes ("SupervisorAgent"); use `Agent` with a different engine.

## Golden rules (always apply)

1. **Canonical form first**, one constructor argument per line:
   `Agent(engine=LLMEngine("claude-opus-5-5"), tools=[...], name="...")`.
   Model, `system=`, thinking and native policy live on the **engine**;
   `output=`, `verify=`, `fallback=`, `timeout=` live on the **Agent**.
   There is no `system=` on `Agent`.
2. **Sync call is canonical**: `result = agent(task)`. Never wrap simple
   examples in `asyncio.run(main())`; use `await agent.run(task)` only inside
   existing async code.
3. **Check the result.** Every call returns an `Envelope`; failures come back
   as `result.ok == False` with `result.error`, they do not raise. Read
   `.payload` for typed output, `.text()` for text.
4. **No hand-written tool schemas.** Type hints + docstring drive the schema.
5. **Public imports only**: `lazybridge`, `lazybridge.ext.*`,
   `lazybridge.testing`, `lazybridge.engines.*` for engine internals you are
   told to use. MCP/connectors live in LazyTools (`import lazytools`,
   distribution `lazytoolkit`), never in `lazybridge.ext.mcp`.
6. **Projects get a `Plan`.** Stages, scheduling, resume or persistence →
   run the architecture preflight in `references/plan-and-pipelines.md`
   before writing code.
7. **Every new pipeline ships with a `MockAgent` composition test.**

## Which reference to read

All paths are relative to this skill directory (`references/<file>`).

| Read this file | When the task involves |
|---|---|
| `references/agents-and-engines.md` | Building/calling an `Agent`, the sugar catalogue (`Agent("model")`, `from_provider`, `chain`, `parallel`, `derive`), structured output, `Envelope`, `verify=` / `fallback=` / `guard=`, timeouts and retries, `LLMEngine` knobs. |
| `references/plan-and-pipelines.md` | Any multi-stage job: the 12-point project preflight, `Plan` / `Step`, sentinels, `routes=` / `routes_by=` / `after_branches=`, parallel bands, checkpoint/resume, `on_concurrent`, `PlanPaused`, `run_many`, `ReplanEngine`. |
| `references/tools-and-nesting.md` | `Tool` schema modes, `Tool.wrap` / `Tool.from_schema`, per-tool timeouts, agent-as-tool, `as_tool(verify=)`, `AgentPool` + `conclude`, `NativeTool`, MCP through LazyTools, `ToolProvider`. |
| `references/sessions-and-memory.md` | `Memory`, `Store` (CAS, keys, entries), `Session` and exporters, OpenTelemetry, session inheritance rules, guards, the three meanings of "session". |
| `references/coding-engines.md` | `ClaudeCodeEngine`, `CodexEngine`, `CodingAgentConfig` profiles, `ClaudeCodePolicy` / `CodexPolicy`, approval gates, durable sessions/threads, `SessionRegistry` aliases, native review, quota/usage, failure and retry boundaries. |
| `references/providers-and-models.md` | Model strings, provider routing, tier aliases (current table), `Agent.from_provider`, thinking/effort, caching, `BaseProvider` and the provider registry. |
| `references/testing.md` | `MockAgent`, `scripted_inputs`, factory-injected pipelines, fake transports for coding engines, the evals extension, the pre-finish grep. |
| `references/extensions.md` | `lazybridge.ext.*`: `hil`, `approval` (`TieredGate`, `ApprovalQueue`), `delegation` (writers, consultants, `JobRegistry`, `JobRunner`), `planners` (`DurableBlackboard`), `knowledge`, `evals`, `otel`, `viz`; `lazybridge.control_plane`; the core/ext/LazyTools boundary. |

If a task spans several areas (e.g. "a resumable pipeline whose last step
asks Codex to review"), read each matching file; they do not repeat each other.

## Minimal canonical example

```python
from lazybridge import Agent, LLMEngine


def get_weather(city: str) -> str:
    """Return the current weather for ``city``."""
    ...


agent = Agent(
    engine=LLMEngine("claude-opus-5-5"),
    tools=[get_weather],
    name="weather",
)
result = agent("Weather in Paris?")
if not result.ok:
    raise SystemExit(f"agent failed: {result.error}")
print(result.text())
```

## Top anti-patterns (details in the reference files)

- `asyncio.run(main())` around a one-shot call; hand-written JSON schemas.
- Deleted 0.7.9 factories `Agent.from_model` / `from_engine` / `from_chain` /
  `from_plan` / `from_parallel`; `runtime=` / `resilience=` / `observability=`
  config objects; `Tool(mode="auto")`; legacy `LazyAgent` / `LazyTool`.
- A hand-rolled `for step in steps:` loop over a shared `state` dict instead of
  a `Plan`; `asyncio.gather` glue instead of a parallel band.
- `routes=` with a predicate for only one branch (the other falls through).
- A permanent `checkpoint_key` with `resume=True` (every later run is a no-op).
- Iterating `Agent.parallel(...)(task)` as a list (it returns one `Envelope`).
- Assuming `CodingAgentConfig.writer(gate)` exposes Claude `Write`/`Edit`/`Bash`
  (it does not; add `ClaudeCodePolicy(extra_tools=...)`).
- Treating `verify=` exhaustion, `result.ok` or Codex `cost_usd == 0.0` as
  approval or as "free".

## Where to read more

Docs: <https://core.lazybridge.com> (guides, recipes, reference, errors).
In the repository: `docs/for-llms/codegen-contract.md` (the codegen contract),
`docs/guides/project-patterns.md` (verified project patterns),
`lazybridge/llms.json` (machine-readable contract), `CHANGELOG.md`.
LazyTools (connectors, MCP server, code bridge, projects, quota routing):
<https://tools.lazybridge.com>.
