# LazyBridge

**Zero-boilerplate Python agent framework.** One `Agent`, swappable
engines (LLM, Human, Supervisor, Plan), and one tool contract —
functions, agents, and agents-of-agents all compose the same way.
Parallelism is automatic when the engine decides; declared when you do.

Pipelines are validated at construction time: a misspelled step name
or an unknown reference surfaces as `PlanCompileError` **before any
LLM call**, not at the first production failure.

```python
from lazybridge import Agent
print(Agent("claude-opus-4-7")("hello").text())
```

## Pick your tier

LazyBridge grows with you — every tier is additive.

| Tier | For | Key imports |
|---|---|---|
| [**Basic**](docs/tiers/basic.md) | one-shot or tool-calling agents | `Agent` · `Tool` · `NativeTool` · `Envelope` |
| [**Mid**](docs/tiers/mid.md) | real apps with memory, tracing, guardrails, composition | `Memory` · `Store` · `Session` · `Guard*` · `chain` · `parallel` · `as_tool` · `HumanEngine` · `EvalSuite` |
| [**Full**](docs/tiers/full.md) | production pipelines: typed hand-offs, routing, resume, OTel | `Plan` · `Step` · `from_prev`/`from_step`/`from_parallel` · `SupervisorEngine` · checkpoint · exporters · `verify=` |
| [**Advanced**](docs/tiers/advanced.md) | extending the framework | `Engine` · `BaseProvider` · `Plan.to_dict` · `register_provider_*` · `core.types` |

## Install

```bash
pip install lazybridge[anthropic]   # or [openai], [google], [deepseek], [all]
```

Set an API key for your provider of choice (`ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `DEEPSEEK_API_KEY`).

## Worked examples

### 1 · Function becomes a tool, auto-schema

```python
from lazybridge import Agent

def get_weather(city: str) -> str:
    """Return current temperature and conditions for ``city``."""
    return f"{city}: 22°C, sunny"

print(Agent("claude-opus-4-7", tools=[get_weather])(
    "what's the weather in Rome and Paris?"
).text())
```

No decorators, no JSON schemas. If your function lacks type hints, pass
`mode="llm"` to have a cheap agent infer the schema — see
[Function → Tool](docs/guides/tool-schema.md).

### 2 · Native tools (no code at all)

```python
from lazybridge import Agent, NativeTool

Agent("claude-opus-4-7", native_tools=[NativeTool.WEB_SEARCH])("AI news this week")
```

`WEB_SEARCH` · `CODE_EXECUTION` · `FILE_SEARCH` · `COMPUTER_USE` ·
`GOOGLE_SEARCH` · `GOOGLE_MAPS` (each supported by a subset of
providers).

### 3 · Tool is tool — agents wrap agents

```python
researcher = Agent("claude-opus-4-7", tools=[search], name="researcher")
editor     = Agent("claude-opus-4-7", tools=[researcher], name="editor")
print(editor("summarise AI trends April 2026").text())
```

Parallelism is emergent: when `editor` decides to call two tools in the
same turn, they run concurrently via `asyncio.gather`. No flag, no
config, no "parallel mode".

### 4 · Declared typed pipeline with resume

```python
from lazybridge import Agent, Plan, Step, Store, from_step

store = Store(db="pipeline.sqlite")

plan = Plan(
    Step(researcher, name="search", writes="hits",   output=Hits),
    Step(ranker,     name="rank",   task=from_step("search"), output=Ranked),
    Step(writer,     name="write",  task=from_step("rank")),
    store=store, checkpoint_key="research", resume=True,
)

Agent.from_engine(plan)("AI trends April 2026")
```

If a step fails mid-plan, the next run with `resume=True` retries from
the failing step only. If the plan is already done, it short-circuits
to the cached `writes` bucket.

### 5 · Human-in-the-loop with a full REPL

```python
from lazybridge import Agent, SupervisorEngine

sup = Agent(engine=SupervisorEngine(
    tools=[search],
    agents=[researcher],   # human can `retry researcher: <feedback>`
))
Agent.chain(researcher, sup, writer)("publish a policy brief")
```

Commands in the REPL: `continue`, `retry <agent>: <feedback>`,
`store <key>`, `<tool>(<args>)`. For approval-only flows use the
lighter [`HumanEngine`](docs/guides/human-engine.md) instead.

## Documentation

* **For humans** — [MkDocs site](https://selvaz.github.io/LazyBridge/):
  [Quickstart](docs/quickstart.md), per-tier pages, guides, decision
  trees ("when to use which"), API reference, errors table.
* **For LLM assistants** — a first-class
  [Claude Skill](lazybridge/skill_docs/SKILL.md) ships with the
  package: `00_overview`, `01_basic` … `04_advanced`,
  `05_decision_trees`, `06_reference`, `99_errors`. Same content as
  the site, rendered dense and signature-first for LLM consumption.
  A minimal [`llms.txt`](llms.txt) index points at both.

Documentation is single-source — fragment files under
`lazybridge/skill_docs/fragments/` render into the skill **and** the
site via `python -m lazybridge.skill_docs._build`. CI enforces no drift.

## What makes LazyBridge different

1. **Tool-is-Tool.** Functions, Agents, Agents-of-Agents all plug into
   `tools=[...]` with the same contract. `SupervisorEngine`,
   `LLMEngine`, and `Plan` all accept the same `tools=[...]` list.
2. **Compile-time plan validation.** `PlanCompileError` at construction
   catches broken DAGs before any LLM call. No other Python agent
   framework does this.
3. **Parallelism as capability.** No `tool_choice="parallel"` knob —
   the engine dispatches concurrent tool calls automatically.
4. **Claude Skill as first-class artifact.** Packaged with the library,
   loadable by Claude Code / any LLM assistant.

## Licence

Apache 2.0.
