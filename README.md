# LazyBridge

**Zero-boilerplate Python agent framework.** One `Agent`, swappable
engines (LLM / Plan / Human / Supervisor), and one tool contract —
plain functions, other Agents, and Agents-of-Agents all compose the
same way. Parallelism is automatic when the engine decides;
deterministic when you declare it.

Pipelines are validated at construction time: a misspelled step name,
a forward-reference in `from_step`, or a parallel-band misuse surfaces
as `PlanCompileError` **before any LLM call** — not at the first
production failure.

```python
from lazybridge import Agent
print(Agent("claude-opus-4-7")("hello").text())
```

## Documentation — two tracks

LazyBridge ships docs for two audiences. **Same source of truth** —
fragment files under `lazybridge/skill_docs/fragments/` render into
both surfaces via `python -m lazybridge.skill_docs._build` (CI enforces
no drift).

| Audience | Where | Style |
|---|---|---|
| **Humans** | [MkDocs site](https://selvaz.github.io/LazyBridge/) — [Quickstart](docs/quickstart.md) → [Getting started](docs/guides/getting-started.md) → [guides](docs/guides/) → [decision trees](docs/decisions/) → [recipes](docs/recipes/) → [API reference](docs/reference.md) | Narrative-first; mermaid diagrams; copy-paste recipes |
| **LLM assistants** | [Claude Skill](lazybridge/skill_docs/SKILL.md) packaged with the wheel; mirrored at [`docs/skill/`](docs/skill/); [`llms.txt`](llms.txt) entry-point | Signature-first; rules block; dense; predictable section structure (`signature` / `rules` / `example` / `pitfalls`) |

Both tracks cover the same surface — Basic / Mid / Full / Advanced
tiers, decision trees ("when to use which"), and the errors table.

## Pick your tier

LazyBridge grows with you — every tier is additive.

| Tier | For | Key imports |
|---|---|---|
| [**Basic**](docs/guides/getting-started.md#basic--one-call) | one-shot or tool-calling agents | `Agent` · `Tool` · `NativeTool` · `Envelope` |
| [**Mid**](docs/guides/getting-started.md#mid--state-observability-multi-agent) | real apps with memory, tracing, guardrails, composition | `Memory` · `Store` · `Session` · `Guard*` · `chain` · `parallel` · `as_tool` · `MCP` · `HumanEngine` · `EvalSuite` |
| [**Full**](docs/guides/getting-started.md#full--declared-pipelines-crash-resume) | production pipelines: typed hand-offs, routing, resume, OTel | `Plan` · `Step` · `from_prev`/`from_step`/`from_parallel` · `SupervisorEngine` · checkpoint · exporters · `verify=` |
| [**Advanced**](docs/guides/getting-started.md#advanced--extending-the-framework) | extending the framework | `BaseProvider` · `Plan.to_dict` · `register_provider_*` · `core.types` |

## Install

```bash
pip install lazybridge[anthropic]   # or [openai], [google], [deepseek], [litellm], [mcp], [otel], [all]
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

No decorators, no JSON schemas. If your function lacks type hints,
pass `mode="llm"` to have a cheap agent infer the schema — see
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

Parallelism is emergent: when `editor` decides to call two tools in
the same turn, they run concurrently via `asyncio.gather`. No flag,
no config, no "parallel mode".

### 4 · MCP servers as tool catalogues

```python
from lazybridge import Agent
from lazybridge.ext.mcp import MCP

fs = MCP.stdio(
    "fs",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp/project"],
    cache_tools_ttl=60.0,            # refresh tool list every 60s
)
agent = Agent("claude-opus-4-7", tools=[fs])
agent("Read README.md and summarise the install steps")
```

The MCP server expands into one LazyBridge `Tool` per MCP tool — no
separate engine, no graph wrappers. See [the MCP recipe](docs/recipes/mcp.md).

### 5 · Declared typed pipeline with resume

```python
from lazybridge import Agent, Plan, Step, Store, from_prev, from_step

store = Store(db="pipeline.sqlite")

# Idiomatic shape: each step has an explicit task instruction; upstream
# data flows through `context=`.  `context=[from_step("a"), from_step("b")]`
# also works to pull from multiple upstream steps without a combiner.
plan = Plan(
    Step(researcher, name="search",
         writes="hits", output=Hits),
    Step(ranker,     name="rank",
         task="Rank these search hits by relevance; return the top 5.",
         context=from_prev,
         output=Ranked),
    Step(writer,     name="write",
         task="Write a 200-word brief from the ranked items.",
         context=from_step("rank")),
    store=store, checkpoint_key="research", resume=True,
)

Agent.from_engine(plan)("AI trends April 2026")
```

If a step fails mid-plan, the next run with `resume=True` retries
from the failing step only. If the plan is already done, it
short-circuits to the cached `writes` bucket. Concurrent runs on the
same `checkpoint_key` are serialised via CAS — pass
`on_concurrent="fork"` for fan-out workflows.

### 6 · Production observability (OTel GenAI conventions)

```python
from lazybridge import Agent, Session, JsonFileExporter
from lazybridge.ext.otel import OTelExporter

sess = Session(
    db="events.sqlite",
    batched=True,                     # non-blocking emit
    on_full="hybrid",                 # default — block on AGENT_*/TOOL_*, drop telemetry
    exporters=[
        JsonFileExporter(path="run.jsonl"),
        OTelExporter(endpoint="http://otelcol:4318"),
    ],
)

researcher = Agent("claude-opus-4-7", tools=[search], name="researcher", session=sess)
print(researcher("summarise this week's AI news").text())

# Cost / tokens / latency breakdown across nested agents.
print(sess.usage_summary())
sess.flush()                          # drain the writer before exit
```

`OTelExporter` emits `gen_ai.system` / `gen_ai.usage.input_tokens` /
`gen_ai.tool.call.id` and the rest of the OpenTelemetry GenAI Semantic
Conventions, with proper parent-child span hierarchy
(`invoke_agent → chat / execute_tool`).

### 7 · Human-in-the-loop with a full REPL

```python
from lazybridge import Agent
from lazybridge.ext.hil import SupervisorEngine

sup = Agent(engine=SupervisorEngine(
    tools=[search],
    agents=[researcher],   # human can `retry researcher: <feedback>`
))
agents = [researcher, sup, writer]
Agent.chain(*agents)("publish a policy brief")
```

Commands in the REPL: `continue`, `retry <agent>: <feedback>`,
`store <key>`, `<tool>(<args>)`. For approval-only flows use the
lighter [`HumanEngine`](docs/guides/human-engine.md) instead.

## Top tasks

* [Tool calling end-to-end](https://selvaz.github.io/LazyBridge/recipes/tool-calling/)
* [Structured output with Pydantic](https://selvaz.github.io/LazyBridge/recipes/structured-output/)
* [Pipeline with typed steps and crash resume](https://selvaz.github.io/LazyBridge/recipes/plan-with-resume/)
* [Human-in-the-loop: approval gates and REPL](https://selvaz.github.io/LazyBridge/recipes/human-in-the-loop/)
* [MCP integration](https://selvaz.github.io/LazyBridge/recipes/mcp/)
* [Decision trees — "when to use which"](https://selvaz.github.io/LazyBridge/decisions/)

## What makes LazyBridge different

1. **Tool-is-Tool.** Functions, Agents, Agents-of-Agents, and tool
   providers (e.g. an MCP server) all plug into `tools=[...]` with
   the same contract. `SupervisorEngine`, `LLMEngine`, and `Plan` all
   accept the same list.
2. **Compile-time plan validation.** `PlanCompileError` at
   construction catches broken DAGs — duplicate names, forward
   references, broken `from_step` / `from_parallel` sentinels,
   parallel-band misuse — before any LLM call.
3. **CAS-protected crash resume.** `Plan` checkpoints to `Store` via
   `compare_and_swap`. Two concurrent runs on the same
   `checkpoint_key` deterministically converge — first writer wins,
   second raises `ConcurrentPlanRunError` instead of silently
   overwriting. `on_concurrent="fork"` isolates parallel runs.
4. **Parallelism as capability.** When the engine emits N tool calls
   in one turn, they run concurrently via `asyncio.gather`. No flag,
   no `tool_choice="parallel"` knob.
5. **Transitive cost roll-up.** `Envelope.metadata.nested_*` aggregates
   token / cost telemetry across an Agent-of-Agents tree — the outer
   envelope reports total pipeline spend without double-counting.
6. **OTel GenAI conventions out of the box.** `OTelExporter` ships
   `gen_ai.*` attributes and proper parent-child spans; existing
   GenAI dashboards render LazyBridge traces unchanged.
7. **LLM-assistant skill as first-class artifact.** A signature-first
   `SKILL.md` ships with the library, loadable by any LLM assistant.

## Licence

Apache 2.0.
