# LazyBridge

[![tests](https://github.com/selvaz/LazyBridge/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/selvaz/LazyBridge/actions/workflows/test.yml)
[![docs](https://github.com/selvaz/LazyBridge/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/selvaz/LazyBridge/actions/workflows/docs.yml)
[![CodeQL](https://github.com/selvaz/LazyBridge/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/selvaz/LazyBridge/actions/workflows/codeql.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

> **Status: stable (1.0.1+).** The core public API (`Agent`, `Plan`,
> `Tool`, `Envelope`) will not break without a major version bump — see
> [CHANGELOG](CHANGELOG.md) and the [Maturity table](https://lazybridge.com/#maturity)
> for which subsystems are still Alpha/Experimental.

**Zero-boilerplate, multi-provider Python framework for LLM agents.** One
`Agent` class, swappable engines (LLM / Plan / Human / Supervisor), and one
tool contract — plain Python functions, other Agents, MCP servers, and full
pipelines all compose through `tools=[...]`. Parallelism is automatic when the
engine emits N tool calls in a turn; deterministic when you declare it.

```python
from lazybridge import Agent, LLMEngine

agent = Agent(
    engine=LLMEngine("claude-opus-4-8"),
)
result = agent("hello")
print(result.text())
```

That's the whole surface when you start. It grows only when your problem
grows.

- **Documentation:** <https://core.lazybridge.com>
- **Recipes:** <https://core.lazybridge.com/recipes/>
- **For LLM assistants** (Claude Skill, `llms.txt`):
  <https://core.lazybridge.com/for-llms/>

## The mental model

Every `Agent` is the composition `Engine + Tools + State`:

- **Engine** — what decides next. `LLMEngine` is the common case; swap for
  `Plan` (deterministic DAG), `HumanEngine` (approval gate), or
  `SupervisorEngine` (REPL).
- **Tools** — anything the agent can invoke. Functions, other `Agent`s,
  `Plan`-backed pipelines, MCP servers, and provider-native tools live in the
  same `tools=[...]` list.
- **State** — `Memory` (in-prompt history), `Store` (durable blackboard),
  `Session` (event bus + observability).

The same `Agent(engine=..., tools=..., ...)` shape supports a one-shot helper,
a hierarchical multi-agent system, and a checkpointed production pipeline —
only the `engine=` argument changes. See
[Concepts → Mental model](https://core.lazybridge.com/concepts/mental-model/).

## Pick your tier

LazyBridge grows with you — every tier is additive.

| Tier | For | Key imports |
|---|---|---|
| **[Basic](https://core.lazybridge.com/guides/basic/agent/)** | one-shot or tool-calling agents | `Agent` · `LLMEngine` · `Tool` · `NativeTool` · `Envelope` |
| **[Mid](https://core.lazybridge.com/guides/mid/memory/)** | real apps with memory, tracing, guardrails, composition | `Memory` · `Store` · `Session` · `Guard*` · `verify=` · `MCP` · `HumanEngine` · `EvalSuite` |
| **[Full](https://core.lazybridge.com/guides/full/plan/)** | production pipelines: typed hand-offs, routing, resume, OTel | `Plan` · `Step` · sentinels · `SupervisorEngine` · checkpoint · exporters |
| **[Advanced](https://core.lazybridge.com/guides/advanced/engine-protocol/)** | extending the framework | `BaseProvider` · `Plan.to_dict` · custom engines · OpenTelemetry · Visualizer |

See [Decisions → Pick your tier](https://core.lazybridge.com/decisions/pick-tier/)
for a flowchart.

## Install

> **PyPI version note.** Releases before `0.7.0` (`0.4.x` and the
> withdrawn `1.0.0`) expose the **legacy `LazyAgent` / `LazyTool` /
> `LazySession` API and do not match this README** — see
> [Migrating from 1.0.0](https://lazybridge.com/migrations/1.0-to-0.7/)
> if you have one of those installed. Pin `lazybridge>=1.0.1`:
>
> ```bash
> pip install "lazybridge[anthropic]"
> ```

```bash
pip install "lazybridge[anthropic]"
# or [openai], [google], [deepseek], [litellm], [yaml], [otel], [encryption], [all]
# Concrete tools (MCP, Gmail, Telegram, gateways, doc readers) ship in the
# sibling lazytoolkit package: pip install "lazytoolkit[mcp]"  (see https://tools.lazybridge.com/)
```

> **Naming note — `lazytoolkit` vs `lazytools`.** The concrete tools ship in a
> single sibling package whose **distribution name** (what you `pip install`) is
> **`lazytoolkit`**, while its **import name** (what you write in code) is
> **`lazytools`**:
>
> ```bash
> pip install "lazytoolkit[mcp]"            # distribution name
> ```
> ```python
> from lazytools.connectors.mcp import MCP  # import name
> ```
>
> This mirrors well-known packages such as `pip install beautifulsoup4` →
> `import bs4`.

Confirm you're on the modern API:

```python
import lazybridge
assert lazybridge.__version__.startswith(("0.7", "0.8", "0.9", "1.")), (
    f"LazyBridge {lazybridge.__version__} predates the modern API — this "
    f"README requires >=0.7.9.  See https://github.com/selvaz/LazyBridge."
)
```

Set an API key for your provider of choice (`ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `DEEPSEEK_API_KEY`).

## Worked examples

### 1 · Function becomes a tool, auto-schema

```python
from lazybridge import Agent, LLMEngine, Tool


def get_weather(city: str) -> str:
    """Return current temperature and conditions for ``city``."""
    return f"{city}: 22°C, sunny"


agent = Agent(
    engine=LLMEngine("claude-opus-4-8"),
    tools=[Tool.wrap(get_weather, name="get_weather")],
)
result = agent("what's the weather in Rome and Paris?")
print(result.text())
```

No decorators, no JSON schemas. Type hints + docstring become the tool's
LLM-facing schema automatically.  The explicit `Tool.wrap(fn, name=...)`
factory pins the LLM-visible name so refactors don't break tool-maps
or plan references; the bare-callable form `tools=[get_weather]` works
too (backward-compatible auto-wrap). See
[Guides → Basic → Tool](https://core.lazybridge.com/guides/basic/tool/).

### 2 · Native tools (no code at all)

```python
from lazybridge import Agent, LLMEngine, NativeTool

agent = Agent(
    engine=LLMEngine("claude-opus-4-8"),
    native_tools=[NativeTool.WEB_SEARCH],
)
agent("AI news this week")
```

`WEB_SEARCH` · `CODE_EXECUTION` · `FILE_SEARCH` · `COMPUTER_USE` ·
`GOOGLE_SEARCH` · `GOOGLE_MAPS` (each supported by a subset of providers).
`CODE_EXECUTION` and `COMPUTER_USE` require
`allow_dangerous_native_tools=True` — they execute code or click your screen.

### 3 · Tool-is-tool — agents wrap agents

```python
from lazybridge import Agent, LLMEngine

def search(query: str) -> str:
    """Search the web and return a short result (stub for the example)."""
    return f"results for {query!r}"


researcher = Agent(
    engine=LLMEngine("claude-opus-4-8"),
    tools=[search],
    name="research",
)
editor = Agent(
    engine=LLMEngine("claude-opus-4-8"),
    tools=[researcher],
    name="editor",
)
result = editor("summarise AI trends April 2026")
print(result.text())
```

Parallelism is emergent: when `editor` decides to call two tools in the same
turn, they run concurrently via `asyncio.gather`. No flag, no config, no
"parallel mode".

### 4 · MCP servers as tool catalogues

```python
from lazybridge import Agent, LLMEngine
from lazytools.connectors.mcp import MCP  # pip install lazytoolkit (import name: lazytools)

fs = MCP.stdio(
    "fs",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp/project"],
    allow=["fs.read_*", "fs.list_*"],   # required since 0.7.9 (deny-by-default)
    cache_tools_ttl=60.0,
)
agent = Agent(
    engine=LLMEngine("claude-opus-4-8"),
    tools=[fs],
)
agent("Read README.md and summarise the install steps")
```

The MCP server expands into one LazyBridge `Tool` per remote tool — no
separate engine, no graph wrappers. See
[LazyTools → MCP](https://tools.lazybridge.com/mcp/).

### 5 · Declared typed pipeline with crash resume

```python
from lazybridge import Agent, LLMEngine, Plan, Step, Store, from_prev, from_step

store = Store(db="pipeline.sqlite")

researcher = Agent(engine=LLMEngine("claude-opus-4-8"), name="search")
ranker     = Agent(engine=LLMEngine("claude-opus-4-8"), name="rank")
writer     = Agent(engine=LLMEngine("gpt-5.4-mini"),          name="write")

pipeline = Agent(
    engine=Plan(
        Step("search", writes="hits"),
        Step("rank",
             task="Rank these search hits by relevance; return the top 5.",
             context=from_prev),
        Step("write",
             task="Write a 200-word brief from the ranked items.",
             context=from_step("rank")),
        store=store,
        checkpoint_key="research",
        resume=True,
    ),
    tools=[researcher, ranker, writer],
)
pipeline("AI trends April 2026")
```

If a step fails mid-plan, the next run with `resume=True` retries from the
failing step only. Concurrent runs on the same `checkpoint_key` are serialised
via `compare_and_swap` — first writer wins, second raises
`ConcurrentPlanRunError`. Pass `on_concurrent="fork"` for fan-out workflows.
See [Guides → Full → Checkpoint & resume](https://core.lazybridge.com/guides/full/checkpoint/).

### 6 · Human-in-the-loop with a full REPL

```python
from lazybridge import Agent, LLMEngine
from lazybridge.ext.hil import supervisor_agent

sup = supervisor_agent(
    tools=[search],
    agents=[researcher],   # human can `retry research: <feedback>`
)
result = sup("publish a policy brief")
print(result.text())
```

REPL commands: `continue`, `retry <agent>: <feedback>`, `store <key>`,
`<tool>(<args>)`. For approval-only flows use the lighter `human_agent(...)`
or `HumanEngine` — see
[Decisions → HumanEngine vs SupervisorEngine](https://core.lazybridge.com/decisions/human-engine-vs-supervisor/).

## What makes LazyBridge different

1. **Tool-is-Tool.** Functions, Agents, Agents-of-Agents, `Plan`-backed
   pipelines, and tool providers (MCP servers, external HTTP gateways) all
   plug into `tools=[...]` with the same contract.
2. **Compile-time plan validation.** `PlanCompileError` at construction
   catches broken DAGs — duplicate names, forward references, broken
   `from_step` / `from_parallel` sentinels — before any LLM call.
3. **CAS-protected crash resume.** `Plan` checkpoints to `Store` via
   `compare_and_swap`. Two concurrent runs on the same `checkpoint_key`
   deterministically converge instead of silently overwriting.
4. **Parallelism as capability.** When the engine emits N tool calls in one
   turn, they run concurrently via `asyncio.gather`. No flag, no
   `tool_choice="parallel"` knob.
5. **Transitive cost roll-up.** `Envelope.metadata.nested_*` aggregates token
   / cost telemetry across an Agent-of-Agents tree — the outer envelope
   reports total pipeline spend without double-counting.
6. **OTel GenAI conventions out of the box.** `OTelExporter` ships
   `gen_ai.*` attributes and proper parent-child spans; existing GenAI
   dashboards render LazyBridge traces unchanged.
7. **First-class LLM-assistant artifact.** A signature-first Claude Skill
   ships with the library at `lazybridge/skill/`, loadable by any LLM
   coding assistant. See
   [For LLM assistants](https://core.lazybridge.com/for-llms/).

## Documentation

The full docs live at <https://core.lazybridge.com>. Highlights:

- **[Concepts](https://core.lazybridge.com/concepts/mental-model/)** —
  the mental model, "everything is a tool", progressive complexity, and
  canonical-vs-sugar.
- **[Guides](https://core.lazybridge.com/guides/basic/agent/)** —
  one focused page per public concept, all following the same
  Signature → Synopsis → When to use / NOT → Example → Pitfalls →
  See also template.
- **[Recipes](https://core.lazybridge.com/recipes/)** — runnable
  examples from `examples/`, embedded verbatim.
- **[Decisions](https://core.lazybridge.com/decisions/)** — "which
  one do I use?" trees for tier, return type, state layer, composition,
  parallelism, HumanEngine vs SupervisorEngine, `verify=` placement,
  checkpointing.
- **[Errors](https://core.lazybridge.com/errors/)** — cause → diagnosis
  → fix table for every framework exception.
- **[For LLM assistants](https://core.lazybridge.com/for-llms/)** —
  Claude Skill install, `/llms.txt` index, `/llms-full.txt` corpus.

## Contributing

Issues and PRs welcome at <https://github.com/selvaz/LazyBridge>. Run the
test suite with `pip install -e ".[test,all]"` then `pytest`. See
[SECURITY.md](SECURITY.md) for the disclosure policy.

## Licence

Apache 2.0 — see [LICENSE](LICENSE).

---

## How This Was Built

LazyBridge is designed by **selvaz** with **Claude Code** and
**ChatGPT Codex** as primary implementation partners.
I focus on architecture, mental model, and trade-offs —
they handle the building under my direction.
