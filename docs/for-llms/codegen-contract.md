# LLM codegen contract

If you're an LLM assistant generating LazyBridge code from a user
request, this page is the authoritative cheat sheet. The strict
machine-readable form is
[`lazybridge/llms.json`](https://github.com/selvaz/LazyBridge/blob/main/lazybridge/llms.json);
this page is the same information annotated for human review.

## Always

- Import from the top level: `from lazybridge import Agent, LLMEngine, Tool`.
- Construct agents canonically: `Agent(engine=LLMEngine("..."))`.
- Name every reusable agent — `Agent(..., name="reviewer")` — so it
  can be referenced as a tool and so the Plan compiler can validate
  sentinels against it.
- Wrap plain functions explicitly: `Tool.wrap(fn, name="...")`.  Raw
  callables still work as a convenience, but the explicit form is
  the canonical contract.  The bare `Tool(...)` constructor is
  available for advanced use (custom-built schemas, schema-cache
  artefacts) — **prefer the `Tool.wrap()` factory** in new code.
- Read `result.text()` for string output; `result.payload` for the
  typed structured output when `output=` was set on the Agent.
- For freshest-model quickstarts, use
  `Agent.from_provider("anthropic", tier="top")` — the tier alias
  resolves to whichever SKU is current.

## Never

- Don't use `LazyAgent`, `LazyTool`, `LazySession`, or
  `LazyContext` — those are the 0.4-era names; current API is
  `Agent`, `Tool.wrap()` / `Tool`, `Session`, and the
  sentinels/`Store`/`Memory` triple.
- Don't pass `mode="auto"` to `Tool.wrap()` or `Tool(...)` — the
  graceful-fallback ladder was deleted in 0.7.9. Choose
  `mode="signature"` (introspect type hints, default),
  `mode="llm"` (delegate the schema to an LLM), or `mode="hybrid"`
  (introspect, then have the LLM enrich).
- Don't call `Agent.from_chain` / `from_engine` / `from_model` /
  `from_plan` / `from_parallel` — all five aliases were deleted in
  0.7.9. Use `Agent(engine=...)` or `Agent.chain` /
  `Agent.parallel`.
- Don't pass `tool_choice="parallel"` — never supported, removed
  from the public ladder in 0.7.9.
- Don't write JSON schemas by hand for annotated Python functions.
  `Tool.wrap()` introspects type hints; if the introspection misses a
  detail, override it with `mode="hybrid"` rather than rebuilding
  the schema manually.
- Don't call `MCP.stdio(...)` without `allow=` or `deny=` — it's
  deny-by-default since 0.7.9 and will raise `ValueError`. Pass
  `allow=["*"]` after auditing the surface, or a glob list like
  `allow=["fs.read_*", "fs.list_*"]`.
- Don't import `lazybridge.external_tools.report_builder` — the
  reporting subsystem moved to the sibling package
  [`lazybridge-reports`](https://github.com/selvaz/LazyReport) in
  0.7.9.
- Don't wrap the entry point in `asyncio.run(main())` for
  quickstarts. `Agent.__call__` auto-detects an event loop; the
  sync path is the normal first-contact experience.

## `tools=` vs `native_tools=`

These two parameters look similar and route to very different
runtimes:

| Parameter | Executor | Use for |
|---|---|---|
| `tools=[...]` | **LazyBridge** runs them | Python functions, sub-Agents, MCP servers, schema-backed `Tool` instances |
| `native_tools=[...]` | **Provider** runs them | `NativeTool.WEB_SEARCH`, `NativeTool.CODE_EXECUTION`, `NativeTool.COMPUTER_USE`, etc. — server-side tools the LLM provider executes itself |

Dangerous server-side tools (`CODE_EXECUTION`, `COMPUTER_USE`)
require `allow_dangerous_native_tools=True` on the Agent. The flag
is intentionally noisy — sandboxed code execution and computer-use
expose a real attack surface and the opt-in is the only signal
LazyBridge has that the operator understands the trust boundary.

Side-by-side example (same task, two runtimes):

```python
from lazybridge import Agent, LLMEngine, Tool

# LazyBridge-executed web search via your own function.
def search_web(query: str) -> str:
    """Search the web — your implementation, your retry/quota policy."""
    ...

local = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    tools=[Tool.wrap(search_web, name="search_web")],
)

# Provider-executed web search via Anthropic's server-side tool.
from lazybridge.core.types import NativeTool

native = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    native_tools=[NativeTool.WEB_SEARCH],
)
```

## Extension import map

LazyBridge's framework extensions live under `lazybridge.ext.*`; the
top-level package only re-exports core primitives. Reach for these
imports when you need the corresponding capability:

| Capability | Import |
|---|---|
| Human-in-the-loop CLI / web UI | `from lazybridge.ext.hil import HumanEngine, human_agent` |
| Supervisor pattern | `from lazybridge.ext.hil import SupervisorEngine, supervisor_agent` |
| MCP servers (stdio / HTTP) | `from lazybridge.ext.mcp import MCP` |
| OpenTelemetry exporter | `from lazybridge.ext.otel import OTelExporter` |
| Eval harness | `from lazybridge.ext.evals import EvalSuite, EvalCase, llm_judge` |
| Live visualizer | `from lazybridge.ext.viz import Visualizer` |
| Blackboard planner | `from lazybridge.ext.planners import BlackboardPlanner` |
| HTTP gateway (`lazybridge serve`) | `from lazybridge.ext.gateway import build_app` |

Reporting (`lazybridge_reports.*`) lives in the sibling package
[lazybridge-reports](https://github.com/selvaz/LazyReport) — install
separately, do **not** look for it under `lazybridge.external_tools`.

## Production-safe defaults

Copy-paste this snippet whenever you need bounded behaviour. Each
knob has a sensible default; explicit values make the agent
resistant to runaway requests, infinite tool loops, and silent stream
stalls:

```python
from lazybridge import Agent, LLMEngine

engine = LLMEngine(
    "claude-opus-4-7",
    max_turns=8,             # cap the inner LLM/tool loop
    max_retries=3,           # transient-error retries (exponential backoff)
    request_timeout=60,      # whole-request wall clock (seconds)
    tool_timeout=30,         # per-tool wall clock (seconds)
    max_parallel_tools=4,    # bound on concurrent tool calls
    stream_idle_timeout=90,  # raise StreamStallError if no token in 90s
)

agent = Agent(
    engine=engine,
    name="my_agent",
    tools=[...],
    output=...,         # optional Pydantic model for structured output
)
```

For provider-side dangerous tools (`CODE_EXECUTION`, `COMPUTER_USE`)
also pass `allow_dangerous_native_tools=True` on the Agent — the flag
is intentionally noisy.

## Canonical patterns

The shortest correct shape for each common request:

| Request | Code |
|---|---|
| One agent | `agent = Agent(engine=LLMEngine("claude-opus-4-7"))` |
| Agent + function tool | `agent = Agent(engine=LLMEngine("claude-opus-4-7"), tools=[Tool.wrap(get_weather, name="get_weather")])` |
| Sequential plan | `Agent(engine=Plan(Step("research"), Step("write")), tools=[research, write], name="pipe")` |
| Parallel fan-out | `fan = Agent.parallel(researcher, analyst); env = fan("topic")` |
| Linear chain | `pipe = Agent.chain(researcher, writer)` |
| Sub-agent as tool | `Agent(engine=LLMEngine(...), tools=[other_agent])` (auto-wrapped) |
| Sub-agent with verifier | `Agent(engine=LLMEngine(...), tools=[other_agent.as_tool(verify=judge)])` |
| MCP server | `Agent(..., tools=[MCP.stdio("fs", command="npx", args=[...], allow=["fs.read_*"])])` |
| Structured output | `Agent(engine=LLMEngine(...), output=Summary)` (where `Summary` is a Pydantic model) |

## Envelope return contract

Every `agent(...)` returns an `Envelope[T]`:

```python
result = agent("do the thing")
result.text()             # -> str — final assistant text
result.payload            # -> T — typed payload when output= was set; else str
result.error              # -> Exception | None — propagated runtime failure
result.metadata.cost_usd  # -> float — pipeline cost (rolls up across nested agents)
```

For `Agent.parallel(...)` post-0.7.9: the returned envelope's
`.text()` joins each branch's text with `[branch_name]` labels,
and `.payload` is `list[Envelope]` for typed per-branch access.

## Error messages are codegen aids

Every `PlanCompileError` / `PlanRuntimeError` /
`UnsupportedFeatureError` carries a four-part body:

1. What failed (step / tool / sentinel by name)
2. What was passed (the offending value)
3. What's expected (the valid set)
4. **A concrete fix snippet** — usually the exact replacement code

Read the bottom of the error string verbatim; the fix is the
replacement source line.
