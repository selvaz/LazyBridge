# LLM codegen contract

This contract targets LazyBridge 1.8.0 plus post-release transport fixes.
Read the current source signatures; do not copy legacy LazyAgent/LazyTool APIs.
The machine-readable companion is
[lazybridge/llms.json](https://github.com/selvaz/LazyBridge/blob/main/lazybridge/llms.json);
it includes the engine, continuity and integration rules summarized here.

## Always

- Prefer top-level imports for core primitives and Agent(engine=...) construction.
- Name reusable agents and pin Tool.wrap names when tool-map or Plan references
  must survive Python refactoring. Raw typed callables are supported too.
- Read payload for typed results and text() for text; check ok/error before use.
- Keep model/system/native policy on the engine, and output/verify on Agent.
- Use provider tier aliases for catalogue selection only when their current
  mapping matches requirements; pin and validate a model ID when needed.
- Execute generated examples with mock transports, then run appropriate
  acceptance checks before side-effecting work.

## Canonical construction

```python
from lazybridge import Agent, ClaudeCodeEngine, CodexEngine, Memory, Session, Store

def project_status() -> str:
    """Return a deterministic status."""
    return "Ready for review."

state = dict(tools=[project_status], memory=Memory(), store=Store(), session=Session())
claude = Agent(name="claude", engine=ClaudeCodeEngine(), **state)
codex = Agent(name="codex", engine=CodexEngine(), **state)
assert claude.memory is codex.memory
```

Use Agent(engine=...) and explicit names for reusable agents and Plan-backed
pipelines. Put model/system/reasoning/native policy and engine retries/deadlines
on the engine. For provider APIs use LLMEngine or Agent.from_provider with a tier
alias; catalogue tiers are distinct from LazyTools quota routing.

Plain callables auto-wrap; Tool.wrap(fn, name=...) pins a public name.
Named Agents, Plan-backed Agents and ToolProvider instances also enter tools.
Provider NativeTool values enter native_tools on LLMEngine. They execute
server-side and are not interchangeable with Python tools. Dangerous provider
capabilities require allow_dangerous_native_tools=True.

## Review and continuity

```python
from lazybridge import Agent, ClaudeCodeEngine, CodexEngine

reviewer = Agent(name="reviewer", engine=CodexEngine(
    system='Return "approved" or "rejected: <reason>". Check boundary cases.',
))
writer = Agent(name="writer", engine=ClaudeCodeEngine(), verify=reviewer, max_verify=3)
result = writer("Draft a weekday deployment rule with input validation.")
print(result.text())
```

verify is an Agent/as_tool constructor option, not a run kwarg. max_verify counts
total attempts. Unknown verdicts reject; exhaustion returns the final draft
with no approval flag. result.ok is execution success. Require deterministic
tests/explicit acceptance records before shipping. The core permits any judge;
select the opposite engine explicitly for independent review.

Native persistence uses Claude persist_session/session_id or Codex
persist_thread/thread_id. Both take session_alias/session_registry.
SessionRegistry keys include kind and cwd. Explicit ID wins over an alias.
Claude session_name is a runtime slot label, not session_alias or a title.
Native resume omits Memory reinjection. Shared Python state does not migrate a
native transcript. Coordinate cross-process transcript access yourself.

Claude writes need extra_tools as well as policy. writer(gate) does not expose
Write/Edit/Bash. File roots confine file hooks, not Bash. Codex writes need
workspace-write and potentially extra writable_roots for Git worktrees.
Read-only native settings do not sandbox Python application tools.

## Return and composition contracts

```python
from lazybridge import Agent
from lazybridge.testing import MockAgent

fan = Agent.parallel(MockAgent(["A"], name="alpha"), MockAgent(["B"], name="beta"))
result = fan("Compare the approaches.")
assert "[alpha]" in result.text() and "[beta]" in result.text()
print(result.text())
```

Every agent call returns Envelope. text() returns text; payload carries the
output type; error is ErrorInfo or None. Agent.parallel returns a labelled-text
join in one Envelope. Use await fan.run_branches(task) for list[Envelope];
do not assume the normal join's payload is a list.

Agent.chain is a linear Plan convenience; Plan/Step supports declared routing,
parallel bands and checkpoints. Instructions live on engines. Agent.run is the
async entry point; agent(task) blocks in existing async loops too.

### Python tools and provider-native tools

```python
from lazybridge import Agent, LLMEngine, NativeTool, Tool

def search_web(query: str) -> str:
    """Return a deterministic search fixture, not live search results."""
    return f"Demonstration result for {query}"

local = Agent(name="local-search", engine=LLMEngine("claude-opus-4-7"),
              tools=[Tool.wrap(search_web, name="search_web")])
native = Agent(name="native-search", engine=LLMEngine(
    "claude-opus-4-7", native_tools=[NativeTool.WEB_SEARCH],
))
```

The application executes local tools; the provider executes native capabilities.
Put allow_dangerous_native_tools=True on LLMEngine if deliberately enabling
dangerous native capabilities. Do not put native_tools on Agent.

### Bounded execution defaults

These example limits are choices for this workload, not universal thresholds.
Keep inner model/tool budgets distinct from outer Agent deadlines:

```python
from lazybridge import Agent, LLMEngine
from pydantic import BaseModel

class Answer(BaseModel):
    summary: str

engine = LLMEngine(
    "claude-opus-4-7", max_turns=8, max_retries=3,
    request_timeout=60, tool_timeout=30, max_parallel_tools=4,
    stream_idle_timeout=90,
)
bounded = Agent(name="bounded", engine=engine, output=Answer, timeout=120)
result = bounded("Summarize the task in one sentence.")
assert isinstance(result.payload, Answer)
```

For coding engines, use the native request/idle/tool limits in their guides.
Choose native write exposure and approval explicitly; no timeout makes an
unsafe tool safe. Human approvals of runtime actions and final diff acceptance
are separate steps.

### Canonical deterministic and nested patterns

```python
from lazybridge import Agent, ClaudeCodeEngine, Plan, Step, from_step
from lazybridge.testing import MockAgent

research = MockAgent("research complete", name="research")
write = MockAgent("draft complete", name="write")
pipe = Agent(name="pipeline", engine=Plan(
    Step("research"), Step("write", context=from_step("research")),
), tools=[research, write])
assert pipe("Draft a policy").text() == "draft complete"

outer = Agent(name="coordinator", engine=ClaudeCodeEngine(), tools=[pipe])
```

Agent.chain is the linear convenience; Agent.parallel joins labelled text;
as_tool(verify=judge) supplies a verified view of a named sub-agent. For MCP
catalogues use LazyTools providers and their explicit exposure policy, rather
than inventing a Core MCPEngine or importing a removed connector shim.

### Envelope return details

Envelope carries task, context, attachments, payload, metadata and ErrorInfo.
Nested token/cost fields supplement the current engine's usage; cost_usd alone
is not a universal subscription charge or total-spend guarantee. Codex reports
no per-turn price. Errors are records, not a propagated Exception object.
Parallel branch Envelopes come from run_branches, not the ordinary join payload.

## Import map

| Capability | Current import/destination |
| --- | --- |
| Coding engines/config/registry | lazybridge ClaudeCodeEngine, CodexEngine, CodingAgentConfig, SessionRegistry |
| Human workflow | lazybridge.ext.hil |
| Approval/delegation | lazybridge.ext.approval / lazybridge.ext.delegation |
| Evals/planners/knowledge | lazybridge.ext.evals / planners / knowledge |
| Observability/visualizer | lazybridge.ext.otel / viz |
| MCP/gateway/code support | lazytools.connectors; [LazyTools docs](https://tools.lazybridge.com/) |
| Durable repository jobs | [LazyTools code bridge](https://tools.lazybridge.com/code-bridge/) |
| Project records/quota routing | [LazyTools projects](https://tools.lazybridge.com/projects/) and code bridge |
| Reporting | Sibling LazyReport package |

LazyTools distribution name is lazytoolkit, import namespace lazytools.
Removed lazybridge.ext.mcp/gateway and external_tools shims are not valid imports.
MCP.stdio requires an explicit allow or deny policy. Coding MCP providers opt in;
--allow-unsafe is tool exposure, not the bridge's durable approval queue.

## Avoid legacy and unsupported shapes

- No LazyAgent/LazyTool/LazySession/LazyContext.
- No Agent.from_chain/from_engine/from_model/from_plan/from_parallel.
  Use Agent(engine=...), Agent.chain or Agent.parallel.
- No Tool(mode="auto"); choose signature, llm or hybrid.
- No tool_choice="parallel".
- No system= on Agent; use the engine.
- No BaseEngine protocol; import Engine from lazybridge.engines.base.
- No async callable verify judge assumption; core callable judges are synchronous.
- Do not promise cross-engine native history migration, cross-process transcript
  locks, automatic uncertain-turn replay or verification acceptance on exhaustion.

## Never duplicate a tool implementation just to expose it

Wrap a library's annotated function directly when its signature is suitable.
Use signature mode for deterministic schemas; hybrid/llm modes are deliberate
schema-generation choices with model cost and validation tradeoffs. A wrapper
is useful for a missing output cap, provenance shape or approval boundary;
it should call the library implementation rather than reimplement it.

For shared expensive state, load once into Store and pass a key among tools.
See [tool wrapping](../decisions/tool-wrapping.md) and
[schema modes](../guides/basic/tool.md#schema-modes-signature-hybrid-llm).
Use await agent.run in an async application; the blocking helper is suitable
for synchronous scripts, not a reason to stall an async handler.

## Error messages as codegen aids

PlanCompileError, PlanRuntimeError and UnsupportedFeatureError often identify
the offending step/tool/value, the expected contract and a concrete fix. Read
the complete error before changing the code. Do not assume every exception has
the same four-part format, or treat a suggested fix as authorization to widen
permissions. Consult [error recovery](error-recovery.md) for runtime failures.

Read [coding agents](../guides/full/coding-agents.md),
[collaboration](../guides/full/claude-writes-codex-reviews.md),
[error recovery](error-recovery.md) and [reference](../reference/index.md).
