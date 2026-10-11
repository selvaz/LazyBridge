# LLM codegen contract

This contract targets LazyBridge 1.8.0 plus post-release transport fixes.
Read the current source signatures; do not copy legacy LazyAgent/LazyTool APIs.
The machine-readable companion is
[lazybridge/llms.json](https://github.com/selvaz/LazyBridge/blob/main/lazybridge/llms.json);
it includes the engine, continuity and integration rules summarized here.

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

Read [coding agents](../guides/full/coding-agents.md),
[collaboration](../guides/full/claude-writes-codex-reviews.md),
[error recovery](error-recovery.md) and [reference](../reference/index.md).
