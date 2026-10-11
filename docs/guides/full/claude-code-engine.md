# Claude Code Engine

`ClaudeCodeEngine` runs the model/tool loop through the locally authenticated
Claude Code runtime via the Claude Agent SDK. It satisfies the same
[Engine protocol](../advanced/engine-protocol.md) as Codex, provider APIs and
Plan. Tools and application state remain LazyBridge objects.

## Setup

Use Python 3.11 or later and install the extra:

```sh
python -m pip install "lazybridge[claude-code]==1.8.0"
claude doctor
```

Install Claude Code and sign in by following the current
[official setup instructions](https://code.claude.com/docs/en/setup).
Run `claude` interactively to authenticate. With a subscription login,
the runtime uses plan quota; the SDK also supports other configured
authentication. The adapter does not copy credentials. Check runtime
authentication if you intend to use a subscription rather than API billing.

The class and constructor work without the extra; the first SDK operation
needs `claude-agent-sdk` and `mcp`. Installation and operating-system
requirements belong to the upstream docs, rather than a frozen npm/Node guide.

## Basic usage and tools

```python
from lazybridge import Agent, ClaudeCodeEngine

def project_status() -> str:
    """Return a deterministic project status."""
    return "The draft is ready for review."

agent = Agent(name="assistant", engine=ClaudeCodeEngine(system="Be concise."),
              tools=[project_status])
result = agent("Read project_status and report the next step.")
print(result.text())
```

LazyBridge normalizes functions, `Tool`, `ToolProvider.as_tools()` and named
child agents. The engine exposes them through a temporary in-process MCP
adapter, without a resident server or manual MCP registration. Tool handlers
call `Tool.run()`, preserving schema validation and tool-error reporting.
Names must be unique. Native built-ins are configured separately.

## Constructor options

| Parameter | Default | Meaning |
| --- | --- | --- |
| `model` | `"sonnet"` | Runtime model alias or full ID. `opus` / `haiku` availability depends on the runtime/account. |
| `cwd`, `system` | `None` | Working directory and system instructions. |
| `max_turns` | `20` | Native agentic-turn cap. |
| `file_roots` | `None` | File-tool roots; omitted with cwd means cwd alone. |
| `web` | `True` | Exposes WebSearch and WebFetch. |
| `reasoning_effort` | `None` | Validated: low, medium, high, xhigh, max. Runtime/model support may differ. |
| `thinking` | `None` | adaptive, disabled, or positive integer token budget. |
| `fallback_model` | `None` | SDK model fallback; distinct from Agent fallback. |
| `session_mode` | `"memory"` | memory or runtime; see continuity below. |
| `session_name` | `None` | Runtime-mode slot label, otherwise agent name. It is not a session title. |
| `session_id`, `persist_session` | `None`, `False` | Resume native ID; keep native history across processes. An ID implies persistence. |
| `session_alias`, `session_registry` | `None` | Named durable binding; alias implies persistence. |
| `tag` | `"lazybridge"` | Best-effort SDK tag on a newly created durable session; None disables. |
| `config` | default CodingAgentConfig | Native policy plus shared approval gate. |
| `approval_gate` | `None` | Direct gate override; a different gate in config conflicts. |
| `request_timeout` | `120.0` | Whole engine run, retries included; None disables. |
| `stream_idle_timeout` | `90.0` | Maximum idle gap during streaming; None disables. |
| `max_retries`, `retry_delay` | `3`, `1.0` | Transient retry budget and exponential-backoff base with jitter. |
| `tool_timeout` | `None` | Per-application-tool deadline. |
| `client` | AgentSdkClient | Injection boundary for offline transport tests. |

Positive deadlines/delay and nonnegative retries are checked at construction.
Configure these on the engine when passing `Agent(engine=...)`.

## Native reads, writes and approvals

Default built-ins are WebSearch/WebFetch and, when roots are available,
Read/Glob/Grep. There is no default Write/Edit/Bash. A writer must grant file
tools as well as supply an approval policy:

```python
from pathlib import Path
from lazybridge import Agent, ClaudeCodeEngine, ClaudeCodePolicy, CodingAgentConfig, TerminalApprovalGate

repo = str(Path.cwd().resolve())
gate = TerminalApprovalGate()
config = CodingAgentConfig(
    claude=ClaudeCodePolicy(
        permission_mode="default",
        preapprove_application_tools=False,
        extra_tools=("Write", "Edit", "Bash"),
    ),
    approval_gate=gate,
)
writer = Agent(name="writer", engine=ClaudeCodeEngine(
    cwd=repo, file_roots=[repo], config=config, request_timeout=600,
))
```

`CodingAgentConfig.writer(gate)` configures approvals but does not add Claude
Write/Edit/Bash. `extra_tools` exposes them; `allowed_tools` preapproves them.
File roots are enforced by hooks on Read/Glob/Grep/Write/Edit/NotebookEdit,
including under permissive modes. Bash can reach any path its process can:
it requires a gate at construction, and file roots do not confine it.
Application tools are preapproved by default; set
`preapprove_application_tools=False` to send them through the gate too.

See [Coding-agent permissions](coding-agent-config.md) for profiles, grant
scope, settings inheritance and compaction. A reviewer profile is read-only
but does not make arbitrary Python tools safe; it denies them unless approved.

## Memory, native sessions and aliases

In default memory mode, each ordinary call is primed from LazyBridge Memory.
After a successful answer, Memory records the task/response. Persistence changes
the history authority: on native resume, the engine does not prepend Memory
again. Explicit Envelope context and sources continue to flow on every turn.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from lazybridge import Agent, ClaudeCodeEngine, SessionRegistry

repo = str(Path.cwd().resolve())
# Temporary registry for this example; use a fixed file for restart continuity.
with TemporaryDirectory() as directory:
    registry = SessionRegistry(path=Path(directory) / "sessions.json")
    engine = ClaudeCodeEngine(cwd=repo, session_alias="writer",
                              session_registry=registry, tag=None)
    Agent(name="writer", engine=engine)("Remember the word PINEAPPLE.")
    handle = engine.session_id
    assert handle is not None
    resumed = ClaudeCodeEngine(cwd=repo, session_id=handle, tag=None)
    print(Agent(name="writer", engine=resumed)("What was the word?").text())
```

For production, omit the custom registry or choose a persistent file. A later
engine with the same alias, kind and resolved cwd resumes the stored ID.
An explicit ID takes precedence and rebinds the alias; failures/timeouts can
still leave a binding once the SDK reports an ID.

`session_mode="runtime"` instead parks the ID on the supplied LazyBridge
Session under `claude-code:{session_name or agent_name}`. A new Session starts
without that handle; this alone is not a cross-process registry. An explicit
session_id wins over the parked ID. `session_name` names the slot; `tag` is
identification metadata. SDK tagging failure warns rather than failing a turn.

Durable runs and streams serialize in-process by engine/alias/native ID,
including first turns and stream teardown. Runtime-mode channels also lock
within an event loop. There is no cross-process transcript lock: coordinate
callers yourself. See [SessionRegistry](session-registry.md).

## Streaming, structured output and attachments

`agent.stream(task)` forwards text chunks and records completed text in Memory.
Session events cover application and native tool calls and model usage; the
Envelope provider is `claude-code`. SDK cost is reported usage, not a promise
about subscription charges.

```python
from pydantic import BaseModel
from lazybridge import Agent, ClaudeCodeEngine

class Status(BaseModel):
    ready: bool
    reason: str

agent = Agent(name="status", engine=ClaudeCodeEngine(), output=Status)
result = agent("Return ready=true and reason='review complete'.")
assert isinstance(result.payload, Status)
print(result.payload.reason)
```

Derivable schemas go to the SDK's native `output_format`. Agent still validates
the returned payload and can repair parse failures. Non-derivable schemas fall
back to post-hoc parsing; the default string output sets no format.

Local/byte images are forwarded as inline Anthropic image blocks. URL-only
images are dropped with a warning, and audio is not forwarded. Generated
media is not returned as a media artifact by this engine.

## Usage budget and failures

`await engine.usage()` and `fetch_claude_usage(model=..., cwd=..., client=...)`
send the CLI's /usage command and parse prose into ClaudeUsageSnapshot.
Unlike Codex's rate-limit RPC, this invokes the SDK runtime. Fields that cannot
be parsed are None; inspect parsed/raw_text and treat missing telemetry as
unknown. Session and weekly windows carry percentages and reset timestamps.
Unknown timezone text leaves reset time unknown. On Windows the extra includes
tzdata. `parse_usage_report(text)` works offline.

Transient network/timeout/429/5xx failures are retried within the run deadline.
Missing or unreadable executables are permanent configuration failures.
Long repository tasks need an explicit deadline; an MCP host may have a
separate timeout. Inspect errors and any stored native handle before resuming.

## Verification and next steps

```sh
python -m pytest tests/unit/engines/claude_code/ -q
```

Tests use fake transports. The live scripts in examples/claude_code require a
real local login and consume runtime resources; they were not run for this docs
refresh.

- [Claude Code writes, Codex reviews](claude-writes-codex-reviews.md)
- [Codex Engine](codex-engine.md) and [coding agents](coding-agents.md)
- [LazyTools MCP/code support](https://tools.lazybridge.com/code-support/)
- [LazyTools repository code bridge](https://tools.lazybridge.com/code-bridge/)
