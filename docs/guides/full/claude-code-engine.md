# Claude Code Engine

`ClaudeCodeEngine` runs the model/tool loop through the locally authenticated
Claude Code runtime via the Claude Agent SDK. It satisfies the same
[Engine protocol](../advanced/engine-protocol.md) as Codex, provider APIs and
Plan. Tools and application state remain LazyBridge objects.

## Setup

Use Python 3.11 or later and install the extra:

```sh
python -m pip install "lazybridge[claude-code]==1.8.2"
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

### Tool-call lifecycle

```text
typed callable / ToolProvider / named child Agent
  → normalized Tool catalogue
  → temporary SDK MCP definitions
  → model selects a tool with structured arguments
  → Tool.run validates and executes
  → result/error goes back to the runtime and optional Session events
```

The runtime can reason about a tool error and try another action. That is not
automatic success: inspect the returned Envelope and your acceptance checks.
Providers must expose as_tools(); reusable child agents need unique names.
You can mix a callable with a specialist without a custom MCP configuration:

```python
from lazybridge import Agent, ClaudeCodeEngine, CodexEngine

def get_quote(symbol: str) -> dict[str, str]:
    """Return a deterministic demonstration quote, not live market data."""
    return {"symbol": symbol, "price": "123.45"}

specialist = Agent(name="quote-review", engine=CodexEngine(
    system="Explain quote inputs and limitations; do not trade.",
))
desk = Agent(name="desk", engine=ClaudeCodeEngine(), tools=[get_quote, specialist])
print(desk("Explain the demonstration quote for EXAMPLE.").text())
```

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

## Claude Code model identifiers

The engine passes model strings to the installed runtime. Its default is
sonnet; runtime and account support determine what can be selected.

| Value | Meaning | How to use it |
| --- | --- | --- |
| sonnet | Runtime Sonnet family alias | Core's default; resolution changes over time |
| opus | Runtime Opus family alias | Verify availability on the target account/provider |
| haiku | Runtime Haiku family alias | Verify availability; no fixed speed or price promise |
| A full Claude model ID | Pin a runtime-supported model version | Verify the exact ID locally; aliases are not version pins |
| inherit | Parent-model selection in SDK sub-agent definitions | Not a recommended top-level engine model |

At the time of writing, see the official
[model configuration](https://code.claude.com/docs/en/model-config) and
[SDK sub-agent documentation](https://platform.claude.com/docs/en/agent-sdk/subagents).
The public CLI docs list more settings than the aliases above; this table is
not an exhaustive runtime catalogue. Do not infer IDs such as sonnet-5 from a
family name. Native runtime aliases are distinct from LLMEngine catalogue IDs
and Agent.from_provider tier aliases.

```python
from lazybridge import ClaudeCodeEngine

default = ClaudeCodeEngine(model="sonnet")
alternative = ClaudeCodeEngine(model="opus", fallback_model="sonnet")
reasoned = ClaudeCodeEngine(reasoning_effort="high", thinking="adaptive")
budgeted = ClaudeCodeEngine(thinking=8_000)
```

fallback_model is a native SDK fallback, not the outer Agent fallback. Claude
validates effort values and thinking shapes at construction; acceptance by a
particular model still depends on the installed runtime.

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

### Runtime session mode

Use runtime mode for a conversation whose native handle is kept on one
LazyBridge Session. It is useful when several Agent objects share a labelled
runtime channel in the same process:

```python
from lazybridge import Agent, ClaudeCodeEngine, Memory, Session

session, memory = Session(), Memory()
engine = ClaudeCodeEngine(session_mode="runtime", session_name="research", tag=None)
researcher = Agent(name="researcher", engine=engine, session=session, memory=memory)
first = researcher("Remember the topic: deployment policy.")
second = researcher("Explain the topic with an example.")
assert first.ok and second.ok
```

Memory still records successful answers for application use and explicit
handoffs such as from_memory. Envelope.context and Agent.sources flow on every
turn. File roots and cwd remain permission inputs, not conversation identities.

`session_mode="runtime"` parks the ID on the supplied LazyBridge
Session under `claude-code:{session_name or agent_name}`. A new Session starts
without that handle; this alone is not a cross-process registry. An explicit
session_id wins over the parked ID. `session_name` names the slot; `tag` is
identification metadata. SDK tagging failure warns rather than failing a turn.

Durable runs and streams serialize in-process by engine/alias/native ID,
including first turns and stream teardown. Runtime-mode channels also lock
within an event loop. There is no cross-process transcript lock: coordinate
callers yourself. See [SessionRegistry](session-registry.md).

### Distinguishing LazyBridge sessions on disk

Claude tagging is post-hoc identification, unlike Codex's creation-time
threadSource. The engine calls the SDK tag_session API once when it creates
a durable session. The SDK listing returns the label as .tag. An alias implies
persistence too; an ephemeral session is not tagged. A tag is neither a
SessionRegistry alias nor an authorization rule.

```python
from lazybridge import ClaudeCodeEngine

default_label = ClaudeCodeEngine(persist_session=True)
app_label = ClaudeCodeEngine(persist_session=True, tag="my-app")
without_label = ClaudeCodeEngine(persist_session=True, tag=None)
```

With the SDK extra installed, this read-only listing identifies candidates for
a retention review. It does not delete transcripts:

```python
try:
    from claude_agent_sdk import list_sessions
except ImportError:
    print("Install lazybridge[claude-code] to list native sessions")
else:
    mine = [item for item in list_sessions() if item.tag == "lazybridge"]
    for item in mine:
        print(item.session_id, item.tag)
```

Before cleanup, confirm ownership, cwd and active use. Removing an alias only
removes the registry binding; it does not delete native history. Native cleanup
uses the installed SDK's session APIs, such as delete_session, after selecting
specific IDs. Custom labels can separate application retention policies.
Tagging failures warn and do not invalidate a completed answer. The official
[session-browser walkthrough](https://platform.claude.com/cookbook/claude-agent-sdk-05-building-a-session-browser)
documents listing, tagging and session-management operations.

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

### Multimodal input

```python
import base64
from lazybridge import Agent, ClaudeCodeEngine, ImageContent

png = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC"
)
image = ImageContent.from_bytes(png, media_type="image/png")
viewer = Agent(name="viewer", engine=ClaudeCodeEngine())
result = viewer("Describe this tiny test image.", images=[image])
assert result.ok
```

Local paths also work: LazyBridge reads and inlines their contents. With images,
the SDK adapter uses an async user-message stream containing text and image
blocks instead of a plain string prompt. Local/byte images are forwarded as inline Anthropic image blocks. URL-only
images are dropped with a warning, and audio is not forwarded. Generated
media is not returned as a media artifact by this engine.

## Reading the account usage budget

`await engine.usage()` and `fetch_claude_usage(model=..., cwd=..., client=...)`
send the CLI's /usage command and parse prose into ClaudeUsageSnapshot.
Unlike Codex's rate-limit RPC, this invokes the SDK runtime. Fields that cannot
be parsed are None; inspect parsed/raw_text and treat missing telemetry as
unknown. Session and weekly windows carry percentages and reset timestamps.
Unknown timezone text leaves reset time unknown. On Windows the extra includes
tzdata. `parse_usage_report(text)` works offline.

| Snapshot field | Meaning |
| --- | --- |
| session | Parsed current-session window, or None |
| weekly | Windows keyed by the CLI's labels; the set is not fixed |
| most_used() | Highest-used weekly window, or None |
| parsed / raw_text | Whether a window matched; untouched report for diagnosis |
| fetched_at | Time the snapshot was captured |

Each UsageWindow has used_percent, resets_raw and resets_at. The parser
resolves a yearless date relative to an aware reference time, including year
rollover. Unknown zones and ambiguous DST times stay unparsed. A successful
but unparseable report differs from runtime/authentication failure, which raises
RuntimeError when fetching. CLI wording can change; do not treat None as zero.

```python
from datetime import datetime, timezone
from lazybridge.engines.claude_code.usage import parse_usage_report

snapshot = parse_usage_report(
    "Current week (all models): 25% used - resets Oct 18, 12pm (UTC)",
    now=datetime(2026, 10, 11, tzinfo=timezone.utc),
)
assert snapshot.parsed
label, window = snapshot.most_used()
assert label == "all models" and window.used_percent == 25
print(window.resets_at)
```

This parser example makes no SDK call. Live engine.usage() uses the SDK runtime
to send /usage; it is not Codex's model-free account RPC and has no promised
latency. Token counts from ordinary runs can be inspected through
Session.usage_summary() independently of quota parsing.

## Failures and troubleshooting

Transient network/timeout/429/5xx failures are retried within the run deadline.
Missing or unreadable executables are permanent configuration failures.
Long repository tasks need an explicit deadline; an MCP host may have a
separate timeout. Inspect errors and any stored native handle before resuming.

| Symptom | Check | Resolution |
| --- | --- | --- |
| claude cannot be found | PATH of the Python process and installed runtime | Follow official setup; restart the process after PATH changes |
| Sign-in requested | Interactive CLI login in the same environment | Complete authentication and repeat a minimal smoke check |
| File access denied | Resolved path and file_roots/cwd | Grant the intended directory; file roots do not confine Bash |
| Tool missing | Agent tools, provider as_tools and unique names | Fix catalogue construction before changing native policy |
| Tool failed | Signature/schema and Session tool events | Inspect the error and tool implementation |
| Write/Edit unavailable | extra_tools versus allowed_tools | Expose the built-in and configure approval separately |
| Bash rejected at construction | Unconfined built-in without a gate | Supply an explicit gate policy |
| Quota parsing empty | parsed and raw_text | Treat as unknown; diagnose wording rather than invent capacity |
| Long run times out | Whole engine and host deadlines | Set a deliberate bound and inspect learned native ID before resume |

## What this integration intentionally does not do

- Replace LazyBridge as the application orchestrator.
- Run a resident/external MCP server for its in-process Python Tool adapter.
- Grant Write/Edit/Bash by default, or expose undeclared application tools.
- Copy CLI credentials or determine the account's billing path.
- Create a second LazyBridge-owned native transcript database. Durable history
  belongs to Claude's runtime; Memory and Session are separate application state.
- Migrate native history to Codex, enforce cross-process transcript locking,
  or turn reviewer feedback into release acceptance automatically.
- Return generated media artifacts through this text-based adapter.

## Verification and next steps

```sh
python -m pytest tests/unit/engines/claude_code/ -q
```

Tests use fake transports. With an authenticated local runtime, these optional
smoke checks isolate the adapter from a complete Agent call:

```sh
python examples/claude_code/live_mcp_smoke.py
python examples/claude_code/live_engine_smoke.py
```

The first checks the in-process MCP bridge; the second exercises the full Agent.
Start with claude doctor and a minimal CLI turn before investigating tool wiring.
Structured-output and multimodal live examples are also in the same directory.
They consume runtime resources and were not run for this documentation refresh.

- [Claude Code writes, Codex reviews](claude-writes-codex-reviews.md)
- [Codex Engine](codex-engine.md) and [coding agents](coding-agents.md)
- [LazyTools MCP/code support](https://tools.lazybridge.com/code-support/)
- [LazyTools repository code bridge](https://tools.lazybridge.com/code-bridge/)
