# Codex Engine

`CodexEngine` drives a LazyBridge Agent through `codex app-server` over JSON-RPC
on subprocess pipes. It uses the local runtime's authentication, exposes Python
tools as dynamic tools and returns an Envelope with provider `codex`.
No resident server or network port is needed.

## Setup

Install and authenticate the CLI using the
[official Codex CLI instructions](https://developers.openai.com/codex/cli/).
Run `codex` and follow its sign-in flow, or use `codex login`.
A subscription login uses plan quota; another configured authentication can
use a different billing path. The adapter does not copy credentials.

Python 3.11 or later and the core package are sufficient; there is no Codex
Python SDK extra:

```sh
python -m pip install "lazybridge==1.8.5"
```

Executable lookup checks CODEX_BIN, PATH and the Windows desktop app's versioned
install directories. Candidate selection prefers the newest stable CLI by
version and excludes helpers; set CODEX_BIN when you need a particular binary.

## Basic usage

```python
from lazybridge import Agent, CodexEngine

def project_status() -> str:
    """Return a deterministic project status."""
    return "The draft is ready for review."

agent = Agent(name="assistant", engine=CodexEngine(system="Be concise."),
              tools=[project_status])
print(agent("Read project_status and report the next step.").text())
```

Functions, ToolProvider instances and named child agents use the same
`tools=[...]` contract as Claude Code. Dynamic tool callbacks dispatch through
Tool.run, independently of runtime MCP configuration. A native read-only sandbox
does not sandbox Python application tools; review their side effects separately.

## Why App Server rather than codex exec

The adapter needs bidirectional application-tool callbacks, native thread
control and read-only sandbox operation. It therefore uses App Server's
JSON-RPC interface. It does not need a flag that bypasses both approvals and
sandboxing to expose Python tools.

Early integration experiments with codex exec had non-interactive MCP approval
limitations. That historical result is not a claim that every current exec
configuration rejects MCP. For current protocol capabilities, see the official
[App Server documentation](https://developers.openai.com/codex/app-server).

## Constructor options

| Parameter | Default | Meaning |
| --- | --- | --- |
| `model` | `None` | Runtime-configured model, or explicit runtime ID. No model-list validation. |
| `cwd`, `system` | `None` | Working directory; system becomes developer instructions. |
| `reasoning_effort` | `None` | Passed through; supported values depend on model/list. No thinking parameter. |
| `thread_id`, `persist_thread` | `None`, `False` | Native resume handle and durable transcript; ID implies persistence. |
| `session_alias`, `session_registry` | `None` | Named durable binding; alias implies persistence. |
| `review_target` | `None` | Native review/start instead of an ordinary prompted turn. |
| `thread_source` | `"lazybridge"` | Creation-time label; None omits it. |
| `config`, `approval_gate` | default config, `None` | Native sandbox/policy and shared gate. Different direct/config gates conflict. |
| `request_timeout`, `stream_idle_timeout` | `120.0`, `90.0` | Whole run including retries; streaming idle gap. None disables either. |
| `max_retries`, `retry_delay` | `3`, `1.0` | Transient budget and exponential-backoff base with jitter. |
| `tool_timeout` | `None` | Per-application-tool deadline. |
| `client` | CodexAppServerClient | Fake transport injection for offline tests. |

Deadlines/delay must be positive, retries nonnegative. See
[Coding-agent permissions](coding-agent-config.md) for writable_roots,
web_search, preapproval, native approval modes and compaction.

## Durable threads and aliases

Default calls start ephemeral threads. An ephemeral ID cannot be resumed even
if its process just exited. Use persistence before the first turn:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from lazybridge import Agent, CodexEngine, SessionRegistry

repo = str(Path.cwd().resolve())
# Temporary registry for the demo; use a fixed file across real restarts.
with TemporaryDirectory() as directory:
    registry = SessionRegistry(path=Path(directory) / "sessions.json")
    engine = CodexEngine(cwd=repo, session_alias="reviewer", session_registry=registry)
    Agent(name="reviewer", engine=engine)("Remember the word PINEAPPLE.")
    handle = engine.thread_id
    assert handle is not None
    resumed = CodexEngine(cwd=repo, thread_id=handle)
    print(Agent(name="reviewer", engine=resumed)("What was the word?").text())
```

Without aliases, `persist_thread=True` exposes engine.thread_id for storage.
A later engine with the same alias/kind/cwd resolves the ID; an explicit ID wins
and rebinds it. Registry updates preserve handles learned on failed turns.

On resume, native history replaces Memory priming. Memory keeps recording
completed answers for application use; explicit context/sources still flow.
Runs and streams queue by engine, alias and thread inside an event loop.
There is no cross-process transcript lock: coordinate access outside the engine.
The new subprocess re-registers dynamic tools on resume.

## Repository writer and native review

```python
from pathlib import Path
from lazybridge import Agent, CodexEngine, CodingAgentConfig, TerminalApprovalGate

repo = str(Path.cwd().resolve())
writer = Agent(name="writer", engine=CodexEngine(
    cwd=repo, config=CodingAgentConfig.writer(TerminalApprovalGate()),
    request_timeout=600,
))
reviewer = Agent(name="reviewer", engine=CodexEngine(
    cwd=repo, config=CodingAgentConfig.reviewer(),
    review_target={"type": "uncommittedChanges"}, persist_thread=True,
))
```

The writer uses workspace-write/on-request. Git worktree index/refs can live
outside cwd; the caller must resolve any required common Git directory and
explicitly permit it with CodexPolicy.writable_roots.

Native review targets are uncommittedChanges, baseBranch with branch, or commit
with sha. `review_target` selects the native harness on every call; the ordinary
Agent task is not sent because review/start has no prompt field. For a steered
follow-up construct another engine with the recorded thread_id and no review_target.
Use an ordinary prompted Agent as a verify judge, rather than native review/start.

## Protocol notes

The transport implements a subprocess protocol, not an HTTP service. Its
request/notification fixtures cover the wire shapes used by this source version.
Use the installed CLI's schema generator when diagnosing version differences;
experimental dynamic-tool support can change between runtimes.

```text
initialize → initialized
thread/start (or thread/resume, re-registering dynamic tools)
turn/start (or review/start for a native review target)
  ↔ application-tool / native approval requests and progress notifications
turn/completed → collect text and attributable usage → close subprocess
```

- Native sandbox enum values are read-only, workspace-write and
  danger-full-access. The camel-case spelling readOnly is not equivalent.
- An ephemeral thread is unresumable. Persistence must be selected before the
  first turn; keeping its printed ID alone does not make it durable.
- Dynamic tools are registered again on resume because Python handlers belong
  to the new process, not the saved transcript.
- thread/tokenUsage/updated totals are cumulative across the thread. The client
  subtracts the pre-turn baseline rather than using only the final model call.
- Turn IDs distinguish new notifications from replayed completions. An older
  turn/completed event must not finish the current request.
- Server-to-client request IDs are independent of client IDs. The read loop
  distinguishes a request by its method field, not just its ID.
- RPC stdout is reserved for the protocol. This source accepts individual lines
  up to 64 MiB and retains the newest 16 KiB of stderr, quoting a bounded excerpt
  in failures. The limit applies to one message, not the full turn.

These are adapter behaviors and compatibility considerations. Validate dynamic
tool fields against the installed runtime rather than assuming one historical
generated schema defines a permanent protocol contract.

## Structured output and attachments

```python
from pydantic import BaseModel
from lazybridge import Agent, CodexEngine

class Status(BaseModel):
    ready: bool
    reason: str

agent = Agent(name="status", engine=CodexEngine(), output=Status)
result = agent("Return ready=true and reason='review complete'.")
assert isinstance(result.payload, Status)
print(result.payload.reason)
```

Object-rooted schemas that can preserve semantics under OpenAI's strict subset
go to native turn/start outputSchema. Objects become closed and all properties
required; nullable optional fields can remain representable. Nonnullable defaults,
open objects, fixed tuples or a non-object root fall back to prompt instructions
and Agent parsing/repair. Default str/Any output sets no schema.

### Multimodal input

```python
import base64
from lazybridge import Agent, CodexEngine, ImageContent

png = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC"
)
image = ImageContent.from_bytes(png, media_type="image/png")
viewer = Agent(name="viewer", engine=CodexEngine())
result = viewer("Describe this tiny test image.", images=[image])
assert result.ok
```

Images use URL/data UserInput items; local paths are inlined by LazyBridge.
URL-only inputs pass through rather than being fetched by the adapter. The
presence of audio or image-generation variants in a protocol does not imply
this text-turn engine supports them as input/output artifacts.
Audio is dropped with a UserWarning. Generated images/audio are not returned
as artifacts by this text-based adapter. Streaming forwards text through the
client callback and commits completed text to Memory.

## Usage accounting

Token notifications are cumulative over the thread; the client subtracts the
pre-turn baseline and attributes notifications by turn ID, ignoring replayed
completion events for older turns. Codex cost_usd is 0.0 in this adapter because
the App Server reports no per-turn price; it is not evidence that a run is free.

## Distinguishing LazyBridge threads on disk

```python
from lazybridge import CodexEngine

default_label = CodexEngine()
app_label = CodexEngine(thread_source="my-app")
without_label = CodexEngine(thread_source=None)
```

thread_source is sent as threadSource only on creation. It lands in
`session_meta.payload.thread_source`, not payload.source. The initialize
handshake separately stamps originator=lazybridge. Labels help identify
rollouts; they do not create cleanup or access-control policy. Changing a
resuming engine's thread_source does not relabel existing history or name the
runtime's resume-picker entry.

For a retention review, inspect session_meta records in the runtime's sessions
directory (normally under ~/.codex, or the configured runtime home). Match
payload.thread_source and, where present, payload.originator. Do not filter the
older guide's payload.source spelling or assume an undocumented delete command.
Check active use and selected IDs before using the installed runtime's cleanup
mechanism. Removing SessionRegistry bindings does not delete rollouts.

This helper parses a record without opening or altering native history:

```python
import json

def is_lazybridge_rollout(line: str) -> bool:
    record = json.loads(line)
    payload = record.get("payload", {})
    return record.get("type") == "session_meta" and (
        payload.get("thread_source") == "lazybridge"
        or payload.get("originator") == "lazybridge"
    )

sample = json.dumps({"type": "session_meta", "payload": {"thread_source": "lazybridge"}})
assert is_lazybridge_rollout(sample)
assert not is_lazybridge_rollout('{"type": "session_meta", "payload": {"source": "interactive"}}')
```

## Reading the subscription quota

```python
import asyncio
from lazybridge.engines.codex.usage import fetch_codex_usage

async def show_quota():
    snapshot = await fetch_codex_usage()
    for window in snapshot.windows:
        print(window.limit_id, window.kind, window.used_percent, window.resets_at)
    return snapshot

snapshot = asyncio.run(show_quota())
```

The function reads account/rateLimits/read without a thread or model turn.
Each is a `CodexUsageWindow` with `limit_id`, `kind`, `used_percent`, `window_duration_minutes`, `resets_at`, and `reached_type`.
Snapshot weekly selects the seven-day bucket by duration, not array position.
resets_at is a UTC datetime or None. Missing executable raises
FileNotFoundError; RPC failures raise RuntimeError. Unknown/missing telemetry
must not be treated as spare quota. snapshot.plan_type and snapshot.raw retain
account details not modeled as windows. Percentage and reset time should be
considered together; no fixed percentage establishes sufficient capacity for
a job. The standalone fetch uses a 30-second default RPC timeout after startup.
Quota routing and its benchmark belong to LazyTools; the benchmark is in progress.

## Transport failures and retry boundaries

CodexTransportError reports phase, PID, return code and a bounded stderr tail.
The client retries one startup failure and, on this source baseline, one
transport failure during thread/start or thread/resume before a turn was sent.
A possibly created but unused thread may remain. The engine does not blindly
replay later stateful failures.

On a durable thread, loss after turn/start was sent can raise CodexTurnUncertain:
the turn might already have executed tools. Resume and inspect before deciding
what to do; automatic replay can duplicate side effects. A known rejection
differs from an uncertain accepted turn. Missing/unreadable executables are
permanent errors, not transient retries.

The default 120-second deadline covers all retries. Use a larger bound for
repository work and align any MCP host deadline. No model validation or
cross-process thread locking is implemented.

```sh
python -m pytest tests/unit/engines/codex/ -q
```

## Troubleshooting and verification

| Symptom | Check | Resolution |
| --- | --- | --- |
| Executable missing | CODEX_BIN/PATH and discovered candidates | Set a known CLI path or follow official installation |
| Startup/RPC failure | Transport phase, return code and stderr tail | Check login/version before retrying model work |
| Invalid model or effort | Runtime-configured model and model/list | Use an advertised ID/effort; the engine does not validate availability |
| Structured output falls back | Strict-schema representability | Keep Agent parsing/validators; do not assume every schema is native |
| Cannot resume | Persistence on the first turn and correct cwd/ID | Start durable work deliberately; an ephemeral ID cannot be repaired by resume |
| Native write denied | Sandbox, approval policy and writable_roots | Configure the intended workspace and Git metadata roots explicitly |
| Python tool denied | preapprove_dynamic_tools and gate | Configure approval independently of the filesystem sandbox |
| Long repository run cancelled | Engine deadline and host timeout | Align both limits and inspect any durable turn before replay |

Optional authenticated smoke checks:

```sh
python examples/codex/live_app_server_smoke.py
python examples/codex/live_engine_smoke.py
```

The first isolates the protocol; the second exercises Agent composition. The
multimodal/composition scripts in the same directory cover additional paths.

## What this integration intentionally does not do

- Validate a model or effort against model/list at construction.
- Lock one native thread across independent Python processes.
- Replay a possibly accepted side-effecting turn automatically.
- Add Claude-style file/web Tool objects; Codex uses its own native sandbox
  and per-process web_search policy instead.
- Migrate native history across engines or create a second transcript store.
- Infer Git worktree metadata roots or authorize undeclared Python side effects.
- Return generated image/audio artifacts or a per-turn dollar price absent from RPC.

These tests use fake transports. Live scripts in examples/codex require an
authenticated runtime and were not run for this docs refresh.

See [coding agents](coding-agents.md),
[Claude Code writes, Codex reviews](claude-writes-codex-reviews.md) and the
[LazyTools code bridge](https://tools.lazybridge.com/code-bridge/) for connected
repository work and durable results.
