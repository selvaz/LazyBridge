# Claude Code Engine

`ClaudeCodeEngine` is a standard LazyBridge [`Engine`](../advanced/engine-protocol.md)
that runs the model/tool loop through the locally installed **Claude Code**
runtime (via the Claude Agent SDK) instead of a raw provider API call. The
application remains a normal `lazybridge.Agent` — Claude Code is only the
engine that drives it.

## Scope and design

The integration is designed to use a Claude.ai account with a Claude Code
compatible plan (for example, Pro or Max), without putting an API key in the
project. It does not create a persistent service, open ports, or store Claude
credentials or separate Claude conversations.

LazyBridge owns conversation memory. LazyBridge owns tools. For one agent run,
the engine creates a temporary **in-process** MCP adapter: it only lets Claude
invoke tools that LazyBridge has already normalised.

## 1. First-time Claude Code setup and sign-in

### Prerequisites

- A Claude.ai account with a plan that includes Claude Code.
- Node.js and npm (Anthropic's npm installation documentation requires Node.js
  18 or later).
- Python 3.11 or later.
- An Internet connection for Claude Code sign-in and model processing.

On Windows, Claude Code is supported in WSL or Git Bash. The Python process
using `ClaudeCodeEngine` may run in the normal project environment as long as
it can find the `claude` command.

### Install Claude Code

In a shell supported by Claude Code:

```bash
npm install -g @anthropic-ai/claude-code
claude doctor
```

If a global npm installation does not suit the environment, use one of the
alternative methods in Anthropic's documentation. Do not use `sudo` with the
global npm installation.

### Sign in once

1. Change into any working directory.
2. Start `claude`.
3. At the sign-in screen, choose the **Claude App / Claude.ai** subscription
   option.
4. Complete browser sign-in with the account that owns the Claude plan.
5. Return to the terminal. The interactive session can then be closed if it
   is not otherwise needed.

Claude Code stores its own credentials in local secure storage.
`ClaudeCodeEngine` reuses that login through the Agent SDK: it does not read,
copy, or write tokens into the repository, project environment variables, or
LazyBridge memory.

Quick verification:

```bash
claude -p "Reply exactly: Claude Code ready"
claude doctor
```

If the first command prints the requested response, sign-in is ready. Do not
set `ANTHROPIC_API_KEY` when the explicit goal is to use Claude.ai sign-in; an
API key uses a separate billing path.

Official Anthropic references: [Claude Code setup](https://docs.anthropic.com/en/docs/claude-code/getting-started)
and [CLI reference](https://docs.anthropic.com/en/docs/claude-code/cli-usage).

## 2. Install the extra

`ClaudeCodeEngine` is always importable from `lazybridge`; actually
constructing/running one needs the optional dependencies:

```bash
pip install "lazybridge[claude-code]"
```

This pulls `claude-agent-sdk` and `mcp`. Without the extra installed,
constructing a `ClaudeCodeEngine` still works, but the first call that
touches the SDK raises a clear `ImportError` pointing back to this install
command — see [`sdk_client.AgentSdkClient`](#) in
`lazybridge.engines.claude_code`.

## 3. Basic usage: a standard LazyBridge engine

```python
from lazybridge import Agent, ClaudeCodeEngine


def get_quote(symbol: str) -> dict[str, str]:
    """Return a deterministic quote for the requested symbol."""
    return {"symbol": symbol, "price": "123.45"}


agent = Agent(
    name="research",
    engine=ClaudeCodeEngine(model="sonnet", system="Be concise."),
    tools=[get_quote],
)

result = agent("Find AMZN's quote using the available tool.")
print(result.text())
```

Nothing changes for a caller compared with a standard LazyBridge engine:

- `Agent`, `Memory`, `Session`, and `stream()` remain LazyBridge concepts.
- `result` is a normal LazyBridge `Envelope` with `provider="claude-code"`.
- Functions, `Tool` objects, `ToolProvider` instances, and other `Agent`
  instances are passed through `tools=[...]`.
- There is no application MCP registration to perform manually.

## 4. Tool lifecycle

```text
tools=[function | ToolProvider | Agent]
                |
                v
LazyBridge normalises/expands them into Tool objects
                |
                v
ClaudeCodeEngine creates an in-process MCP adapter for this run only
                |
                v
Claude Code selects a tool and sends validated arguments
                |
                v
LazyBridge Tool.run(**args) runs the function or child agent
                |
                v
Result or error returns to Claude, then to the LazyBridge Envelope
```

JSON schemas come from LazyBridge `Tool.definition()`. Argument validation
therefore stays in normal `Tool.run()` execution. A tool failure is returned
to Claude as a tool error, allowing the model to correct its arguments or
choose another strategy rather than terminating the entire run immediately.

### Tool providers and child agents

```python
agent = Agent(
    name="coordinator",
    engine=ClaudeCodeEngine(),
    tools=[my_provider, specialist_agent],
)
```

LazyBridge expands `my_provider.as_tools()` before engine execution.
`specialist_agent` becomes a normal tool with a `task` argument; when Claude
calls it, LazyBridge runs that child agent with its usual configuration.

Tool names must be unique — the same `Agent`/`build_tool_map` rule LazyBridge
uses everywhere else.

## 5. Memory, sessions, and streaming

### Memory

LazyBridge `Memory` is the sole conversation memory. Before a run, the engine
builds its prompt from messages already in that memory; after a successful
run, it adds the task and response. It does not use SDK `resume`, persistent
Claude sessions, or a parallel Claude database (unless `session_mode="runtime"`
— see below).

To keep continuity, pass the same `Memory` to the agent. To isolate a run,
use a new `Memory` or no memory.

### Session

If the agent has a `Session`, the engine emits normal LazyBridge events:
agent start/finish, tool call, tool result, and tool error, plus a
`MODEL_RESPONSE` event shaped like `LLMEngine`'s (`provider="claude-code"`,
token/cost usage) so `Session.usage_summary()` and any cost-report tooling
that reads `event_type="model_response"` sees this engine's usage too.
Existing observability consumers do not need Claude Code-specific logic.

### Streaming

`agent.stream(...)` forwards Agent SDK text chunks as a LazyBridge stream. At
the end, the complete text is added to the same `Memory`, just as it is for
`agent(...)`.

## 6. Web and filesystem: default profile

The engine's default profile is intentionally useful but read-only:

| Capability | Default | Enablement | Boundary |
| --- | --- | --- | --- |
| Web search (`WebSearch`) | yes | `web=True` (default) | no implicit application tools |
| Web fetch (`WebFetch`) | yes | `web=True` (default) | same as above |
| File read (`Read`) | only with a root | `cwd=...` or `file_roots=[...]` | declared roots only |
| File discovery (`Glob`) | only with a root | same as above | declared roots only |
| Text search (`Grep`) | only with a root | same as above | declared roots only |
| Shell (`Bash`) | no | not exposed | — |
| File modifications (`Write`, `Edit`) | no | not exposed | — |
| Native Claude Code subagents | no | not exposed | — |

Example with a restricted filesystem:

```python
engine = ClaudeCodeEngine(
    model="sonnet",
    cwd="C:/work/project",
    # Or, without cwd:
    # file_roots=["C:/work/project", "C:/work/reference-docs"],
)
```

When `cwd` is set and `file_roots` is omitted, `cwd` automatically becomes the
only readable root. A path outside the declared roots is denied by the
engine's permission callback. To disable web tools:

```python
ClaudeCodeEngine(web=False)
```

## 7. Engine configuration

```python
ClaudeCodeEngine(
    model="sonnet",                 # supported Claude model alias/name
    cwd="C:/work/project",          # optional working directory
    system="...",                   # optional system instructions
    max_turns=20,                    # Claude agentic-turn limit
    file_roots=["C:/work/project"], # optional read boundaries
    web=True,                        # WebSearch/WebFetch
    reasoning_effort=None,           # "low"/"medium"/"high"/"xhigh"/"max"
    thinking=None,                   # "adaptive" / "disabled" / token budget int
    fallback_model=None,
    session_mode="memory",           # "memory" (default) or "runtime"
    session_name=None,
    request_timeout=120.0,           # per-run deadline; None disables
    stream_idle_timeout=90.0,        # idle gap before TimeoutError in stream()
    max_retries=3,                   # transient-failure retries, exp. backoff + jitter
    retry_delay=1.0,
    tool_timeout=None,               # per-tool asyncio.wait_for deadline
)
```

Retries follow the same "429/5xx/network/timeout" policy `LLMEngine` uses,
at the coarser granularity available to an SDK/CLI-backed engine (pass/fail
on the whole call, not individual HTTP round-trips). A missing/unreadable
`claude` executable (`FileNotFoundError`/`PermissionError`) is treated as a
permanent configuration problem and is **not** retried.

### Claude Code model identifiers

The stable model values accepted by the Claude Code CLI interface are:

| Value | Meaning | Recommendation |
| --- | --- | --- |
| `"sonnet"` | The latest Sonnet model available to the authenticated Claude Code account. | Default. |
| `"opus"` | The latest Opus model available to the authenticated Claude Code account. | Use for harder tasks when the plan permits it. |
| `"haiku"` | The latest Haiku model, where it is enabled by the Claude Code runtime and account. | Fast/low-cost choice; validate on the target account. |
| `"inherit"` | Inherit the model from a parent Claude Code agent definition. | Not useful as this engine's top-level default. |
| A full Claude model ID | A specifically pinned model version accepted by the installed Claude Code runtime. | Use only after verifying it locally. |

The installed Agent SDK explicitly identifies `sonnet`, `opus`, `haiku`, and
`inherit` as model aliases for agent definitions. Anthropic's public CLI
reference currently documents only `sonnet` and `opus`, so `haiku` must be
validated with the target CLI version and subscription before it becomes a
production default. Do not hard-code informal names such as `"sonnet-5"` or
`"sonnet-4.6"` — they are not documented Claude Code aliases.

### Runtime session mode

The default `session_mode="memory"` sends LazyBridge `Memory` as prompt
context on every call. Set `session_mode="runtime"` to make the Claude Code
session itself the conversation source instead:

```python
engine = ClaudeCodeEngine(
    model="sonnet",
    session_mode="runtime",
    session_name="research",  # optional fixed channel inside one LazyBridge Session
    reasoning_effort="high",
    thinking="adaptive",
)
```

The first call in a LazyBridge `Session` starts a clean Claude session and
stores its ID on that `Session`. Later calls with the same engine/channel
resume it. A new LazyBridge `Session` has no stored ID and starts cleanly
again. `Memory` is still updated after each answer so it can be explicitly
supplied to other agents through `from_memory(...)`, but it is not reinjected
into the parent runtime conversation. `Envelope.context`, `Agent.sources`,
`from_agent(...)`, and `from_memory(...)` continue to be passed on every turn.

An explicit `file_roots` list overrides the root inferred from `cwd`. Do not
grant broad directories for convenience — declare only directories the agent
needs to inspect.

## Distinguishing LazyBridge sessions on disk

Unlike `CodexEngine`'s `threadSource` (sent at thread creation, on the wire),
the Agent SDK has no creation-time metadata field. It does have a public,
post-hoc tagging API — `claude_agent_sdk.tag_session(session_id, tag,
directory=...)`, appending a `{"type": "tag", ...}` JSONL line that
`list_sessions()` reads back as `.tag` — and `ClaudeCodeEngine` uses it
automatically:

```python
engine = ClaudeCodeEngine(persist_session=True)                    # tag="lazybridge" (default)
engine = ClaudeCodeEngine(persist_session=True, tag="my-app")      # a caller-specific label
engine = ClaudeCodeEngine(persist_session=True, tag=None)          # skip tagging
```

Tagging fires once, on the run that *creates* a durable session — not on
every resume, since the SDK's "last tag wins" semantics make repeated calls
redundant I/O, not idempotent no-ops. It requires `persist_session=True`
(or an explicit `session_id`); an ephemeral session is never tagged, since
there is nothing durable to tag. A tagging failure raises a `UserWarning`
rather than failing the run — it is identification metadata, not something
correctness should depend on.

Finding and cleaning up tagged sessions later (`list_sessions()` has no
server-side tag filter — filter the returned list):

```python
from claude_agent_sdk import delete_session, list_sessions

mine = [s for s in list_sessions() if s.tag == "lazybridge"]
for s in mine:
    delete_session(s.session_id)
```

## 8. Verification and troubleshooting

```bash
pytest tests/unit/engines/claude_code/ -q
python examples/claude_code/live_mcp_smoke.py
python examples/claude_code/live_engine_smoke.py
```

Unit tests do not use Claude credentials. The two example scripts make real
requests using the local Claude Code login: `live_mcp_smoke.py` verifies the
in-process MCP bridge directly; `live_engine_smoke.py` runs through a
complete `lazybridge.Agent`.

Common issues:

- **`claude` cannot be found**: install Claude Code and ensure it is on the
  `PATH` of the process running Python.
- **Sign-in is requested**: start `claude` interactively and complete
  Claude.ai authentication, then repeat the quick verification command.
- **File access is denied**: add the correct directory to `file_roots` or set
  `cwd` to the desired root. Do not broaden permissions with Bash or bypass
  modes.
- **A tool is not visible**: pass it through `Agent(tools=[...])`, confirm a
  provider exposes `as_tools()`, and check for duplicate names.
- **A tool fails**: the error is available to Claude and in `Session` events;
  check the tool signature, schema, and implementation.

## 9. What this integration intentionally does not do

- It does not replace LazyBridge as the orchestrator.
- It does not create separate persistent Claude memory.
- It does not run an external or resident MCP server.
- It does not grant filesystem write access or shell execution by default.
- It does not expose tools that were not explicitly passed to the agent.

These limits are intentional: they keep the engine replaceable, configuration
predictable, and the security boundary small.

## Structured output

`Agent(output=<type>)` is enforced by the model, not by prompt discipline. The
engine derives the JSON schema and passes it as the Agent SDK's
`output_format` (`{"type": "json_schema", "schema": ...}`, the CLI's
`--json-schema`); the CLI returns the validated object on
`ResultMessage.structured_output`, so `Envelope.payload` is the parsed model.

```python
class Quote(BaseModel):
    symbol: str
    price: float

agent = Agent(name="quoter", engine=ClaudeCodeEngine(model="sonnet"), output=Quote)
```

This is the same server-side guarantee `LLMEngine` gets from
`StructuredOutputConfig`. Verified live (claude_agent_sdk 0.2.128) with a
plain Pydantic schema including an optional field and a nested model
(`$defs`) — no strict-mode rewrite needed, unlike Codex's `turn/start`
`outputSchema`. `output=str` (the default) sets no `output_format`; if the
schema cannot be derived, the run falls back to LazyBridge's post-hoc JSON
parse and retry.

## Multimodal

`images=` is forwarded as Anthropic image content blocks. Because a
plain-string prompt has nowhere to carry an attachment, a run with images
switches to the SDK's async user-message stream and sends `content` as
`[{"type": "text", ...}, {"type": "image", ...}]`.

```python
agent("What is in this chart?", images=["C:/work/chart.png"])
```

Inline bytes only: the CLI accepts a `base64` source but rejects a `url` one,
so URL-only images are dropped with a `UserWarning` naming the URL rather than
being fetched behind the caller's back — pass a path or bytes and LazyBridge
inlines them. `audio=` is never forwarded, since Claude accepts no audio
input; it is dropped with a warning too.

## See also: Codex Engine

[`CodexEngine`](codex-engine.md) is the same `Engine` contract backed by the
**Codex App Server** (`codex app-server` over JSON-RPC, not `codex exec`).
Both engines compose identically with LazyBridge — the differences are
documented in that guide: Codex reports no dollar cost, primes structured
output in the prompt rather than enforcing it, and has no persistent-thread
equivalent of `session_mode="runtime"`.
