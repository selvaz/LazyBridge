# Codex Engine

`CodexEngine` is a standard LazyBridge [`Engine`](../advanced/engine-protocol.md)
that runs the model/tool loop through the locally installed **Codex** CLI
instead of a raw provider API call. It is the OpenAI-side counterpart of
[`ClaudeCodeEngine`](claude-code-engine.md): the application stays a normal
`lazybridge.Agent`, and Codex is only the engine that drives it.

## Scope and design

The integration uses an authenticated ChatGPT/Codex account without putting an
API key in the project. It starts no persistent service, opens no port, and
copies no credential.

LazyBridge keeps owning conversation memory and tools. For one agent run the
engine starts **one ephemeral, read-only, approval-free thread** and exposes
the current LazyBridge tool list to it as App Server *dynamic tools*; every
call comes straight back to `Tool.run()`.

## Durable threads

`persist_thread=True` makes the thread outlive the subprocess, and
`engine.thread_id` is then the handle another process can pick up:

```python
engine = CodexEngine(cwd=repo, persist_thread=True)
Agent(engine, name="reviewer")("review src/parser.py")
handle = engine.thread_id            # store it

# ...later, another process entirely:
Agent(CodexEngine(cwd=repo, thread_id=handle), name="reviewer")("and the retry path?")
```

The follow-up does not re-read what the first turn already read: Codex' own
transcript carries it. That moves the conversation's home, so the engine also:

- **stops prepending `Memory`** on a resumed thread (Codex has the history;
  sending it again gives the model two chronologies of the same conversation).
  `Memory` keeps recording for the application's own audit/recovery use;
- **refuses to retry a turn lost after the request was sent**, raising
  `CodexTurnUncertain` instead — `turn/start` is not idempotent, and the server
  can accept a turn and then drop the connection *before answering*, so the
  window opens at send, not at acknowledgement. Resume the thread and inspect
  it before deciding;
- **serialises runs per thread id** inside the process. A thread is one
  transcript: two turns appended at once interleave.

Durable threads are stored by the Codex CLI itself (they show up in its session
history), and both processes must share the same Codex home and account.

## Setup

```bash
npm install -g @openai/codex
codex --login
```

The engine does not require `codex` to be on `PATH`: it resolves `CODEX_BIN`
first, then `PATH`, then the Codex desktop app's versioned install directory
(`%LOCALAPPDATA%/OpenAI/Codex/bin/<hash>/codex.exe`), which the installer does
not add to `PATH`.

No Python extra is needed — unlike `lazybridge[claude-code]`, this engine has
no SDK dependency, only the CLI.

## Usage

```python
from lazybridge import Agent, CodexEngine

agent = Agent(
    name="research",
    engine=CodexEngine(system="Be concise."),
    tools=[search, specialist_agent],
)
print(agent("Summarise the latest filing").text())
```

`CodexEngine` composes like any other engine — `tools=[...]` with a child
agent, `Agent.chain`, `AgentPool`, `Agent.stream`, `output=<model>` — because
those all dispatch through `Engine.run()`/`Engine.stream()` and never
special-case the engine type.

Constructor options mirror `LLMEngine`/`ClaudeCodeEngine` where the underlying
runtime supports them: `model`, `cwd`, `system`, `reasoning_effort`,
`request_timeout`, `stream_idle_timeout`, `max_retries`, `retry_delay`,
`tool_timeout`. The `system` value is sent as an App Server developer
instruction, separately from the user task, so it retains instruction priority.

## Why not `codex exec`

`codex exec --json` was the obvious candidate and was rejected: it cancels
non-interactive MCP tool calls unless `--dangerously-bypass-approvals-and-sandbox`
is passed, and that flag also removes the read-only sandbox. `CodexEngine`
therefore talks to `codex app-server` over JSON-RPC directly and never uses
that bypass.

## Protocol notes

Verified against a live `codex app-server` (codex-cli 0.148.0); the
authoritative schema comes from `codex app-server generate-json-schema --out <dir>`.

- `thread/start` takes `sandbox: "read-only"` — the kebab-case `SandboxMode`
  enum. `"readOnly"` is rejected outright with `unknown variant`.
- `dynamicTools` on `thread/start` works but is **absent from the generated
  schema**, so it is the field most likely to break on a Codex upgrade. The
  test fixture asserts the exact request shape so a regression fails locally.
- Token usage arrives only through `thread/tokenUsage/updated` notifications;
  `turn/completed` carries none. `total` is cumulative **over the thread**, not
  the turn (measured: 15137 after turn 1, 30292 after turn 2), so on a resumed
  thread the engine subtracts the total reported before the turn began. `last`
  is not used instead because it is only the final model call and would drop
  the ones made before each tool round-trip.
- Every notification carries `threadId`/`turnId`, and `turn/start` returns the
  turn's id. The engine attributes usage by turn id and ignores a
  `turn/completed` naming a different turn — on a resumed thread the server can
  replay older ones, and taking the first would return a previous answer.
- `thread/resume` restores a durable thread **in a different process**
  (verified live), and accepts `dynamicTools` even though the generated schema
  omits the field there too — a resumed thread must re-register them, since the
  callbacks live in the new subprocess.
- `ephemeral: true` means unresumable, not merely short-lived:
  `thread/resume` on such a thread answers `no rollout found for thread id`.
- **`cost_usd` is always `0.0`.** Under ChatGPT-plan auth the App Server
  reports plan rate-limit percentages, never a per-turn price. Token counts
  are populated, so `Session.usage_summary()` still attributes tokens per
  agent.
- The App Server numbers its own requests to the client from 0, independently
  of the client's ids, so the read loop dispatches on the presence of
  `method` rather than on the id.

## Multimodal

`images=` is forwarded: a URL passes through as an `image` UserInput item, and
inline bytes are sent as a `data:` URL.

```python
agent("What is in this chart?", images=["C:/work/chart.png"])
```

`audio=` is **not** forwarded. The protocol has `audio`/`localAudio` variants
and accepts them without error, but the model then reports it cannot access
the attachment (verified live with both), so the engine drops it with a
`UserWarning` instead of paying for an ignored attachment.

Codex can also *produce* images — the protocol's `ThreadItem` union includes an
`imageGeneration` item with a `savedPath`. This engine reads only the final
`agentMessage` text, so a generated image would not reach the `Envelope`.
Generated audio exists only in the realtime session API, not in this
turn-based path.

## Structured output

`output=<model>` is enforced server-side via `turn/start`'s native
`outputSchema` whenever the schema qualifies — the same guarantee
`ClaudeCodeEngine` gets from the Agent SDK's `output_format`. `turn/start`
only accepts OpenAI-*strict* schemas: `additionalProperties: false` on every
object **and** `required` listing every property, which a plain Pydantic
schema doesn't satisfy on its own (a bare schema fails the turn with
`invalid_json_schema`). `lazybridge.core.structured.to_openai_strict_schema`
does that rewrite: an optional property is carried into `required` only when
it already accepts `null` as written (an `Optional[X] = None` field, a
`Literal[..., None]`), since that's the only case where forcing it into
`required` doesn't change what a round-trip through the model then accepts.

Not every schema qualifies. The engine falls back to the old prompt-priming
behavior — ask for the shape in the prompt, let
`Agent._validate_and_retry`'s post-hoc repair catch drift — whenever the
native rewrite would have to change accepted semantics instead of just
restating them:

- an optional property whose own type never admits `null` (e.g.
  `count: int = 5` — forcing it into `required` would let the model legally
  answer `null` there, which the destination model then rejects);
- a model with `extra="allow"` (or any object whose `additionalProperties`
  is explicitly `true`/typed) — closing it would forbid dynamic fields the
  destination type actually accepts;
- a fixed-length tuple (`tuple[str, int]`, rendered as `prefixItems`) — a
  keyword outside the strict-mode subset with no equivalent to translate it
  into;
- the output type's top level isn't an object (e.g. `output=list[str]`).

`output=str` (the default) and `output=Any` are unaffected either way — there
is nothing to constrain.

## Distinguishing LazyBridge threads on disk

Every thread a `CodexEngine` creates is tagged with the App Server's own
`threadSource` field (`ThreadStartParams.threadSource` — verified against the
generated protocol schema, and observed on disk as
`session_meta.payload.source` in a real rollout file under
`~/.codex/sessions/...`, alongside values like `"vscode"` the interactive CLI
sets for its own sessions):

```python
engine = CodexEngine()                          # thread_source="lazybridge" (default)
engine = CodexEngine(thread_source="my-app")     # a caller-specific label
engine = CodexEngine(thread_source=None)         # omit it entirely
```

This is creation-time metadata only — sent on `thread/start`, never
re-sent on `thread/resume` — because the protocol has no endpoint to change
it after a thread exists. It has no bearing on `codex resume`'s picker
(which titles sessions from their content, not this field); it exists so a
script can tell LazyBridge-created threads apart from interactive ones by
grepping rollout files for `"source": "lazybridge"`, e.g. as a starting point
for a retention/cleanup pass.

## Reading the subscription quota

The subscription behind `CodexEngine` has a weekly quota, and it can be read
without spending any of it. `fetch_codex_usage()` asks the App Server for
`account/rateLimits/read` -- no thread and no turn are involved:

```python
from lazybridge.engines.codex.usage import fetch_codex_usage

snapshot = await fetch_codex_usage()
weekly = snapshot.weekly()          # the seven-day window, or None
if weekly is not None:
    print(weekly.used_percent, weekly.resets_at)
```

Each window carries **both** a percentage and a reset time (`resets_at` is
timezone-aware UTC, or `None` if the server did not say), and they are meant to
be read together: 85% is comfortable on the first day of a window and critical
on the last.

- `snapshot.windows` lists every bucket the account reported; an account can
  report more than one. Each is a `CodexUsageWindow` with `limit_id`, `kind`
  (`"primary"` or `"secondary"`), `used_percent`, `window_duration_minutes`,
  `resets_at` and `reached_type`.
- `snapshot.weekly(limit_id="codex")` picks the seven-day bucket by its
  **duration** (`WEEKLY_WINDOW_MINUTES`, 10080), not by position -- `primary`
  is not always the weekly one -- and returns `None` when there is none.
- `snapshot.plan_type` and `snapshot.raw` (the untouched response) are there
  for anything the dataclass does not model yet.

It needs a local Codex login, like the engine itself. Once the App Server
process is actually running, `fetch_codex_usage` raises `RuntimeError` if it
times out (30 s by default), closes the stream, or refuses the method -- an
older `codex` binary answers "method not found". Locating or starting the
`codex` binary itself is a separate failure mode: a `codex` CLI missing from
`PATH`/`CODEX_BIN`/the app's install directory raises `FileNotFoundError`
before any of that, not `RuntimeError`. A caller that gates work on the
number must treat all of those, and a `None` from `weekly()`, as
**unknown**, never as *plenty left*.

Claude Code's equivalent is different in kind: there is no typed field, so its
figure is read out of the prose the CLI prints for `/usage` and can only be as
reliable as that wording.

## Not implemented yet

- **No model validation.** `model=` is passed straight to `thread/start`
  without checking it against the account's `model/list`, so an invalid model
  surfaces as whatever error the App Server returns.
- **No cross-process locking.** `persist_thread=True` serialises runs against
  one thread id *within* a process; two processes resuming the same thread at
  once is still on the caller to prevent.
- **No built-in file/web tools.** Claude's `Read`/`Glob`/`Grep`/`WebSearch`
  surface has no exposed counterpart here; Codex runs with its own read-only
  sandbox rooted at `cwd`.
