# Coding engines: ClaudeCodeEngine and CodexEngine

Read when an Agent must run through the locally authenticated Claude Code or
Codex runtime: reviewers, writers, durable conversations, approvals, quota.
Source of truth: `lazybridge/engines/claude_code/`, `lazybridge/engines/codex/`,
`lazybridge/engines/coding.py`, `lazybridge/engines/sessions.py`; docs
`docs/guides/full/{coding-agents,coding-agent-config,claude-code-engine,codex-engine,session-registry,claude-writes-codex-reviews}.md`.
For background writers and job records see `extensions.md` (delegation).

## Same Agent contract, different runtime

```python
from lazybridge import Agent, ClaudeCodeEngine, CodexEngine, LLMEngine, Memory, Session, Store

state = dict(tools=[project_status], memory=Memory(), store=Store(), session=Session())
claude = Agent(name="claude", engine=ClaudeCodeEngine(), **state)
codex = Agent(name="codex", engine=CodexEngine(), **state)
api = Agent(name="api", engine=LLMEngine("gpt-6.1-sol"), **state)
```

| | `ClaudeCodeEngine` | `CodexEngine` |
|---|---|---|
| Transport | Claude Agent SDK (`pip install "lazybridge[claude-code]"`) | `codex app-server` JSON-RPC over stdio (no Python extra; CLI found via `CODEX_BIN`, `PATH`, Windows desktop-app dirs, newest stable wins) |
| Python tools | in-process MCP adapter | App Server dynamic tools |
| Default native access | `WebSearch`/`WebFetch` (`web=True`), `Read`/`Glob`/`Grep` under `cwd`/`file_roots` | read-only sandbox |
| Writes | only via `ClaudeCodePolicy(extra_tools=("Write","Edit","Bash"))` + policy | `sandbox="workspace-write"` (+ `writable_roots`) |
| Model default | `"sonnet"` (aliases `opus`/`haiku` or full ids) | runtime-configured (`model=None`), passed through unvalidated |
| Effort | `reasoning_effort` validated (low/medium/high/xhigh/max), `thinking` (adaptive/disabled/int budget), `fallback_model` | `reasoning_effort` passed through; no `thinking` |
| Durable history | `persist_session=True` / `session_id=` | `persist_thread=True` / `thread_id=` |
| Named continuity | `session_alias=` / `session_registry=` | same |
| Structured output | native SDK `output_format` | native `outputSchema` when the schema is OpenAI-strict representable, else prompt priming + Agent repair |
| Cost | SDK-reported | `cost_usd` is always 0.0 (no price in RPC) — not "free" |
| Quota | `await engine.usage()` parses `/usage` prose (spends a small turn) | `lazybridge.engines.codex.usage.fetch_codex_usage()` (no turn) |
| Labels on disk | `tag="lazybridge"` on new durable sessions | `thread_source="lazybridge"` + `originator=lazybridge` |

Shared knobs: `cwd`, `system`, `config`, `approval_gate`, `request_timeout`
(120 s, whole run incl. retries), `stream_idle_timeout` (90 s), `max_retries`,
`retry_delay`, `tool_timeout`, `client` (inject a fake transport in tests).
Claude only: `max_turns` (20), `file_roots`, `web`, `session_mode`
(`"memory"`/`"runtime"`), `session_name` (runtime-mode slot label, not an
alias). Codex only: `review_target`, `thread_source`.

## Permission profiles (`CodingAgentConfig`)

| Profile | Claude | Codex | Python application tools |
|---|---|---|---|
| default `CodingAgentConfig()` | read/web built-ins, auto permission mode | read-only, approval `never` | preapproved |
| `CodingAgentConfig.reviewer()` | read/web, no write extras | read-only, `never` | fail closed without a gate |
| `CodingAgentConfig.writer(gate)` | `permission_mode="default"`, **no write tools added** | `workspace-write`, `on-request` | sent to the gate |

```python
from lazybridge import (Agent, ClaudeCodeEngine, ClaudeCodePolicy, CodexEngine,
                        CodingAgentConfig, TerminalApprovalGate)

gate = TerminalApprovalGate()                      # dev only; production: TieredGate
claude_writer = Agent(name="writer", engine=ClaudeCodeEngine(
    cwd=repo, file_roots=[repo], request_timeout=600,
    config=CodingAgentConfig(
        claude=ClaudeCodePolicy(permission_mode="default",
                                preapprove_application_tools=False,
                                extra_tools=("Write", "Edit", "Bash")),
        approval_gate=gate,
    ),
))
codex_reviewer = Agent(name="reviewer", engine=CodexEngine(
    cwd=repo, config=CodingAgentConfig.reviewer(),
))
```

Policy fields. `ClaudeCodePolicy`: `permission_mode` (None = auto `dontAsk` /
`default`), `preapprove_application_tools=True`, `allowed_tools`,
`disallowed_tools`, `extra_tools`, `setting_sources=()` (no ambient
settings), `auto_compact_window`. `CodexPolicy`: `sandbox` (`read-only` /
`workspace-write` / `danger-full-access`, kebab-case), `approval_policy`
(`never` / `on-request` / `untrusted`), `preapprove_dynamic_tools=True`,
`writable_roots` (needed when `cwd` is a git worktree: the index lives under
the main repo's `.git/worktrees/<name>`), `web_search`
(`disabled`/`cached`/`indexed`/`live`, per-process `-c`), `auto_compact_token_limit`.

Rules that bite:

- A gate cannot approve a tool the model was never offered: Claude writers
  need `extra_tools`. Granting is not preapproving (`allowed_tools` does that).
- `file_roots` confine file hooks only. `Bash` (or any unconfinable built-in)
  without an approval gate is refused at construction (`ValueError`).
- Native read-only sandboxes do not sandbox your Python tools.
- `ApprovalRequest` (provider, kind = tool/command/file_change/permissions/
  user_input, name, arguments, cwd, reason, thread_id, turn_id, raw) →
  `ApprovalDecision.allow()` / `allow_for_session()` / `deny(reason)` /
  `cancel()`. Sync or async `ApprovalGate`. No gate when approval is needed
  means deny. A direct `approval_gate=` that differs from `config`'s conflicts.
- `allow_for_session` grants live on the LazyBridge `Session`, keyed by
  (provider, agent name, kind, name[, command]); not persisted. Use
  `lazybridge.ext.approval.TieredGate` for scoped, audited four-tier policy.
- Since 1.5.0 `LLMEngine(approval_gate=...)` exists too, so swapping a coding
  engine for an API engine keeps the safety layer.
- Approving runtime actions never approves the final diff, merge or publish.

## Durable conversations and aliases

```python
from lazybridge import Agent, CodexEngine, SessionRegistry

registry = SessionRegistry()            # ~/.lazybridge/sessions.json or $LAZYBRIDGE_SESSIONS_FILE
engine = CodexEngine(cwd=repo, session_alias="reviewer", session_registry=registry)
Agent(name="reviewer", engine=engine)("Remember the word PINEAPPLE.")
print(engine.thread_id)                  # native id, bound to the alias
```

- Default runs are ephemeral and **cannot** be resumed later; choose
  persistence before the first turn.
- Registry key is `(kind, scope, name)`: kind `codex`/`claude`, scope the
  resolved cwd. Same alias in two repos or two engines = two conversations.
  Names: ASCII letter first, then letters/digits/`_.-`, max 64 chars.
- Explicit `thread_id=` / `session_id=` wins and rebinds the alias. API:
  `resolve`, `bind`, `rename`, `forget`, `entries`;
  `default_session_registry()` / `set_default_session_registry()` in
  `lazybridge.engines`.
- On resume the engine stops prepending `Memory`; explicit `context=` and
  `sources=` still flow. Runs/streams serialize in-process per engine, alias
  and id; there is **no cross-process transcript lock**.
- Claude `session_mode="runtime"` parks the id on one LazyBridge `Session`
  (process-local); it is not a substitute for the registry.

## Native review (Codex)

`CodexEngine(review_target={"type": "uncommittedChanges"})` (or
`{"type": "baseBranch", "branch": "main"}`, `{"type": "commit", "sha": ...}`)
runs the App Server `review/start` harness: severity-tagged findings with
file:line. The Agent's prompt is **not** sent. Pair with
`persist_thread=True` and follow up with a new engine on that `thread_id`
without `review_target`. For a `verify=` judge use an ordinary prompted agent.

## Collaboration patterns

- Writer/reviewer across runtimes: `Agent(engine=ClaudeCodeEngine(...),
  verify=codex_reviewer, max_verify=3)`; exhaustion returns the last draft
  without approval.
- Second opinion panel: `Agent.parallel(claude_reviewer, codex_reviewer)`
  (labelled join, not consensus).
- Coding agents nest anywhere: `tools=[...]`, `Agent.chain`, `Plan` steps.

## Failures and retries

- Both Codex errors live in `lazybridge.engines.codex.app_server`.
- `CodexTransportError` (a `ConnectionError`) carries phase, PID, return
  code and a 16 KiB stderr tail. Startup and pre-turn `thread/start` /
  `thread/resume` failures are retried once; nothing after a turn was sent
  is replayed.
- `CodexTurnUncertain`: a durable turn may already have run tools. Resume and
  inspect before deciding; never blindly replay side effects.
- Missing CLI/executable is permanent, not transient.
- Long repository tasks: raise `request_timeout` and align any MCP host
  timeout (the default 120 s bounds all retries).

## Outside LazyBridge

Detached repository jobs, approval tickets in chat, quota-aware model routing
and named code sessions for an assistant are LazyTools features
(`lazytools-code-bridge`, `lazytools.routing`, the MCP `code_*` tools); see
<https://tools.lazybridge.com/code-bridge/>.
