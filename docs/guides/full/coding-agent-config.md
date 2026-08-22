# Coding-agent permissions and approvals

`CodingAgentConfig` configures Claude Code and Codex with one shared approval
gate while preserving each runtime's native sandbox and permission system.

## Safe profiles

```python
from lazybridge import (
    Agent,
    ClaudeCodeEngine,
    CodexEngine,
    CodingAgentConfig,
    TerminalApprovalGate,
)

gate = TerminalApprovalGate()

writer = Agent(
    name="writer",
    engine=ClaudeCodeEngine(
        cwd="C:/work/project",
        file_roots=["C:/work/project"],
        config=CodingAgentConfig.writer(gate),
    ),
    tools=[...],
)

reviewer = Agent(
    name="reviewer",
    engine=CodexEngine(
        cwd="C:/work/project",
        config=CodingAgentConfig.reviewer(),
    ),
)
```

The writer profile uses Claude's `default` permission mode and Codex's
`workspace-write` sandbox with `on-request` approvals. Application tools are
not pre-approved and are sent to the gate. The reviewer profile is read-only
and fails closed for application tools.

Omitting `config=` preserves the engines' original trusted behavior for
backward compatibility. Security-sensitive applications should select a
profile explicitly.

## Custom policy and approval UI

```python
from lazybridge import (
    ApprovalDecision,
    ClaudeCodePolicy,
    CodexPolicy,
    CodingAgentConfig,
)

async def approve(request):
    if request.kind == "tool" and request.name in {"read_ticket", "get_quote"}:
        return ApprovalDecision.allow()
    if request.kind == "command" and request.name.startswith("git status"):
        return ApprovalDecision.allow_for_session()
    return ApprovalDecision.deny(f"Blocked {request.kind}: {request.name}")

config = CodingAgentConfig(
    claude=ClaudeCodePolicy(
        permission_mode="default",
        preapprove_application_tools=False,
        allowed_tools=("Read", "Glob", "Grep"),
        disallowed_tools=("Bash(rm *)",),
        setting_sources=(),
    ),
    codex=CodexPolicy(
        sandbox="workspace-write",
        approval_policy="on-request",
        preapprove_dynamic_tools=False,
    ),
    approval_gate=approve,
)
```

`ApprovalRequest` normalizes the provider, action kind, name, arguments,
working directory, reason, and native request identifiers. A gate returns
`ApprovalDecision.allow()`, `.allow_for_session()`, `.deny()`, or `.cancel()`.

Use `TerminalApprovalGate` for local development. In production, pass an async
callback backed by your web UI, Telegram bot, queue, or policy service. Keep
workflow approvals such as “approve this complete diff before push” as a
separate `HumanEngine` step; this gate controls individual runtime actions.

### How far a grant reaches

`allow_for_session()` is remembered **per agent, per LazyBridge `Session`** —
keyed by `(provider, agent_name)` and parked on the `Session` object, so:

- a grant survives the next runs of that agent instead of re-prompting every
  turn (the dispatcher itself is rebuilt per run, so a naive cache would);
- a second agent sharing the same engine instance is still asked separately,
  which matters because engines are shared freely — that is why
  `resolve_agent_name` exists;
- each provider keeps its own grants, so approving a tool for Codex does not
  silently approve the same name for Claude Code.

Without a `Session` there is nowhere to persist a grant, so it degrades to
"for the rest of this run". `allow()` is always single-use.

## Provider mapping

| Shared configuration | Claude Code | Codex |
|---|---|---|
| Runtime sandbox | `file_roots`, built-in tool selection | `sandbox` |
| Native approval mode | `permission_mode` | `approval_policy` |
| Granting more built-ins | `extra_tools` | (n/a — the sandbox decides) |
| Pre-approved app tools | `allowed_tools` / MCP allow rules | `preapprove_dynamic_tools` |
| Hard denial | `disallowed_tools` | sandbox plus gate denial |
| Human callback | `can_use_tool` | App Server approval requests and dynamic tools |
| Ambient configuration | `setting_sources` | local Codex configuration plus thread overrides |
| When the agent compacts | `auto_compact_window` | `auto_compact_token_limit` |

`ClaudeCodePolicy.permission_mode` defaults to `None`, which lets the engine
pick per run: `"dontAsk"` when nothing needs gating (application tools only,
all pre-approved) and `"default"` when something does — built-in tools, an
approval gate, or `preapprove_application_tools=False`. Pin a value only when
you want to override that; a hardcoded `"default"` would put a fully
pre-approved, tool-only agent into prompting mode with no callback able to
answer.

## Deciding when an agent compacts

Both CLIs summarise their own history when it grows too long. Left alone they
use a default tuned for the model, which is usually the right thing — but a
long-running agent that keeps refilling its context can be told to compact
earlier, per agent, without touching a machine-wide configuration file.

```python
from lazybridge.engines.coding import ClaudeCodePolicy, CodexPolicy, CodingAgentConfig

claude = CodingAgentConfig(claude=ClaudeCodePolicy(auto_compact_window=140_000))
codex = CodingAgentConfig(codex=CodexPolicy(auto_compact_token_limit=140_000))
```

The two numbers do **not** mean quite the same thing, and the difference is
worth knowing before you copy one into the other.

`auto_compact_window` is a *window*: Claude Code compacts when usage
approaches it, and the effective threshold is the minimum of your value and
the model's real context window. So it can bring compaction forward, never
push it beyond what the model allows. It travels to the agent as the
`CLAUDE_CODE_AUTO_COMPACT_WINDOW` environment variable, which outranks every
settings file — deliberately, because an agent reads none of them by default
(`setting_sources` is empty), and the alternative would be inheriting a
human's personal settings wholesale to deliver one number.

`auto_compact_token_limit` is a *trigger*: the token count at which Codex
starts summarising. It travels as a `-c model_auto_compact_token_limit=<n>`
override on that agent's own App Server subprocess, so the shared
`~/.codex/config.toml` is never touched and no other Codex on the machine
changes behaviour.

Codex's companion setting `model_context_window` is deliberately not exposed.
It describes the budget rather than enlarging the model's real limit, and
setting it is reported upstream to break auto-compaction outright
([openai/codex#16068](https://github.com/openai/codex/issues/16068)).

Both paths are verified end to end rather than by construction: a Claude Code
agent given `auto_compact_window=137000` echoes that value back from
`$CLAUDE_CODE_AUTO_COMPACT_WINDOW` inside its own subprocess, and
`codex app-server --strict-config -c model_auto_compact_token_limit=140000`
starts cleanly where an invented key is rejected outright.

Neither knob exists for `LLMEngine`: an API-backed agent has no compaction to
schedule, and its budgets are turns and tool calls rather than tokens.

## Granting write tools to a gated agent

`ClaudeCodePolicy.extra_tools` adds names to the built-in set the SDK is
allowed to offer the model. Without it, no approval gate can ever be *asked*
about a write — the model simply never has `Write`/`Edit`/`Bash`:

```python
config = CodingAgentConfig(
    claude=ClaudeCodePolicy(
        preapprove_application_tools=False,
        extra_tools=("Write", "Edit", "Bash"),
    ),
    approval_gate=my_gate,
)
```

Granting is **not** pre-approving: unless a name is also in `allowed_tools`,
every call still routes through the gate. One asymmetry matters — `file_roots`
confinement is enforced by a hook over the *file* tools (`Read`/`Glob`/`Grep`/
`Edit`/`Write`/`NotebookEdit`), and `Bash` is not path-confinable that way: an
approved command can touch anything its process can. Its only boundary is the
gate's policy, so the engine **refuses at construction** to grant `Bash`
without an `approval_gate` rather than advertising a confinement it cannot
deliver.

A worked, end-to-end tiered policy (allow / ask-once / ask-every-time / deny,
with a terminal or chat approval channel) lives in
[Coding agents in practice](coding-agents.md#privileges-a-four-tier-policy).
