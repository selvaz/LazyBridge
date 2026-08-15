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

`ClaudeCodePolicy.permission_mode` defaults to `None`, which lets the engine
pick per run: `"dontAsk"` when nothing needs gating (application tools only,
all pre-approved) and `"default"` when something does — built-in tools, an
approval gate, or `preapprove_application_tools=False`. Pin a value only when
you want to override that; a hardcoded `"default"` would put a fully
pre-approved, tool-only agent into prompting mode with no callback able to
answer.
| Pre-approved app tools | `allowed_tools` / MCP allow rules | `preapprove_dynamic_tools` |
| Hard denial | `disallowed_tools` | sandbox plus gate denial |
| Human callback | `can_use_tool` | App Server approval requests and dynamic tools |
| Ambient configuration | `setting_sources` | local Codex configuration plus thread overrides |
