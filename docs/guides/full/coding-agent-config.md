# Coding-agent permissions and approvals

CodingAgentConfig supplies one approval contract while retaining each runtime's
native controls. It configures actions, rather than final workflow acceptance.

## Profiles and a working writer

```python
from pathlib import Path
from lazybridge import (
    Agent, ClaudeCodeEngine, ClaudeCodePolicy, CodexEngine, CodingAgentConfig,
    TerminalApprovalGate,
)

repo = str(Path.cwd().resolve())
gate = TerminalApprovalGate()
# writer(gate) alone does not expose Claude Write/Edit/Bash.
claude_config = CodingAgentConfig(
    claude=ClaudeCodePolicy(permission_mode="default",
                            preapprove_application_tools=False,
                            extra_tools=("Write", "Edit", "Bash")),
    approval_gate=gate,
)
writer = Agent(name="writer", engine=ClaudeCodeEngine(
    cwd=repo, file_roots=[repo], config=claude_config, request_timeout=600,
))
reviewer = Agent(name="reviewer", engine=CodexEngine(
    cwd=repo, config=CodingAgentConfig.reviewer(),
))
codex_writer = Agent(name="codex-writer", engine=CodexEngine(
    cwd=repo, config=CodingAgentConfig.writer(gate), request_timeout=600,
))
```

| Profile | Claude | Codex | Application tools |
| --- | --- | --- | --- |
| Default config | Derived read/web built-ins, automatic permission mode | read-only, never | Preapproved |
| reviewer() | Read/web; no write extras | read-only, never | Fail closed without a gate |
| writer(gate) | default permission mode; no write extras added | workspace-write, on-request | Sent to the gate |

The default preapproval trusts your Python tools. Native read-only permissions
do not remove side effects from a tool you supplied. Use a profile explicitly
when that distinction matters.

## Custom policy

```python
from lazybridge import ApprovalDecision, ClaudeCodePolicy, CodexPolicy, CodingAgentConfig

async def approve(request):
    if request.kind == "tool" and request.name in {"project_status", "read_ticket"}:
        return ApprovalDecision.allow()
    return ApprovalDecision.deny(f"Blocked {request.kind}: {request.name}")

config = CodingAgentConfig(
    claude=ClaudeCodePolicy(
        preapprove_application_tools=False,
        extra_tools=("Write", "Edit"),
        setting_sources=(),
    ),
    codex=CodexPolicy(
        sandbox="workspace-write", approval_policy="on-request",
        preapprove_dynamic_tools=False,
    ),
    approval_gate=approve,
)
```

ApprovalRequest carries provider, kind, name, arguments, cwd, reason,
thread_id, turn_id and raw runtime data. Kinds are tool, command, file_change,
permissions and user_input. A sync or async gate returns ApprovalDecision:
allow, allow_for_session, deny or cancel. A direct engine approval_gate conflicts
with a different config gate. No gate means deny when approval is required.

TerminalApprovalGate provides a terminal UI. For durable approvals use the
[core approval extension](../../reference/extensions.md#tiered-approval-gate)
or [LazyTools code bridge](https://tools.lazybridge.com/code-bridge/).
Final diff/release approval is a separate workflow step.

## Native policy fields

| ClaudeCodePolicy | Default | Meaning |
| --- | --- | --- |
| permission_mode | None | Auto-select dontAsk if nothing needs gating, default otherwise. Explicit modes: default, dontAsk, acceptEdits, bypassPermissions, plan, auto. |
| preapprove_application_tools | True | Trust in-process MCP/Python tools without calling the gate. |
| allowed_tools / disallowed_tools | () / () | Preapproval and denial rules; do not expose a missing built-in. |
| extra_tools | () | Adds names to the derived built-in set. |
| setting_sources | () | No ambient user/project/local settings by default. |
| auto_compact_window | None | Per-agent context window used to bring compaction forward. |

File roots belong to ClaudeCodeEngine and constrain file hooks, including
Read/Glob/Grep/Write/Edit/NotebookEdit. Bash is outside that confinement, so
granting Bash or another unconfined built-in requires an approval gate.
Permissive permission modes do not bypass the engine's file-root hook.

| CodexPolicy | Default | Meaning |
| --- | --- | --- |
| sandbox | read-only | read-only, workspace-write or danger-full-access. |
| approval_policy | never | never, on-request or untrusted. |
| preapprove_dynamic_tools | True | Trust Python tools independently of native sandbox. |
| writable_roots | None | Extra absolute paths additive to cwd under workspace-write; ignored by other sandbox modes. |
| web_search | None | Runtime default, or disabled/cached/indexed/live per subprocess. |
| auto_compact_token_limit | None | Per-process token count that triggers compaction. |

Git worktrees may need explicit access to their shared Git metadata outside
cwd for add/commit. The caller resolves those paths; LazyBridge does not infer
Git topology or authorize unrelated workspaces.

## Grant scope and compaction

Generic allow_for_session grants live on the LazyBridge Session, scoped by
(provider, agent_name) and action kind/name. Command grants also fingerprint
a usable arguments.command string; commands without one are not cached.
Without a Session, grants last only for that run. allow is single-use.
This cache does not persist through a process restart or native transcript.
Gates that manage their own narrower scope can set manages_session_grants=True.
TieredGate uses its own cwd/policy-fingerprint scope and audit trail.

```python
from lazybridge import ClaudeCodePolicy, CodexPolicy, CodingAgentConfig

claude_config = CodingAgentConfig(claude=ClaudeCodePolicy(auto_compact_window=140_000))
codex_config = CodingAgentConfig(codex=CodexPolicy(
    web_search="live", auto_compact_token_limit=140_000,
))
```

Claude's window becomes CLAUDE_CODE_AUTO_COMPACT_WINDOW. The effective threshold
cannot exceed the model's actual context window. Codex's trigger becomes
model_auto_compact_token_limit in per-process -c overrides, alongside web_search.
Neither changes the user's global configuration. Codex model_context_window
is not exposed. LLMEngine uses its own turns/tools/deadlines, with no native
history-compaction knob.

See [coding agents](coding-agents.md#privileges-a-four-tier-policy) and
[Claude Code writes, Codex reviews](claude-writes-codex-reviews.md).
