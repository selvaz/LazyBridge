# Coding agents in practice

A worked guide to the two local-runtime engines — `ClaudeCodeEngine` and
`CodexEngine` — covering model selection, reasoning knobs, privileges, durable
conversations, and the recipes those combine into. The per-engine reference
pages ([Claude Code](claude-code-engine.md), [Codex](codex-engine.md)) describe
each runtime on its own; this page is about choosing and configuring them.

Both are ordinary LazyBridge engines: `Agent`, `Memory`, `Session`, `tools=`,
`output=`, `stream()` all behave exactly as they do with `LLMEngine`. Neither
needs an API key — each reuses its CLI's own login.

## Which engine

| | `ClaudeCodeEngine` | `CodexEngine` |
|---|---|---|
| Transport | Claude Agent SDK (in-process) | JSON-RPC to `codex app-server` |
| Install | `pip install "lazybridge[claude-code]"` + `claude` CLI | `codex` CLI only (no Python extra) |
| Shell | **No** — no `Bash` unless granted via `extra_tools` | Yes, inside its sandbox |
| File reads | `Read`/`Glob`/`Grep`, confined to `file_roots` | its own read-only sandbox rooted at `cwd` |
| Web | `web=True` → `WebSearch`/`WebFetch` | account-level `web__run`, via `~/.codex/config.toml` |
| Structured output | server-enforced (`output_format`) | prompted, not enforced |
| Reported cost | real `cost_usd` | always `0.0` (plan auth reports no price) |
| Reasoning knobs | `reasoning_effort` **and** `thinking` | `reasoning_effort` only |
| Durable conversation | `persist_session` → `session_id` | `persist_thread` → `thread_id` |

Rule of thumb: **Codex** when you want a shell and a second opinion from
another model family; **Claude Code** when you want enforced structured output,
a real cost figure, or fine-grained per-tool approval.

## Quick start

```python
from lazybridge import Agent, ClaudeCodeEngine, CodexEngine

claude = Agent(
    engine=ClaudeCodeEngine(model="sonnet", cwd="C:/work/project"),
    name="claude-reader",
)
codex = Agent(
    engine=CodexEngine(cwd="C:/work/project"),
    name="codex-reader",
)

print(claude("What does src/parser.py do?").text())
print(codex("What does src/parser.py do?").text())
```

`cwd` alone makes that directory readable (it becomes the sole `file_roots`
entry for Claude, and Codex's sandbox root). Neither can write.

## Models

### Claude Code

Aliases resolve to the newest model of that family available to the
authenticated account:

```python
ClaudeCodeEngine(model="sonnet")   # default
ClaudeCodeEngine(model="opus")     # harder tasks, if the plan allows
ClaudeCodeEngine(model="haiku")    # fast and cheap; verify on your account
ClaudeCodeEngine(model="claude-sonnet-4-5-20250929")  # a pinned full id
```

Do not invent informal names like `"sonnet-5"` — they are not aliases. A
`fallback_model=` is tried when the primary is overloaded:

```python
ClaudeCodeEngine(model="opus", fallback_model="sonnet")
```

### Codex

`model=` is passed straight to `thread/start` **without validation** — an
invalid name surfaces as whatever error the App Server returns. Leaving it
`None` uses whatever `~/.codex/config.toml` configures:

```python
CodexEngine()                       # the CLI's configured model
CodexEngine(model="gpt-5.6-sol")    # explicit
```

## Reasoning: two different knobs

| Knob | Engines | Values | What it controls |
|---|---|---|---|
| `reasoning_effort` | both | `"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"` (Claude); free-form per model (Codex) | how much reasoning the model spends |
| `thinking` | Claude only | `"adaptive"`, `"disabled"`, or an int token budget | extended-thinking mode |

```python
# Claude: both knobs, independently
ClaudeCodeEngine(model="opus", reasoning_effort="high", thinking="adaptive")
ClaudeCodeEngine(model="sonnet", thinking=8000)      # explicit token budget
ClaudeCodeEngine(model="haiku", thinking="disabled")  # fastest

# Codex: effort only — there is no `thinking` parameter
CodexEngine(reasoning_effort="medium")
CodexEngine(reasoning_effort="xhigh")
```

Claude validates `reasoning_effort` at construction (a typo raises
immediately). Codex passes it through unvalidated, because the App Server
advertises each model's accepted values through `model/list` and they differ
per model.

## Privileges: a four-tier policy

The engines consult one `ApprovalGate` for every action that is not
pre-approved. That single callback is enough to express a real policy — the
example below sorts every tool call into four tiers:

| Tier | Meaning |
|---|---|
| allow | runs without asking (the agent's job) |
| session | the first call asks; the grant then sticks |
| ask | every call asks (irreversible or outward-facing) |
| deny | never runs, no human can approve it here |

```python
from fnmatch import fnmatch

from lazybridge import (
    Agent, ApprovalDecision, ClaudeCodeEngine, ClaudeCodePolicy, CodingAgentConfig,
)

POLICY = (
    ("allow",   "Read"),       ("allow", "Glob"),  ("allow", "Grep"),
    ("allow",   "WebSearch"),  ("allow", "WebFetch"),
    ("session", "Write"),      ("session", "Edit"),
    ("ask",     "Bash"),
)

async def gate(request):
    for tier, pattern in POLICY:
        if not fnmatch(request.name, pattern):
            continue
        if tier == "allow":
            return ApprovalDecision.allow()
        approved = await ask_the_human(request)     # your UI/bot/queue
        if not approved:
            return ApprovalDecision.deny("refused by the approver")
        return ApprovalDecision.allow_for_session() if tier == "session" else ApprovalDecision.allow()
    return ApprovalDecision.deny(f"{request.name} matches no rule (default deny)")

engine = ClaudeCodeEngine(
    cwd="C:/work/sandbox",
    file_roots=["C:/work/sandbox"],
    config=CodingAgentConfig(
        claude=ClaudeCodePolicy(
            preapprove_application_tools=False,
            extra_tools=("Write", "Edit", "Bash"),
        ),
        approval_gate=gate,
    ),
)
```

Three things make this work, and each is easy to get wrong:

- **`extra_tools`** is what puts `Write`/`Edit`/`Bash` in the model's hands at
  all. Without it the gate is never consulted about a write, because the tool
  does not exist for that agent.
- **`preapprove_application_tools=False`** is what sends your own
  `tools=[...]` through the gate too. Left at its default, they bypass it.
- **Default deny.** The final `return` matters more than any rule above it: a
  gate that falls through to "allow" on the pattern you forgot to write is not
  a gate.

Beware compound shell commands: matching `git add x && git commit` against
`"git *"` approves the commit under the tier of `git add`. Split on `&&`,
`;` and `|` and decide by the most severe segment.

### How far a grant reaches

`allow_for_session()` is remembered per `(provider, agent_name)` **on the
LazyBridge `Session` object**. Without a `Session` there is nowhere to keep
it, so it degrades to "the rest of this run". `allow()` is always single-use.
Approving a name for Codex never approves it for Claude.

### The confinement asymmetry

`file_roots` is enforced by a `PreToolUse` hook over the file tools, so it
holds even under permissive settings. **`Bash` is not confinable that way** —
an approved command reaches anything its process can. The engine therefore
refuses at construction to grant `Bash` (or any non-file, non-web built-in)
unless an `approval_gate` is configured:

```python
ClaudeCodeEngine(
    file_roots=["C:/work/sandbox"],
    config=CodingAgentConfig(claude=ClaudeCodePolicy(extra_tools=("Bash",))),
)
# ValueError: extra_tools grants ('Bash',) which file_roots cannot confine;
# configure CodingAgentConfig.approval_gate so a policy governs them
```

## Durable conversations

A follow-up on the same conversation skips re-reading everything the first
turn already read:

```python
# Codex
engine = CodexEngine(cwd=repo, persist_thread=True)
Agent(engine, name="reviewer")("review src/parser.py")
handle = engine.thread_id                     # keep this

# ...another process entirely, later:
Agent(CodexEngine(cwd=repo, thread_id=handle), name="reviewer")("and the retry path?")
```

```python
# Claude Code — same shape, different attribute
engine = ClaudeCodeEngine(cwd=repo, persist_session=True)
Agent(engine, name="reviewer")("review src/parser.py")
handle = engine.session_id

Agent(ClaudeCodeEngine(cwd=repo, session_id=handle), name="reviewer")("and the retry path?")
```

On a resumed conversation the engine **stops prepending `Memory`** — the
runtime already holds the history, and sending it again would state every past
turn twice.

### Finding and cleaning up what you created

Both engines label the conversations they start, so a retention pass can find
them without touching interactive ones:

```python
CodexEngine(thread_source="lazybridge")        # default; lands in the rollout file
ClaudeCodeEngine(tag="lazybridge")             # default; via the SDK's tag_session
```

```python
# Claude: list, filter, delete
from claude_agent_sdk import delete_session, list_sessions

for s in list_sessions():
    if s.tag == "lazybridge":
        delete_session(s.session_id)
```

Codex threads additionally carry `session_meta.payload.originator ==
"lazybridge"` unconditionally (from the `initialize` handshake), so they are
identifiable even without `thread_source`. Delete one with
`codex delete <id>`.

## Tools

Both engines take the same `tools=[...]` — plain functions, `Tool` objects,
`ToolProvider` instances, other agents — and each routes them through its own
channel: Codex registers them as App Server *dynamic tools*, Claude Code
builds a temporary in-process MCP server. Neither path depends on the
runtime's own MCP configuration, which is what makes them work where a
CLI-level MCP registration would be refused.

```python
def get_quote(symbol: str) -> dict[str, str]:
    """Return a quote for the requested symbol."""
    return {"symbol": symbol, "price": "123.45"}

Agent(engine=CodexEngine(), tools=[get_quote, specialist_agent], name="desk")
```

## Recipes

**A read-only reviewer, fail-closed:**

```python
Agent(
    engine=CodexEngine(
        cwd=repo,
        system="Report defects with file:line. Never patch.",
        config=CodingAgentConfig.reviewer(),
    ),
    name="reviewer",
)
```

**A design partner with tools and the web:**

```python
Agent(
    engine=ClaudeCodeEngine(cwd=repo, web=True, model="opus", thinking="adaptive"),
    tools=[datahub_tools, search_tools],
    name="consultant",
)
```

**A gated writer** — see [the four-tier policy](#privileges-a-four-tier-policy)
above; pair it with `CodingAgentConfig.writer(gate)` if you want Codex's
`workspace-write` sandbox on the same profile.

**Two families on one question**, for a second opinion that shares none of the
first's assumptions:

```python
verdicts = Agent.parallel(claude_reviewer, codex_reviewer, name="panel")(task)
```

## Timeouts and retries

```python
ClaudeCodeEngine(
    request_timeout=600.0,     # per-run deadline; None disables
    stream_idle_timeout=90.0,  # idle gap before stream() gives up
    max_retries=3,             # transient failures only
    tool_timeout=None,         # per-tool deadline
)
```

A real review reads several files and runs `git`; the 120 s default is sized
for a single question, not for that. When such an engine is served through an
MCP host, give the *host* a matching tool timeout (Claude Code:
`MCP_TOOL_TIMEOUT` in ms) or it cancels the call before the answer arrives.

Retries cover transient failures (network, 429/5xx, timeouts). A missing or
unreadable CLI is a permanent configuration problem and is never retried. On a
**durable** Codex thread, a turn lost after the request was sent raises
`CodexTurnUncertain` instead of retrying: `turn/start` is not idempotent, so
replaying could duplicate a turn that already ran. Resume the thread and
inspect it before deciding.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "I don't have file-writing tools" | the built-in was never granted | add `extra_tools=("Write", …)` |
| Every application tool is denied | `preapprove_application_tools=False` with no gate | configure `approval_gate`, or leave pre-approval on |
| `ValueError` naming `Bash` at construction | unconfinable tool without a gate | pass an `approval_gate` |
| File access denied | path outside `file_roots` | add the directory; do not reach for `Bash` |
| `codex` not found | not on `PATH` (the desktop install is not) | set `CODEX_BIN`, or install via npm |
| Codex answers with no cost | expected — plan auth reports none | use token counts instead |
| MCP tool calls rejected inside Codex | its `approval_policy="never"` refuses them | pass the tools via `tools=[...]` (dynamic tools) instead |

## See also

- [Claude Code Engine](claude-code-engine.md) — setup, session modes, multimodal
- [Codex Engine](codex-engine.md) — protocol notes, native review harness
- [Coding-agent permissions](coding-agent-config.md) — the profiles and the provider mapping
