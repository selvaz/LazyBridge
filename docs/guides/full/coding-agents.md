# Coding agents in practice

Claude Code and Codex are engines inside a Python Agent. Keep the same tool and
state references when changing engines; pair different runtimes for writing and
review. Provider APIs and deterministic Plans compose through that same contract.

## Swap the engine, keep the state

```python
from lazybridge import Agent, ClaudeCodeEngine, CodexEngine, LLMEngine, Memory, Session, Store

def project_status() -> str:
    """Return a deterministic project status."""
    return "The draft is ready for review."

state = dict(tools=[project_status], memory=Memory(), store=Store(), session=Session())
claude = Agent(name="claude", engine=ClaudeCodeEngine(), **state)
codex = Agent(name="codex", engine=CodexEngine(), **state)
api = Agent(name="api", engine=LLMEngine("gpt-6.1-sol"), **state)
assert claude.memory is codex.memory is api.memory
assert claude.store is codex.store is api.store
assert claude.session is codex.session is api.session
```

Changing engines shares application state. A Claude native transcript cannot
be migrated into a Codex thread. Native tools, permissions, structured schemas
and attachments differ; engine interchangeability describes the Agent contract.

## Choose the runtime

| Capability | Claude Code | Codex |
| --- | --- | --- |
| Transport | Claude Agent SDK | codex app-server JSON-RPC |
| Python tools | Temporary in-process MCP adapter | App Server dynamic tools |
| Default native access | WebSearch/WebFetch; rooted Read/Glob/Grep | Native read-only sandbox; shell available within runtime policy |
| Writes | Explicit Write/Edit extras plus policy | workspace-write plus policy |
| Shell confinement | File roots do not confine Bash; gate required | Runtime sandbox and approvals |
| Structured output | SDK output_format for derivable schemas | Native outputSchema for representable strict object schemas; fallback otherwise |
| Reasoning | reasoning_effort plus thinking | reasoning_effort, passed through |
| Durable history | persist_session / session_id | persist_thread / thread_id |
| Named continuity | session_alias / SessionRegistry | session_alias / SessionRegistry |
| Usage telemetry | /usage prose parsing | account/rateLimits/read RPC |
| Dollar cost | SDK-reported cost | No reported price; adapter records 0.0 |

With subscription login the runtimes use plan quota. The adapters reuse the
configured runtime authentication and do not prohibit API billing.
Use supported runtime aliases/IDs: Claude defaults to sonnet; Codex defaults
to its configured model. Model availability and effort support remain runtime
concerns; no invented aliases or guaranteed low-cost ranking is implied.

## Models and reasoning choices

Claude's sonnet/opus/haiku aliases resolve through the installed runtime and
account. Full supported IDs can pin a model. See the
[model identifier table](claude-code-engine.md#claude-code-model-identifiers)
and current official docs rather than assuming a permanent family version or
price order. fallback_model is a native fallback for Claude; outer Agent
fallback is a different mechanism.

Codex model=None keeps the runtime-configured model; an explicit ID passes to
thread/start without a model/list availability check. Invalid IDs/efforts can
therefore fail at runtime. Native IDs and aliases differ from LLMEngine's
provider catalogue and LazyTools' quota-routing tiers.

| Knob | Engine | Meaning |
| --- | --- | --- |
| reasoning_effort | Both | Model effort; Claude validates its allowed strings, Codex passes through |
| thinking | Claude | adaptive, disabled, or a positive token budget |
| fallback_model | Claude | Native SDK model fallback |

```python
from lazybridge import ClaudeCodeEngine, CodexEngine

claude_default = ClaudeCodeEngine(model="sonnet")
claude_reasoned = ClaudeCodeEngine(model="opus", reasoning_effort="high", thinking="adaptive")
claude_budgeted = ClaudeCodeEngine(thinking=8_000)
codex_default = CodexEngine()
codex_reasoned = CodexEngine(reasoning_effort="medium")
```

Supported effort modes depend on the model/runtime. An integer thinking
budget and an effort label are not interchangeable units. There is no Codex
thinking argument.

## Privileges: a four-tier policy

Use allow for known permitted actions, session for a scoped remembered grant,
ask for individual approval, and deny as the default. The
[core TieredGate extension](../../reference/extensions.md#tiered-approval-gate)
implements this policy and audit trail. The
[permission guide](coding-agent-config.md) shows complete config and callback
examples. Claude writer profiles still need extra_tools; supplying a gate does
not itself expose Write/Edit/Bash.

| Tier | Behavior |
| --- | --- |
| allow | Run a known permitted action without asking |
| session | Ask once, then remember a grant within the gate's scope |
| ask | Ask for each action |
| deny | Refuse, including unmatched actions |

This complete policy uses the Core extension instead of an undefined human
callback. It configures a terminal approval channel but does not invoke writes:

```python
from pathlib import Path
from lazybridge import Agent, ClaudeCodeEngine, ClaudeCodePolicy, CodingAgentConfig
from lazybridge.ext.approval import Rule, TerminalChannel, TieredGate

repo = str(Path.cwd().resolve())
gate = TieredGate(channel=TerminalChannel(), rules=(
    Rule("allow", "Read"), Rule("allow", "Glob"), Rule("allow", "Grep"),
    Rule("allow", "WebSearch"), Rule("allow", "WebFetch"),
    Rule("session", "Write"), Rule("session", "Edit"),
    Rule("ask", "Bash"),
))
config = CodingAgentConfig(
    claude=ClaudeCodePolicy(preapprove_application_tools=False,
                            extra_tools=("Write", "Edit", "Bash")),
    approval_gate=gate,
)
writer = Agent(name="writer", engine=ClaudeCodeEngine(
    cwd=repo, file_roots=[repo], config=config,
))
```

Rules are ordered and unmatched actions deny. The broad Write/Edit session
rules above are a policy choice, not a file-specific grant; narrow rules and
argument scope for sensitive workflows. TieredGate owns provider/agent/cwd/
policy scope and structured audit records. A generic allow_for_session grant
otherwise lives on the LazyBridge Session, with action/command scope described
in the permission guide. A native transcript alone does not persist that grant.

### The confinement asymmetry

Claude file_roots constrain native file hooks, including under permissive
permission settings. Bash can touch what its process can touch; it needs a
separate gate. The constructor detects a missing gate:

```python
from pathlib import Path
from lazybridge import ClaudeCodeEngine, ClaudeCodePolicy, CodingAgentConfig

repo = str(Path.cwd().resolve())
try:
    ClaudeCodeEngine(file_roots=[repo], config=CodingAgentConfig(
        claude=ClaudeCodePolicy(extra_tools=("Bash",)),
    ))
except ValueError as exc:
    print("Gate required:", exc)
else:
    raise AssertionError("Expected unconfined-tool validation")
```

Application tools run in Python, outside the native filesystem sandbox.
Command policies must consider every compound segment, rather than approve
a whole command just because it starts with a permitted prefix.

## Durable conversations

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from lazybridge import Agent, ClaudeCodeEngine, CodexEngine, SessionRegistry

repo = str(Path.cwd().resolve())
with TemporaryDirectory() as directory:
    registry = SessionRegistry(path=Path(directory) / "sessions.json")
    claude = Agent(name="writer", engine=ClaudeCodeEngine(
        cwd=repo, session_alias="draft", session_registry=registry, tag=None,
    ))
    codex = Agent(name="reviewer", engine=CodexEngine(
        cwd=repo, session_alias="draft", session_registry=registry,
    ))
    claude("Remember the draft topic: weekend deployment.")
    codex("Remember the review topic: weekend deployment.")
    assert registry.resolve("claude", repo, "draft") is not None
    assert registry.resolve("codex", repo, "draft") is not None
```

These are two native conversations because engine kind is part of the key.
Use a persistent registry file in real work. Resumes omit repeated Memory
priming while preserving explicit context. Runs/streams serialize in-process;
separate processes require caller coordination. Explicit native IDs override
alias lookup. [SessionRegistry](session-registry.md) explains precedence,
corruption handling and registry/transcript locking.

### Keeping raw native handles

Aliases are optional. A persisted engine exposes its handle for an application
to store and resume later:

```python
from pathlib import Path
from lazybridge import Agent, ClaudeCodeEngine, CodexEngine

repo = str(Path.cwd().resolve())
claude_engine = ClaudeCodeEngine(cwd=repo, persist_session=True, tag=None)
Agent(name="claude-reviewer", engine=claude_engine)("Remember the review topic.")
claude_handle = claude_engine.session_id
assert claude_handle is not None
Agent(name="claude-followup", engine=ClaudeCodeEngine(
    cwd=repo, session_id=claude_handle, tag=None,
))("Explain the remembered topic.")

codex_engine = CodexEngine(cwd=repo, persist_thread=True)
Agent(name="codex-reviewer", engine=codex_engine)("Remember the review topic.")
codex_handle = codex_engine.thread_id
assert codex_handle is not None
Agent(name="codex-followup", engine=CodexEngine(
    cwd=repo, thread_id=codex_handle,
))("Explain the remembered topic.")
```

To resume in another process, persist the appropriate ID with its engine kind
and cwd or use SessionRegistry. Claude runtime mode is a Session-local channel,
not a substitute for a durable registry.

### Finding and retaining native history

Claude labels newly created durable sessions with tag; Codex uses thread_source
and the initialize originator. These labels identify history rather than grant
access. Follow the engine guides' read-only listing and rollout inspection
steps. Review active use before cleanup; there is no documented Core
delete-all operation, and deleting an alias does not remove native history.

## Tools and practical recipes

Mix deterministic Python tools with named specialist agents:

```python
from lazybridge import Agent, ClaudeCodeEngine, CodexEngine, CodingAgentConfig

def project_context() -> str:
    """Return the deterministic design constraints for this example."""
    return "Deploy on weekdays; require input validation and a passing test suite."

reviewer = Agent(name="reviewer", engine=CodexEngine(
    config=CodingAgentConfig.reviewer(),
    system="Report defects with evidence. Never patch files.",
))
consultant = Agent(name="consultant", engine=ClaudeCodeEngine(web=True),
                   tools=[project_context, reviewer])
print(consultant("Propose a deployment design using project_context.").text())
```

The consultant can invoke its specialist as a Tool. A fail-closed reviewer
profile keeps native access read-only and refuses unapproved application tools.
Python tool side effects still need application policy.

For two independent answers on the same question, use a parallel panel:

```python
from lazybridge import Agent, ClaudeCodeEngine, CodexEngine, CodingAgentConfig

claude_reviewer = Agent(name="claude-review", engine=ClaudeCodeEngine(
    config=CodingAgentConfig.reviewer(), system="Review the policy's edge cases.",
))
codex_reviewer = Agent(name="codex-review", engine=CodexEngine(
    config=CodingAgentConfig.reviewer(), system="Review the policy's edge cases.",
))
panel = Agent.parallel(claude_reviewer, codex_reviewer, name="panel")
result = panel("Review a rule allowing deployments only on weekdays.")
assert "[claude-review]" in result.text() and "[codex-review]" in result.text()
```

This is a labelled join of separate answers, not a consensus or acceptance
record. Use run_branches when you need individual Envelopes. A gated writer's
complete policy is above; native repository review targets are in the Codex guide.

## Composition and collaboration

Both engines work as child agents in tools, Agent.chain, Agent.parallel and
Plan steps. A producer can have an opposite-engine verify judge, or a pipeline
can wrap only its critical step with verify. Start with
[Claude Code writes, Codex reviews](claude-writes-codex-reviews.md) for runnable
in-process review and credential-free feedback/exhaustion tests.

verify is a bounded repair loop, not release acceptance. Deterministic tests,
schema checks and explicit acceptance records must gate shipping.

## Integration through LazyTools

[LazyTools](https://tools.lazybridge.com/) owns connectors, code-support tools,
the MCP server, repository code bridge, projects and quota routing.
Its distribution is lazytoolkit and import namespace lazytools.
Do not import removed lazybridge.ext.mcp/gateway or external_tools modules.

- [Code support](https://tools.lazybridge.com/code-support/): synchronous codex/claude ask, write, review and named-session tools; the caller blocks for the delegated turn.
- [MCP server](https://tools.lazybridge.com/mcp-server/): opt in to coding providers and configure per-call model/effort/session overrides there.
- [Code bridge](https://tools.lazybridge.com/code-bridge/): detached repository jobs, durable tickets, result/evidence and tier-based --review-of routing that excludes the producer engine.
- [Projects](https://tools.lazybridge.com/projects/): Store-backed plans, claims and verification records.
- [Quota routing](https://tools.lazybridge.com/code-bridge/#choose-a-model-from-quota): deterministic eligibility ladder; benchmark in progress.

Core approval/delegation hooks remain available for applications building their
own orchestration. Since 1.1.0, native output/attachments/usage, policy controls,
durable aliases/locking and admission/job lifecycle hooks have expanded the
engine workflow. [Changelog](../../changelog.md) tracks the release sequence.

| Release | Coding-engine and collaboration changes |
| --- | --- |
| 1.1.0 | Claude Code and Codex engines behind the Agent contract. |
| 1.2.0 | Claude extra_tools, durable-session tagging and Codex thread_source labels. |
| 1.2.1 | Native Codex structured schemas, per-agent compaction, Claude usage parsing and shared bounded tool execution. |
| 1.3.0 | Codex web_search overrides and TieredGate. |
| 1.4.0 | Claude native tool events; structured step outputs delivered correctly as context. |
| 1.5.0 | Generic delegation/knowledge/approval queues, Codex usage RPC and portable approval_gate on LLMEngine. |
| 1.6.0–1.6.2 | Durable aliases/SessionRegistry, writable roots and command-scope policy; stream/alias locks and transport diagnostics. |
| 1.7.0 | Durable late approval consumption, wait_budget/expiry and admission/job provenance hooks. |
| 1.8.0 and subsequent main | Writer alias/factory/lifecycle overrides, JobRunner/JobContext, reservation release/refund and exact structured delegation outcomes; improved Codex executable discovery and pre-turn transport retry. |

## Timeouts and failures

Default engine request_timeout is 120 seconds for the whole run, retries
included. Set a larger deadline for repository tasks and align an MCP host's
separate tool timeout. Missing/unreadable CLI is permanent. Codex distinguishes
pre-turn transport failure from CodexTurnUncertain after a durable turn was
sent; inspect native history before replaying possible side effects.
Neither engine provides cross-process transcript locking or automatic recovery
of an uncertain repository turn.

### Troubleshooting

| Symptom | Likely boundary | Next check |
| --- | --- | --- |
| Claude says it has no writing tool | Built-in exposure | extra_tools, not only writer(gate) |
| Application tools all deny | Gate/preapproval | Select an explicit policy and inspect normalized requests |
| Bash rejected at construction | No unconfined-tool gate | Configure a deliberate approval rule |
| File access denied | Native roots/sandbox | Inspect resolved paths and engine policy |
| Codex CLI missing | Executable discovery | Set CODEX_BIN or follow the engine setup guide |
| Codex dollar cost is zero | RPC reports no price | Inspect tokens/quota; do not interpret zero as free |
| A runtime MCP call is refused | Runtime-specific registration/approval | Python tools use tools/dynamic callbacks; external MCP follows LazyTools docs |
| Follow-up forgets history | Ephemeral call or wrong alias/cwd/kind | Inspect persistence and native ID before resuming |
| Review returns but is not approved | Bounded verify exhaustion | Run deterministic acceptance checks and inspect review evidence |

See [Claude Code](claude-code-engine.md) and [Codex](codex-engine.md) for full
constructor tables, structured schema limits, streaming and multimodal behavior.
