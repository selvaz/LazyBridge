# Claude Code writes, Codex reviews

Use different engines for production and review while keeping the same Agent
contract. This guide starts with draft text in one Python process. Repository
changes use native write permissions and the LazyTools code bridge.

## In-process verify

Install and authenticate both runtimes using the
[Claude Code](claude-code-engine.md#setup) and [Codex](codex-engine.md#setup) guides.

```python
from lazybridge import Agent, ClaudeCodeEngine, CodexEngine, Session

session = Session()
reviewer = Agent(
    name="codex-reviewer",
    engine=CodexEngine(system=(
        'Start with "approved" or "rejected: <reason>". '
        "Check that every weekday boundary is covered. Return actionable feedback."
    )),
    session=session,
)
writer = Agent(
    name="claude-writer",
    engine=ClaudeCodeEngine(system="Draft Python code and explain the boundary cases."),
    session=session,
    verify=reviewer,
    max_verify=3,
)
result = writer("Draft a function can_deploy(day: int) using Monday=0 through Sunday=6. "
                "Allow weekdays only and reject invalid day numbers.")
print(result.text())
```

verify belongs on construction, not on run(). The judge Agent receives the
candidate text and original task. Rejection supplies context for a fresh
attempt on the same original task. max_verify=3 allows three total writer
attempts, each reviewed. Swap the two engine constructors to reverse the roles.

The core does not force reviewer independence: passing the same engine as
judge is legal. This example explicitly selects Codex to review Claude.

## Exercise feedback without credentials

This standalone example adapts MockAgent to the engine contract and uses real Agent verification.
It runs without SDKs, accounts or a subprocess:

```python
from lazybridge import Agent, Session
from lazybridge.testing import MockAgent

class DraftEngine:
    def __init__(self, responses):
        self.mock = MockAgent(responses, name="drafts")

    async def run(self, env, **kwargs):
        return await self.mock.run(env)

session = Session()
reviewer = MockAgent([
    "rejected: Sunday must be blocked.",
    "approved",
], name="mock-reviewer")
writer = Agent(
    name="mock-writer",
    engine=DraftEngine(["First draft", "Corrected draft"]),
    session=session,
    verify=reviewer,
    max_verify=3,
)
result = writer("Draft a weekday deployment rule.")
assert result.text() == "Corrected draft"
print(result.text())
```

The verdict parser accepts boolean results from synchronous callables and
approval/rejection prefixes (approved, accept, allow, pass, ok, yes, good, valid;
rejected, deny, block, fail, no, bad, invalid), case-insensitively after supported
Markdown/label cleanup. Unknown verdicts reject. An Agent judge sees output
plus original task; a callable judge receives only output text.

## Exhaustion and deterministic acceptance

```python
from lazybridge import Agent
from lazybridge.testing import MockAgent

class DraftEngine:
    def __init__(self, responses):
        self.mock = MockAgent(responses, name="drafts")

    async def run(self, env, **kwargs):
        return await self.mock.run(env)

writer = Agent(
    name="exhausted-writer",
    engine=DraftEngine(["Draft A", "Draft B"]),
    verify=lambda text: "rejected: missing boundary tests",
    max_verify=2,
)
result = writer("Draft the deployment rule.")
assert result.text() == "Draft B"  # Returned despite the second rejection.
```

No approval flag is added to the returned Envelope. result.ok means execution
succeeded, not that review accepted it. A release gate must explicitly check
tests/acceptance evidence. For the deployment function, deterministic checks
should cover all seven day numbers and invalid inputs; do not execute generated
code from arbitrary text without a controlled repository/test environment.
For project workflows, record acceptance against the
[LazyTools project contract](https://tools.lazybridge.com/projects/).

## Continue by name

Claude persist_session/session_id and Codex persist_thread/thread_id retain
native history. Both accept session_alias with SessionRegistry. Give producer
and reviewer separate aliases, such as deployment-writer and deployment-reviewer.
Resumes omit repeated Memory priming but preserve explicit context.
[SessionRegistry](session-registry.md) documents cross-process resume, aliases
and locking boundaries. Native history cannot move between engines.

## Repository work through LazyTools

For a repository patch, follow the
[LazyTools code bridge workflow](https://tools.lazybridge.com/code-bridge/)
rather than treating a verified text draft as a reviewed diff:

1. Launch a detached Claude writer with a bounded brief and repository cwd.
   Use a named native session for follow-ups.
2. Poll pending approval tickets and resolve requested actions through the
   configured operator policy.
3. Wait for the writer result and inspect its diff, checks and workspace evidence.
4. Start tier-based review using --review-of with the writer job ID. That routing
   excludes the producing engine and refuses if no opposite reviewer is eligible.
5. Inspect findings and deterministic tests; record acceptance before release.
   If revision is needed, continue the writer session and review the new result.

The bridge provides durable jobs/tickets/results, cancellation and best-effort
workspace evidence; it does not automatically retry or resume timed-out turns.
Evidence covers the entire repository, including pre-existing changes.
See LazyTools for CLI arguments, installation, operator policy and quota routing;
the quota routing benchmark is in progress.

For shorter synchronous calls, use
[LazyTools code-support MCP tools](https://tools.lazybridge.com/code-support/)
(codex_write, claude_code_write and review/ask/session tools). The caller remains
blocked until the turn completes. Coding providers explicitly opt in through the
[MCP server](https://tools.lazybridge.com/mcp-server/); exposing mutating tools
with --allow-unsafe is distinct from the bridge's durable approval queue.
