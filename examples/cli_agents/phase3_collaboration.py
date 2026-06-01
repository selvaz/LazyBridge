"""Phase 3 — Claude Code + Codex collaboration pipeline.

Four-step Plan:
  1. claude_analyst  — reads + analyses using Claude Code (mode='read')
  2. codex_analyst   — reads + critiques using Codex (mode='read'), sees claude's analysis
  3. synthesizer     — merges the two analyses into a concrete implementation plan
  4. executor        — implements the plan using Claude Code (mode='write')

Why Plan (not AgentPool)?
  The flow is fixed and sequential. Plan with from_step is simpler, more
  predictable, and each step frees memory before the next one starts.
  Use AgentPool only when routing is dynamic or you need multi-round dialogue.

Prerequisites:
  pip install lazybridge lazytoolkit
  # Both CLIs in PATH and authenticated
"""

from lazybridge import Agent, LLMEngine, Memory, Plan, Step, from_step
from lazybridge.dedup_guard import DeduplicateGuard
from lazytools.connectors.cli_agents import check_clis_available, claude_code, codex

# ─── startup check ───────────────────────────────────────────────────────────

available = check_clis_available()
if not all(available.values()):
    missing = [k for k, v in available.items() if not v]
    raise SystemExit(f"Missing CLIs: {missing}")

# ─── shared state ───────────────────────────────────────────────────────────

# claude_analyst writes here; codex_analyst reads it via sources=.
# Safe because Plan steps are strictly sequential (no parallel execution here).
dialogue = Memory(strategy="summary")

# ─── step 1: Claude Code analyses ──────────────────────────────────────────────

claude_analyst = Agent(
    name="claude_analyst",
    engine=LLMEngine(
        "claude-opus-4-8",
        tool_timeout=None,
        system=(
            "Analyse the task using claude_code in mode='read'. "
            "Propose a concrete implementation approach. Be concise."
        ),
    ),
    tools=[claude_code],
    memory=dialogue,
    guard=DeduplicateGuard(verbose=False),
)

# ─── step 2: Codex critiques ────────────────────────────────────────────────

codex_analyst = Agent(
    name="codex_analyst",
    engine=LLMEngine(
        "gpt-5.4",
        tool_timeout=None,
        system=(
            "Analyse the task using codex in mode='read'. "
            "Critique or confirm claude_analyst's approach. Be concise."
        ),
    ),
    tools=[codex],
    sources=[dialogue],  # receives claude's analysis as context
    guard=DeduplicateGuard(verbose=False),
)

# ─── step 3: synthesizer ───────────────────────────────────────────────────

synthesizer = Agent(
    name="synthesizer",
    engine=LLMEngine(
        "claude-opus-4-8",
        system=(
            "You receive two code analyses (Claude Code and Codex). "
            "Produce a single, concrete, step-by-step implementation plan."
        ),
    ),
)

# ─── step 4: executor implements ──────────────────────────────────────────────

executor = Agent(
    name="executor",
    engine=LLMEngine(
        "claude-opus-4-8",
        tool_timeout=None,
        system="Implement the plan using claude_code in mode='write'.",
    ),
    tools=[claude_code],
)

# ─── pipeline ─────────────────────────────────────────────────────────────────

pipeline = Agent(
    name="pipeline",
    engine=Plan(
        Step("claude_analyst"),
        Step("codex_analyst", context=from_step("claude_analyst")),
        Step("synthesizer", context=from_step("codex_analyst")),
        Step("executor", context=from_step("synthesizer")),
    ),
    tools=[claude_analyst, codex_analyst, synthesizer, executor],
)

result = pipeline("Add rate limiting to the /api/login endpoint")
print(result.text())
