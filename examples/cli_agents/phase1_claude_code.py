"""Phase 1 — Claude Code as a LazyBridge tool.

Minimal example: give claude_code to an agent and ask it to inspect the
current directory.

Prerequisites:
  pip install lazybridge lazytoolkit
  # Claude Code CLI in PATH, authenticated (claude.ai subscription or ANTHROPIC_API_KEY)
"""

from lazybridge import Agent, LLMEngine
from lazytools.connectors.cli_agents import check_clis_available, claude_code

available = check_clis_available()
if not available["claude"]:
    raise SystemExit("Install Claude Code CLI first: https://docs.anthropic.com/claude-code")

agent = Agent(
    engine=LLMEngine(
        "claude-opus-4-8",
        # tool_timeout=None: let subprocess.run handle the timeout;
        # avoids the zombie-process hazard when the engine fires first.
        tool_timeout=None,
        system="You are a code analysis expert. Use claude_code to inspect the codebase.",
    ),
    tools=[claude_code],
)

result = agent("Count Python files in the current directory and summarise each one in one sentence.")
print(result.text())
