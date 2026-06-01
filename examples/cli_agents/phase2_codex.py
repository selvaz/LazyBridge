"""Phase 2 — Codex as a LazyBridge tool.

Minimal example: give codex to an agent and ask it to inspect source code.

Prerequisites:
  pip install lazybridge lazytoolkit
  # Codex CLI in PATH, authenticated via `codex login`
"""

from lazybridge import Agent, LLMEngine
from lazytools.connectors.cli_agents import check_clis_available, codex

available = check_clis_available()
if not available["codex"]:
    raise SystemExit("Install Codex CLI first: https://github.com/openai/codex")

agent = Agent(
    engine=LLMEngine(
        "gpt-5.4",
        tool_timeout=None,
        system="You are a code analysis expert. Use codex to inspect the codebase.",
    ),
    tools=[codex],
)

result = agent("List all public functions in main.py and describe what each one does.")
print(result.text())
