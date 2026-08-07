"""Live validation of ``ClaudeCodeEngine`` through a real LazyBridge Agent."""

from __future__ import annotations

from lazybridge import Agent, ClaudeCodeEngine, Memory, Session


def lazybridge_probe() -> str:
    """Return a deterministic string for the Claude Code engine smoke test."""
    return "lazybridge-engine-ok"


def main() -> None:
    agent = Agent(
        name="claude-code-engine-smoke",
        engine=ClaudeCodeEngine(
            model="sonnet",
            system="Use the provided LazyBridge tool when the user requests its probe value.",
        ),
        # Standard LazyBridge input: Agent normalises the callable for us.
        tools=[lazybridge_probe],
        memory=Memory(),
        session=Session(),
    )
    result = agent("Call lazybridge_probe exactly once and reply only with its result.")
    print(result.text())
    print(f"provider={result.metadata.provider}")
    if result.error:
        print(f"error={result.error.type}: {result.error.message}")


if __name__ == "__main__":
    main()
