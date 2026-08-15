"""Live validation of ``CodexEngine`` through a real LazyBridge Agent."""

from __future__ import annotations

from lazybridge import Agent, CodexEngine, Memory, Session


def lazybridge_probe() -> str:
    """Return a deterministic string for the Codex engine smoke test."""
    return "lazybridge-engine-ok"


def main() -> None:
    agent = Agent(
        name="codex-engine-smoke",
        engine=CodexEngine(
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
    print(
        f"input_tokens={result.metadata.input_tokens} output_tokens={result.metadata.output_tokens} cost_usd={result.metadata.cost_usd}"
    )
    if result.error:
        print(f"error={result.error.type}: {result.error.message}")


if __name__ == "__main__":
    main()
