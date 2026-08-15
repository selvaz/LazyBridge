"""Live check that ``output=<model>`` is enforced natively, not by prompting.

``ClaudeCodeEngine`` passes the derived JSON schema to the Agent SDK's
``output_format`` (the CLI's ``--json-schema``), so the final message is
constrained server-side and comes back already parsed on
``ResultMessage.structured_output`` — same guarantee ``LLMEngine`` gets from
``StructuredOutputConfig``.

Requires a locally authenticated ``claude`` CLI:

    .venv\\Scripts\\python.exe examples\\claude_code\\live_structured_output_smoke.py
"""

from __future__ import annotations

from pydantic import BaseModel

from lazybridge import Agent, ClaudeCodeEngine


class Leg(BaseModel):
    venue: str


class Quote(BaseModel):
    symbol: str
    price: float
    note: str | None = None
    legs: list[Leg] = []


def main() -> None:
    agent = Agent(
        name="structured-output-smoke",
        engine=ClaudeCodeEngine(model="sonnet"),
        output=Quote,
    )
    result = agent("The symbol is AMZN and its price is 123.45, traded on NASDAQ. Report it.")
    print(f"payload={result.payload!r}")
    print(f"type={type(result.payload).__name__}")
    if result.error:
        print(f"error={result.error.type}: {result.error.message}")


if __name__ == "__main__":
    main()
