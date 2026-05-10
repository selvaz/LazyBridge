"""Structured output via Pydantic — ``output_type=`` gives ``result.payload``
the typed shape the agent should return.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from lazybridge.testing import MockAgent


class Summary(BaseModel):
    title: str
    bullets: list[str]


def main() -> None:
    # MockAgent emits the JSON string verbatim; a real LLMEngine with
    # ``output_type=Summary`` would parse it into the Pydantic model
    # automatically and expose it on ``result.payload``.  We show both
    # paths so the production pattern is clear.
    mock_payload = Summary(
        title="LazyBridge in 30 seconds",
        bullets=["Agent = Engine + Tools + State", "Tools auto-wrap", "Errors always raise"],
    )
    agent = MockAgent([json.dumps(mock_payload.model_dump())], name="summarizer", output=Summary)

    result = agent("Summarise LazyBridge.")
    print("result.text():", result.text()[:60], "…")

    # In a real Agent(output_type=Summary, engine=LLMEngine(...)) the
    # next line would already be a Summary instance.  With MockAgent
    # we parse manually to show the contract.
    summary = Summary.model_validate_json(result.text())
    print("summary.title:", summary.title)
    print("summary.bullets:", summary.bullets)


if __name__ == "__main__":
    main()
