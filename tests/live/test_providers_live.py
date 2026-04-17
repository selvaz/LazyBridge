"""Opt-in live smoke tests for real model providers."""

from __future__ import annotations

import pytest

from lazybridge import LazyAgent, LazyTool
from tests.live.helpers import live_model, require_live_provider

LIVE_PROMPT = (
    "You MUST call the add_numbers tool with a=24681 and b=13579. "
    "Do NOT compute the result yourself — only the tool's return value is accepted. "
    "After receiving the tool result, reply with exactly: RESULT=<tool_result>"
)


def _run_live_tool_smoke(provider: str) -> None:
    require_live_provider(provider)

    calls: list[tuple[int, int]] = []

    def add_numbers(a: int, b: int) -> int:
        """Add two integers and return the result."""
        calls.append((a, b))
        return a + b

    agent = LazyAgent(provider, model=live_model(provider))
    tool = LazyTool.from_function(add_numbers)

    response = agent.loop(
        LIVE_PROMPT,
        tools=[tool],
        max_tokens=128,
    )

    assert calls == [(24681, 13579)], "Tool was never called — model answered without using the tool"
    assert "38260" in response.content
    assert response.model
    assert response.stop_reason == "end_turn"
    assert response.usage.input_tokens >= 0
    assert response.usage.output_tokens >= 0


@pytest.mark.live
def test_openai_live_tool_loop_smoke():
    _run_live_tool_smoke("openai")


@pytest.mark.live
def test_anthropic_live_tool_loop_smoke():
    _run_live_tool_smoke("anthropic")


@pytest.mark.live
def test_google_live_tool_loop_smoke():
    _run_live_tool_smoke("google")
