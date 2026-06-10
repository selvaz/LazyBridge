"""Regression tests for core-audit fixes on request-parameter building.

Covers provider-side translation of the unified CompletionRequest into
wire params — no SDK calls, providers are instantiated via ``__new__``.
"""

from __future__ import annotations

import pytest

from lazybridge.core.providers.anthropic import AnthropicProvider
from lazybridge.core.types import CompletionRequest, Message, ToolDefinition


def _bare_anthropic() -> AnthropicProvider:
    p = AnthropicProvider.__new__(AnthropicProvider)
    p.api_key = "sk-test"
    p._user_model = "claude-sonnet-4-6"
    p.model = "claude-sonnet-4-6"
    p.fallback_model = None
    return p


_TOOL = ToolDefinition(
    name="get_weather",
    description="Weather lookup.",
    parameters={"type": "object", "properties": {}},
)


# ---------------------------------------------------------------------------
# Anthropic tool_choice — the API only accepts auto / any / tool / none.
# Before the fix, "required" and "any" both produced {"type": "required"},
# which does not exist and 400s server-side.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ("auto", {"type": "auto"}),
        ("none", {"type": "none"}),
        ("required", {"type": "any"}),
        ("any", {"type": "any"}),
        ("get_weather", {"type": "tool", "name": "get_weather"}),
    ],
)
def test_anthropic_tool_choice_maps_to_valid_api_types(choice, expected):
    p = _bare_anthropic()
    req = CompletionRequest(messages=[Message.user("hi")], tools=[_TOOL], tool_choice=choice)
    params = p._build_params(req)
    assert params["tool_choice"] == expected


def test_anthropic_tool_choice_without_tools_is_not_emitted():
    """A dangling tool_choice with no tools param is a guaranteed 400."""
    p = _bare_anthropic()
    req = CompletionRequest(messages=[Message.user("hi")], tool_choice="auto")
    params = p._build_params(req)
    assert "tool_choice" not in params
