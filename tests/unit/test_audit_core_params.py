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


# ---------------------------------------------------------------------------
# Google — tool_choice mapping + single merged ToolConfig
# ---------------------------------------------------------------------------


def _bare_google():
    from lazybridge.core.providers.google import GoogleProvider

    p = GoogleProvider.__new__(GoogleProvider)
    p.api_key = "fake"
    p.model = "gemini-2.5-flash"
    p._user_model = "gemini-2.5-flash"
    p.fallback_model = None
    p.strict_native_tools = False
    return p


def _google_config_kwargs(request, gtypes):
    """Run _build_config with a stubbed SDK and return GenerateContentConfig kwargs."""
    from unittest.mock import patch

    from lazybridge.core.providers import google as google_module

    p = _bare_google()
    with patch.object(google_module, "_gtypes", gtypes):
        p._build_config(request)
    return gtypes.GenerateContentConfig.call_args.kwargs


@pytest.mark.parametrize(
    ("choice", "expected_mode", "expected_allowed"),
    [
        ("auto", "AUTO", None),
        ("required", "ANY", None),
        ("any", "ANY", None),
        ("none", "NONE", None),
        ("get_weather", "ANY", ["get_weather"]),
    ],
)
def test_google_tool_choice_maps_to_function_calling_config(choice, expected_mode, expected_allowed):
    """tool_choice was completely ignored by GoogleProvider before the fix."""
    from unittest.mock import MagicMock

    gtypes = MagicMock()
    req = CompletionRequest(messages=[Message.user("hi")], tools=[_TOOL], tool_choice=choice)
    kwargs = _google_config_kwargs(req, gtypes)

    assert "tool_config" in kwargs
    fcc_kwargs = gtypes.FunctionCallingConfig.call_args.kwargs
    assert fcc_kwargs["mode"] == expected_mode
    if expected_allowed is None:
        assert "allowed_function_names" not in fcc_kwargs
    else:
        assert fcc_kwargs["allowed_function_names"] == expected_allowed


def test_anthropic_skills_are_serialised_into_container():
    """request.skills only set beta headers before the fix — the skills list
    never reached the API payload (silent no-op)."""
    from lazybridge.core.types import SkillsConfig

    p = _bare_anthropic()
    req = CompletionRequest(
        messages=[Message.user("make a deck")],
        skills=SkillsConfig(skills=["powerpoint", "pdf", "skill_abc123"]),
    )
    params = p._build_params(req)

    assert params["container"] == {
        "skills": [
            {"type": "anthropic", "skill_id": "pptx", "version": "latest"},
            {"type": "anthropic", "skill_id": "pdf", "version": "latest"},
            {"type": "custom", "skill_id": "skill_abc123", "version": "latest"},
        ]
    }
    # Skills require the code-execution container tool.
    assert any(t.get("name") == "code_execution" for t in params["tools"])


def test_google_tool_config_merges_maps_and_function_calling():
    """Maps lat/lng (retrieval_config) must not cancel function_calling_config."""
    from unittest.mock import MagicMock

    from lazybridge.core.types import NativeTool

    gtypes = MagicMock()
    req = CompletionRequest(
        messages=[Message.user("hi")],
        tools=[_TOOL],
        tool_choice="auto",
        native_tools=[NativeTool.GOOGLE_MAPS],
        extra={"google_maps_lat": 41.9, "google_maps_lng": 12.5},
    )
    kwargs = _google_config_kwargs(req, gtypes)

    tc_kwargs = gtypes.ToolConfig.call_args.kwargs
    assert "function_calling_config" in tc_kwargs
    assert "retrieval_config" in tc_kwargs  # merged, not overwritten
