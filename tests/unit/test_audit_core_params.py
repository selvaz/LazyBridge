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


# ---------------------------------------------------------------------------
# OpenAI — chat/responses param parity and contract fixes
# ---------------------------------------------------------------------------


def _bare_openai():
    from lazybridge.core.providers.openai import OpenAIProvider

    p = OpenAIProvider.__new__(OpenAIProvider)
    p.api_key = "fake"
    p.model = "gpt-4o-mini"
    p._user_model = "gpt-4o-mini"
    p.fallback_model = None
    return p


def test_openai_chat_reasoning_effort_only_on_reasoning_models():
    import warnings as w

    from lazybridge.core.types import ThinkingConfig

    p = _bare_openai()
    req = CompletionRequest(messages=[Message.user("x")], model="gpt-4o", thinking=ThinkingConfig())
    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        params = p._build_chat_params(req)
    assert "reasoning_effort" not in params
    assert any("not a reasoning model" in str(x.message) for x in caught)

    req2 = CompletionRequest(messages=[Message.user("x")], model="o3-mini", thinking=ThinkingConfig())
    params2 = p._build_chat_params(req2)
    assert params2["reasoning_effort"]


def test_openai_responses_temperature_not_dropped_by_thinking():
    from lazybridge.core.types import ThinkingConfig

    p = _bare_openai()
    req = CompletionRequest(
        messages=[Message.user("x")], model="gpt-4o", temperature=0.3, thinking=ThinkingConfig()
    )
    params = p._build_responses_params(req)
    # Non-reasoning model: temperature must survive even with thinking set.
    assert params["temperature"] == 0.3

    req2 = CompletionRequest(messages=[Message.user("x")], model="o3-mini", temperature=0.3)
    params2 = p._build_responses_params(req2)
    # Reasoning model: temperature is rejected by the API — omit it.
    assert "temperature" not in params2


def test_openai_system_history_messages_are_forwarded():
    from lazybridge.core.types import Role

    p = _bare_openai()
    req = CompletionRequest(
        messages=[Message.system("inline rule"), Message.user("hi")],
        system="top-level prompt",
        model="gpt-4o-mini",
    )
    chat = p._messages_to_openai(req)
    assert [m["content"] for m in chat if m["role"] == "system"] == ["top-level prompt", "inline rule"]
    items = p._messages_to_responses_input(req)
    assert [m["content"] for m in items if m.get("role") == "system"] == ["top-level prompt", "inline rule"]


def test_openai_chat_path_warns_on_dropped_native_tools():
    import warnings as w

    from lazybridge.core.types import NativeTool

    p = _bare_openai()
    p.strict_native_tools = False
    req = CompletionRequest(
        messages=[Message.user("x")], model="gpt-4o-mini", native_tools=[NativeTool.WEB_SEARCH]
    )
    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        p._build_chat_params(req)
    assert any("Responses API" in str(x.message) for x in caught)


def test_deepseek_strict_native_tools_raises_on_chat_path():
    from lazybridge.core.providers.base import UnsupportedNativeToolError
    from lazybridge.core.providers.deepseek import DeepSeekProvider
    from lazybridge.core.types import NativeTool

    p = DeepSeekProvider.__new__(DeepSeekProvider)
    p.api_key = "fake"
    p.model = "deepseek-v4-flash"
    p.strict_native_tools = True
    req = CompletionRequest(
        messages=[Message.user("x")], model="deepseek-v4-flash", native_tools=[NativeTool.WEB_SEARCH]
    )
    with pytest.raises(UnsupportedNativeToolError):
        p._build_chat_params(req)


def test_openai_responses_failed_raises():
    from types import SimpleNamespace as NS

    p = _bare_openai()
    failed = NS(status="failed", error=NS(message="overloaded"), output=[], usage=None, model="gpt-4o")
    with pytest.raises(RuntimeError, match="overloaded"):
        p._parse_responses_response(failed)


def test_openai_reasoning_tokens_from_responses_usage_shape():
    from types import SimpleNamespace as NS

    from lazybridge.core.providers.openai import OpenAIProvider
    from lazybridge.core.types import UsageStats

    usage = UsageStats()
    raw = NS(output_tokens_details=NS(reasoning_tokens=42))
    raw.completion_tokens_details = None
    out = OpenAIProvider._populate_reasoning_tokens(usage, raw)
    assert out.thinking_tokens == 42


def test_merge_streamed_tool_calls_dedupes_same_id_across_indices():
    from lazybridge.core.providers.openai import _merge_streamed_tool_calls

    accum = {
        0: {"id": "call_1", "name": "f", "args": '{"a"'},
        1: {"id": "call_1", "name": "", "args": ": 1}"},
    }
    calls = _merge_streamed_tool_calls(accum)
    assert len(calls) == 1
    assert calls[0].arguments == {"a": 1}


def test_lmstudio_prefixed_tier_alias_resolves():
    from lazybridge.core.providers.lmstudio import LMStudioProvider

    p = LMStudioProvider.__new__(LMStudioProvider)
    p.api_key = "fake"
    p.model = None
    p._user_model = None
    p.fallback_model = None
    req = CompletionRequest(messages=[Message.user("x")], model="lmstudio/cheap")
    assert p._resolve_model(req) == p._TIER_ALIASES["cheap"]
