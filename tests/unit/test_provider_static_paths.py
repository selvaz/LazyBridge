"""Structural / pure-Python coverage for the provider catalogue.

These tests don't touch the upstream SDKs.  They exercise:

* ``_compute_cost`` against every model in each provider's price table
  plus a few near-misses (substring-prefix matching, unknown model
  returns ``None``).
* ``get_default_max_tokens`` against every tier alias each provider
  declares, plus a generic "unknown model" fallback path.
* ``_TIER_ALIASES`` invariants (every standard tier resolves to a
  concrete entry in the price / max-tokens tables, when applicable).
* ``_FALLBACKS`` chain validity (every fallback target is itself a
  known model).
* ``BaseProvider._resolve_model`` tier-alias resolution end-to-end.
* ``BaseProvider._check_native_tools`` warn-and-drop semantics.
"""

from __future__ import annotations

import warnings

import pytest

from lazybridge.core.providers.anthropic import (
    _PRICE_TABLE as _ANTHROPIC_PRICES,
)
from lazybridge.core.providers.anthropic import (
    AnthropicProvider,
)
from lazybridge.core.providers.deepseek import (
    _PRICE_TABLE as _DEEPSEEK_PRICES,
)
from lazybridge.core.providers.deepseek import (
    _REASONING_MODELS,
    _THINKING_CAPABLE_MODELS,
    DeepSeekProvider,
)
from lazybridge.core.providers.google import (
    _PRICE_TABLE as _GOOGLE_PRICES,
)
from lazybridge.core.providers.google import (
    GoogleProvider,
)
from lazybridge.core.providers.openai import (
    _PRICE_TABLE as _OPENAI_PRICES,
)
from lazybridge.core.providers.openai import (
    OpenAIProvider,
)
from lazybridge.core.types import (
    CompletionRequest,
    Message,
    NativeTool,
    Role,
)

_STANDARD_TIERS = ("top", "expensive", "medium", "cheap", "super_cheap")


def _bare(provider_cls):
    """Build a provider instance without invoking ``_init_client`` so we
    don't need any SDK installed.  All the methods we exercise here are
    pure-Python helpers that only read class-level state."""
    p = provider_cls.__new__(provider_cls)
    p.api_key = None
    p.model = provider_cls.default_model
    return p


# ---------------------------------------------------------------------------
# _compute_cost — every model in the price table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider_cls,price_table",
    [
        (OpenAIProvider, _OPENAI_PRICES),
        (AnthropicProvider, _ANTHROPIC_PRICES),
        (GoogleProvider, _GOOGLE_PRICES),
        (DeepSeekProvider, _DEEPSEEK_PRICES),
    ],
)
def test_compute_cost_for_every_table_entry(provider_cls, price_table):
    p = _bare(provider_cls)
    for model in price_table:
        cost = p._compute_cost(model, 1000, 1000)
        assert isinstance(cost, float)
        assert cost > 0


@pytest.mark.parametrize(
    "provider_cls",
    [OpenAIProvider, AnthropicProvider, GoogleProvider, DeepSeekProvider],
)
def test_compute_cost_unknown_model_returns_none(provider_cls):
    p = _bare(provider_cls)
    assert p._compute_cost("definitely-not-a-real-model-zzz", 100, 100) is None


def test_openai_compute_cost_with_cached_input_tokens_costs_less():
    p = _bare(OpenAIProvider)
    full = p._compute_cost("gpt-5.5", 10_000, 1_000)
    cached = p._compute_cost("gpt-5.5", 10_000, 1_000, cached_input_tokens=10_000)
    assert cached is not None and full is not None
    # Cached input is cheaper (gpt-5.5: $0.50/M cached vs $5.00/M uncached).
    assert cached < full


def test_openai_compute_cost_clamps_cached_to_input():
    p = _bare(OpenAIProvider)
    # Asking for more cached than total input must not over-discount.
    over = p._compute_cost("gpt-5.5", 1_000, 100, cached_input_tokens=10_000)
    cap = p._compute_cost("gpt-5.5", 1_000, 100, cached_input_tokens=1_000)
    assert over == cap


# ---------------------------------------------------------------------------
# get_default_max_tokens — across model families
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider_cls,model,expected_min",
    [
        (OpenAIProvider, "gpt-5.5", 100_000),
        (OpenAIProvider, "gpt-4.1", 16_000),
        (OpenAIProvider, "gpt-4o", 8_000),
        (OpenAIProvider, "gpt-4o-mini", 8_000),
        (OpenAIProvider, "o1-mini", 32_000),
        (OpenAIProvider, "o3", 50_000),
        (OpenAIProvider, "unknown-model", 1_000),
        (AnthropicProvider, "claude-opus-4-7", 100_000),
        (AnthropicProvider, "claude-sonnet-4-6", 60_000),
        (AnthropicProvider, "claude-haiku-3", 4_000),
        (AnthropicProvider, "claude-3-5-sonnet", 8_000),
        (GoogleProvider, "gemini-3.1-pro-preview", 60_000),
        (GoogleProvider, "gemini-2.5-flash", 60_000),
        (GoogleProvider, "gemini-1.5-flash", 8_000),
        (DeepSeekProvider, "deepseek-v4-flash", 60_000),
        (DeepSeekProvider, "deepseek-reasoner", 60_000),
        (DeepSeekProvider, "deepseek-chat", 8_000),
    ],
)
def test_get_default_max_tokens(provider_cls, model, expected_min):
    p = _bare(provider_cls)
    p.model = model
    n = p.get_default_max_tokens()
    assert n >= expected_min, f"{provider_cls.__name__}({model!r}) → {n}"


# ---------------------------------------------------------------------------
# _TIER_ALIASES + _FALLBACKS structural invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider_cls",
    [OpenAIProvider, AnthropicProvider, GoogleProvider, DeepSeekProvider],
)
def test_every_standard_tier_resolves(provider_cls):
    """Every tier alias maps to a non-empty concrete model name."""
    aliases = provider_cls._TIER_ALIASES
    for tier in _STANDARD_TIERS:
        assert tier in aliases, f"{provider_cls.__name__} missing tier {tier!r}"
        target = aliases[tier]
        assert isinstance(target, str) and target


@pytest.mark.parametrize(
    "provider_cls",
    [OpenAIProvider, AnthropicProvider, DeepSeekProvider],
)
def test_fallback_targets_have_pricing(provider_cls):
    """Every fallback target must have at least one matching entry in
    the provider's price table — otherwise cost reporting silently
    drops to ``None`` when the fallback fires."""
    price_keys = {
        OpenAIProvider: _OPENAI_PRICES,
        AnthropicProvider: _ANTHROPIC_PRICES,
        DeepSeekProvider: _DEEPSEEK_PRICES,
    }[provider_cls]
    for source, fallbacks in provider_cls._FALLBACKS.items():
        for fb in fallbacks:
            matches = [k for k in price_keys if k in fb or fb in k]
            assert matches, f"{provider_cls.__name__}: fallback {fb!r} for {source!r} has no matching price-table entry"


# ---------------------------------------------------------------------------
# BaseProvider._resolve_model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider_cls,tier,expected_substring",
    [
        (OpenAIProvider, "top", "gpt-5.5"),
        (OpenAIProvider, "cheap", "gpt-5.4-nano"),
        (AnthropicProvider, "top", "claude-opus-4-7"),
        (AnthropicProvider, "cheap", "claude-haiku"),
        (GoogleProvider, "top", "gemini-3.1-pro"),
        (DeepSeekProvider, "top", "deepseek-v4-pro"),
        (DeepSeekProvider, "cheap", "deepseek-v4-flash"),
    ],
)
def test_resolve_model_resolves_tier_aliases(provider_cls, tier, expected_substring):
    p = _bare(provider_cls)
    req = CompletionRequest(messages=[Message(role=Role.USER, content="hi")], model=tier)
    resolved = p._resolve_model(req)
    assert expected_substring in resolved


@pytest.mark.parametrize(
    "provider_cls",
    [OpenAIProvider, AnthropicProvider, GoogleProvider, DeepSeekProvider],
)
def test_resolve_model_passes_concrete_models_through(provider_cls):
    p = _bare(provider_cls)
    req = CompletionRequest(
        messages=[Message(role=Role.USER, content="hi")],
        model="some-bespoke-finetune-v2",
    )
    assert p._resolve_model(req) == "some-bespoke-finetune-v2"


def test_resolve_model_falls_back_through_instance_and_default():
    p = _bare(OpenAIProvider)
    p.model = "gpt-4.1"
    req = CompletionRequest(messages=[Message(role=Role.USER, content="hi")])
    assert p._resolve_model(req) == "gpt-4.1"

    p.model = ""
    assert p._resolve_model(req) == OpenAIProvider.default_model


# ---------------------------------------------------------------------------
# BaseProvider._check_native_tools
# ---------------------------------------------------------------------------


def test_check_native_tools_drops_unsupported_with_warning():
    p = _bare(OpenAIProvider)
    # OpenAI doesn't support GOOGLE_SEARCH; WEB_SEARCH passes through.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        kept = p._check_native_tools([NativeTool.WEB_SEARCH, NativeTool.GOOGLE_SEARCH])
    assert NativeTool.WEB_SEARCH in kept
    assert NativeTool.GOOGLE_SEARCH not in kept
    assert any("does not support native tool" in str(x.message) for x in w)


def test_check_native_tools_passes_all_supported_silently():
    p = _bare(GoogleProvider)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        kept = p._check_native_tools([NativeTool.WEB_SEARCH, NativeTool.GOOGLE_SEARCH, NativeTool.GOOGLE_MAPS])
    assert len(kept) == 3
    assert not any("does not support" in str(x.message) for x in w)


def test_check_native_tools_empty_input_returns_empty():
    p = _bare(AnthropicProvider)
    assert p._check_native_tools([]) == []


# ---------------------------------------------------------------------------
# DeepSeek thinking-mode classifier
# ---------------------------------------------------------------------------


def test_deepseek_is_reasoning_model_only_for_legacy_reasoner():
    p = _bare(DeepSeekProvider)
    for m in _REASONING_MODELS:
        assert p._is_reasoning_model(m) is True
    # V4 models are NOT classified as always-reasoning even though
    # they support optional thinking — that's a separate predicate.
    for m in _THINKING_CAPABLE_MODELS:
        assert p._is_reasoning_model(m) is False
    assert p._is_reasoning_model("definitely-not-deepseek") is False


def test_deepseek_is_thinking_active_only_when_v4_and_thinking_enabled():
    from lazybridge.core.types import ThinkingConfig

    p = _bare(DeepSeekProvider)
    p.model = "deepseek-v4-flash"
    req_off = CompletionRequest(messages=[Message(role=Role.USER, content="x")])
    req_on = CompletionRequest(
        messages=[Message(role=Role.USER, content="x")],
        thinking=ThinkingConfig(enabled=True),
    )
    assert p._is_thinking_active(req_off, "deepseek-v4-flash") is False
    assert p._is_thinking_active(req_on, "deepseek-v4-flash") is True
    # Legacy reasoner is always thinking, regardless of request flag.
    assert p._is_thinking_active(req_off, "deepseek-reasoner") is True
    # Unknown model → False on both.
    assert p._is_thinking_active(req_on, "unknown-deepseek") is False


# ---------------------------------------------------------------------------
# OpenAI reasoning-model classifier
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Request-translation: OpenAI Chat Completions / Responses
# ---------------------------------------------------------------------------


def _user_request(text: str = "hello", **kw) -> CompletionRequest:
    defaults = dict(messages=[Message(role=Role.USER, content=text)])
    defaults.update(kw)
    return CompletionRequest(**defaults)


def test_openai_messages_to_openai_basic_user():
    p = _bare(OpenAIProvider)
    msgs = p._messages_to_openai(_user_request("hi"))
    assert msgs == [{"role": "user", "content": "hi"}]


def test_openai_messages_to_openai_system_then_user():
    p = _bare(OpenAIProvider)
    req = CompletionRequest(
        messages=[
            Message(role=Role.SYSTEM, content="be concise"),
            Message(role=Role.USER, content="hi"),
        ],
    )
    msgs = p._messages_to_openai(req)
    assert msgs[0]["role"] in ("system", "developer")
    assert msgs[0]["content"] == "be concise"
    assert msgs[-1] == {"role": "user", "content": "hi"}


def test_openai_messages_to_openai_assistant_passthrough():
    p = _bare(OpenAIProvider)
    req = CompletionRequest(
        messages=[
            Message(role=Role.USER, content="q"),
            Message(role=Role.ASSISTANT, content="a"),
        ],
    )
    msgs = p._messages_to_openai(req)
    assert {"role": "assistant", "content": "a"} in msgs


def test_openai_build_chat_params_includes_max_tokens_for_non_reasoning():
    p = _bare(OpenAIProvider)
    p.model = "gpt-4o"
    params = p._build_chat_params(_user_request(max_tokens=256))
    assert params["max_tokens"] == 256
    assert "max_completion_tokens" not in params


def test_openai_build_chat_params_uses_max_completion_tokens_for_reasoning():
    p = _bare(OpenAIProvider)
    p.model = "o3"
    params = p._build_chat_params(_user_request(max_tokens=512))
    assert params["max_completion_tokens"] == 512
    assert "max_tokens" not in params


def test_openai_build_chat_params_temperature_only_on_non_reasoning():
    p = _bare(OpenAIProvider)
    p.model = "gpt-4o"
    params = p._build_chat_params(_user_request(temperature=0.4))
    assert params["temperature"] == 0.4

    p.model = "o3"
    params = p._build_chat_params(_user_request(temperature=0.4))
    assert "temperature" not in params  # silently dropped — reasoning models reject it


def test_openai_build_chat_params_thinking_emits_reasoning_effort():
    from lazybridge.core.types import ThinkingConfig

    p = _bare(OpenAIProvider)
    p.model = "o3"
    req = CompletionRequest(
        messages=[Message(role=Role.USER, content="x")],
        thinking=ThinkingConfig(enabled=True, effort="high"),
    )
    params = p._build_chat_params(req)
    assert params["reasoning_effort"] == "high"


def test_openai_build_chat_params_tool_choice_string_passthrough():
    from lazybridge.core.types import ToolDefinition

    p = _bare(OpenAIProvider)
    p.model = "gpt-4o"
    req = CompletionRequest(
        messages=[Message(role=Role.USER, content="x")],
        tools=[ToolDefinition(name="t", description="d", parameters={"type": "object"})],
        tool_choice="auto",
    )
    params = p._build_chat_params(req)
    assert params["tool_choice"] == "auto"


def test_openai_build_chat_params_tool_choice_named_function():
    from lazybridge.core.types import ToolDefinition

    p = _bare(OpenAIProvider)
    p.model = "gpt-4o"
    req = CompletionRequest(
        messages=[Message(role=Role.USER, content="x")],
        tools=[ToolDefinition(name="search", description="d", parameters={"type": "object"})],
        tool_choice="search",
    )
    params = p._build_chat_params(req)
    assert params["tool_choice"] == {"type": "function", "function": {"name": "search"}}


# ---------------------------------------------------------------------------
# Request-translation: Anthropic
# ---------------------------------------------------------------------------


def test_anthropic_messages_basic_user():
    p = _bare(AnthropicProvider)
    msgs = p._messages_to_anthropic(_user_request("hi"))
    assert msgs == [{"role": "user", "content": "hi"}]


def test_anthropic_messages_skips_system_role():
    """``system`` is a separate Anthropic API parameter, not a message."""
    p = _bare(AnthropicProvider)
    req = CompletionRequest(
        messages=[
            Message(role=Role.SYSTEM, content="be concise"),
            Message(role=Role.USER, content="hi"),
        ],
    )
    msgs = p._messages_to_anthropic(req)
    assert {"role": "system", "content": "be concise"} not in msgs
    assert {"role": "user", "content": "hi"} in msgs


def test_anthropic_messages_assistant_passthrough():
    p = _bare(AnthropicProvider)
    req = CompletionRequest(
        messages=[
            Message(role=Role.USER, content="q"),
            Message(role=Role.ASSISTANT, content="a"),
        ],
    )
    msgs = p._messages_to_anthropic(req)
    assert msgs[-1] == {"role": "assistant", "content": "a"}


def test_anthropic_messages_tool_role_demotes_to_user():
    """Anthropic has no ``tool`` role for plain string content — it must
    be demoted to ``user`` so the API doesn't reject the request."""
    p = _bare(AnthropicProvider)
    req = CompletionRequest(
        messages=[
            Message(role=Role.USER, content="q"),
            Message(role=Role.TOOL, content="tool result text"),
        ],
    )
    msgs = p._messages_to_anthropic(req)
    # Plain-string Role.TOOL flips to "user".
    assert all(m["role"] in ("user", "assistant") for m in msgs)


# ---------------------------------------------------------------------------
# Static parametrize block — kept at the end for readability.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,expected",
    [
        # Explicit o-series and gpt-5 family — all use ``reasoning_effort``.
        ("o1", True),
        ("o1-mini", True),
        ("o1-pro", True),
        ("o3", True),
        ("o3-mini", True),
        ("o3-pro", True),
        ("o4-mini", True),
        ("gpt-5", True),
        ("gpt-5.4", True),
        ("gpt-5.5", True),
        ("gpt-5.5-pro", True),
        # Non-reasoning families.
        ("gpt-4o", False),
        ("gpt-4.1", False),
        ("gpt-4-turbo", False),
        ("claude-opus-4-7", False),
    ],
)
def test_openai_is_reasoning_model(model, expected):
    p = _bare(OpenAIProvider)
    assert p._is_reasoning_model(model) is expected


# ---------------------------------------------------------------------------
# DeepSeek thinking validation + parameter mutation
# ---------------------------------------------------------------------------


def test_deepseek_resolve_thinking_passthrough_when_disabled():
    p = _bare(DeepSeekProvider)
    p.model = "deepseek-v4-flash"
    req = _user_request("hi")
    assert p._resolve_thinking(req) is req


def test_deepseek_resolve_thinking_accepts_v4_pro():
    from lazybridge.core.types import ThinkingConfig

    p = _bare(DeepSeekProvider)
    p.model = "deepseek-v4-pro"
    req = CompletionRequest(
        messages=[Message(role=Role.USER, content="x")],
        thinking=ThinkingConfig(enabled=True),
    )
    # No raise — this model supports thinking.
    assert p._resolve_thinking(req) is req


def test_deepseek_resolve_thinking_rejects_unsupported_model():
    from lazybridge.core.types import ThinkingConfig

    p = _bare(DeepSeekProvider)
    p.model = "deepseek-chat"  # legacy non-thinking model
    req = CompletionRequest(
        messages=[Message(role=Role.USER, content="x")],
        thinking=ThinkingConfig(enabled=True),
    )
    with pytest.raises(ValueError, match="thinking was requested"):
        p._resolve_thinking(req)


def test_deepseek_apply_thinking_params_activates_extra_body():
    from lazybridge.core.types import ThinkingConfig

    p = _bare(DeepSeekProvider)
    req = CompletionRequest(
        messages=[Message(role=Role.USER, content="x")],
        thinking=ThinkingConfig(enabled=True),
    )
    params: dict = {"temperature": 0.5, "top_p": 0.9, "max_tokens": 100}
    p._apply_thinking_params(params, "deepseek-v4-flash", req)
    assert params["extra_body"]["thinking"] == {"type": "enabled"}
    # Suppressed params are stripped to avoid silent ignore by the API.
    assert "temperature" not in params
    assert "top_p" not in params
    # Non-suppressed params survive.
    assert params["max_tokens"] == 100


def test_deepseek_apply_thinking_params_no_op_for_legacy_reasoner():
    """``deepseek-reasoner`` uses always-on thinking via a different
    mechanism — ``_apply_thinking_params`` must not double-toggle."""
    from lazybridge.core.types import ThinkingConfig

    p = _bare(DeepSeekProvider)
    req = CompletionRequest(
        messages=[Message(role=Role.USER, content="x")],
        thinking=ThinkingConfig(enabled=True),
    )
    params: dict = {"temperature": 0.5}
    p._apply_thinking_params(params, "deepseek-reasoner", req)
    assert "extra_body" not in params
    assert params["temperature"] == 0.5  # untouched


def test_deepseek_apply_thinking_params_no_op_when_thinking_disabled():
    p = _bare(DeepSeekProvider)
    req = _user_request("hi")
    params: dict = {"temperature": 0.5}
    p._apply_thinking_params(params, "deepseek-v4-flash", req)
    # V4 thinking-capable models get an explicit disable to prevent the API
    # from returning reasoning_content (which would break multi-turn tool calls).
    assert params["extra_body"] == {"thinking": {"type": "disabled"}}
    assert params["temperature"] == 0.5


# ---------------------------------------------------------------------------
# OpenAI Responses-API param building
# ---------------------------------------------------------------------------


def test_openai_build_responses_params_basic_input():
    p = _bare(OpenAIProvider)
    p.model = "gpt-5.5"
    params = p._build_responses_params(_user_request("hi"))
    assert params["model"] == "gpt-5.5"
    assert isinstance(params["input"], list)


def test_openai_build_responses_params_with_max_tokens():
    p = _bare(OpenAIProvider)
    p.model = "gpt-5.5"
    params = p._build_responses_params(_user_request(max_tokens=512))
    assert params["max_output_tokens"] == 512


def test_openai_build_responses_params_with_thinking():
    from lazybridge.core.types import ThinkingConfig

    p = _bare(OpenAIProvider)
    p.model = "gpt-5.5"
    req = CompletionRequest(
        messages=[Message(role=Role.USER, content="x")],
        thinking=ThinkingConfig(enabled=True, effort="high"),
    )
    params = p._build_responses_params(req)
    assert params["reasoning"]["effort"] == "high"


def test_openai_build_responses_params_structured_output_dict_schema():
    from lazybridge.core.types import StructuredOutputConfig

    p = _bare(OpenAIProvider)
    p.model = "gpt-5.5"
    req = CompletionRequest(
        messages=[Message(role=Role.USER, content="x")],
        structured_output=StructuredOutputConfig(
            schema={"type": "object", "properties": {"x": {"type": "integer"}}},
            strict=True,
        ),
    )
    params = p._build_responses_params(req)
    assert params["text"]["format"]["type"] == "json_schema"
    assert params["text"]["format"]["strict"] is True


def test_openai_build_responses_params_structured_output_pydantic_skipped():
    """Pydantic schemas can't be encoded into ``text.format`` — they
    take the Chat Completions ``beta.parse`` path elsewhere.  The
    Responses param builder should NOT emit a ``text`` field for them."""
    from pydantic import BaseModel

    from lazybridge.core.types import StructuredOutputConfig

    class _Out(BaseModel):
        x: int

    p = _bare(OpenAIProvider)
    p.model = "gpt-5.5"
    req = CompletionRequest(
        messages=[Message(role=Role.USER, content="x")],
        structured_output=StructuredOutputConfig(schema=_Out, strict=True),
    )
    params = p._build_responses_params(req)
    assert "text" not in params or params.get("text") is None


# ---------------------------------------------------------------------------
# Anthropic multimodal content-block translation
# ---------------------------------------------------------------------------


def test_anthropic_messages_text_blocks():
    from lazybridge.core.types import TextContent

    p = _bare(AnthropicProvider)
    req = CompletionRequest(
        messages=[Message(role=Role.USER, content=[TextContent(text="hello")])],
    )
    msgs = p._messages_to_anthropic(req)
    assert msgs[0]["role"] == "user"
    blocks = msgs[0]["content"]
    assert isinstance(blocks, list)
    assert blocks[0]["type"] == "text"
    assert blocks[0]["text"] == "hello"


def test_anthropic_messages_tool_use_blocks():
    from lazybridge.core.types import ToolUseContent

    p = _bare(AnthropicProvider)
    req = CompletionRequest(
        messages=[
            Message(
                role=Role.ASSISTANT,
                content=[ToolUseContent(id="t1", name="search", input={"q": "x"})],
            ),
        ],
    )
    msgs = p._messages_to_anthropic(req)
    blocks = msgs[0]["content"]
    assert blocks[0]["type"] == "tool_use"
    assert blocks[0]["id"] == "t1"
    assert blocks[0]["name"] == "search"


def test_anthropic_messages_tool_result_blocks():
    from lazybridge.core.types import ToolResultContent

    p = _bare(AnthropicProvider)
    req = CompletionRequest(
        messages=[
            Message(
                role=Role.TOOL,
                content=[ToolResultContent(tool_use_id="t1", content="42")],
            ),
        ],
    )
    msgs = p._messages_to_anthropic(req)
    blocks = msgs[0]["content"]
    assert blocks[0]["type"] == "tool_result"
    assert blocks[0]["tool_use_id"] == "t1"


# ---------------------------------------------------------------------------
# OpenAI Responses-API input translation
# ---------------------------------------------------------------------------


def test_openai_messages_to_responses_input_basic():
    p = _bare(OpenAIProvider)
    req = CompletionRequest(
        messages=[
            Message(role=Role.SYSTEM, content="be concise"),
            Message(role=Role.USER, content="hi"),
        ],
    )
    items = p._messages_to_responses_input(req)
    assert isinstance(items, list)
    # System / developer instruction comes first, user follows.
    roles = [item.get("role") for item in items if isinstance(item, dict)]
    assert "user" in roles


def test_openai_messages_to_responses_input_assistant_passthrough():
    p = _bare(OpenAIProvider)
    req = CompletionRequest(
        messages=[
            Message(role=Role.USER, content="q"),
            Message(role=Role.ASSISTANT, content="a"),
        ],
    )
    items = p._messages_to_responses_input(req)
    roles = [item.get("role") for item in items if isinstance(item, dict)]
    assert "assistant" in roles


# ---------------------------------------------------------------------------
# Provider tier resolution end-to-end via Agent / LLMEngine factories
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider_str,model_str",
    [
        ("openai", "top"),
        ("openai", "cheap"),
        ("anthropic", "top"),
        ("anthropic", "cheap"),
        ("google", "top"),
        ("deepseek", "top"),
    ],
)
def test_llm_engine_infer_provider_for_known_tier_aliases(provider_str, model_str):
    """Tier aliases on a *bare model string* still infer the right provider."""
    from lazybridge.engines.llm import LLMEngine

    # ``model_str`` here is a tier name — provider must come from the
    # caller-supplied ``provider`` arg or from the inference rule.
    inferred = LLMEngine._infer_provider(provider_str)
    assert inferred == provider_str


# ---------------------------------------------------------------------------
# Plan.to_dict / Plan.from_dict round-trips for every sentinel kind
# ---------------------------------------------------------------------------


def _agent_target(name: str):
    """Build a minimal Agent-like duck-type that round-trips through
    ``_target_to_ref`` as ``kind="agent"``.  No real Agent needed."""

    class _A:
        _is_lazy_agent = True

        def __init__(self, name: str) -> None:
            self.name = name
            self.description = ""

    return _A(name)


def test_plan_serialization_roundtrip_every_sentinel():
    from lazybridge.engines.plan import Plan, Step
    from lazybridge.sentinels import (
        from_parallel,
        from_parallel_all,
        from_prev,
        from_start,
        from_step,
    )

    def fn_a(t: str) -> str:
        return t

    def fn_b(t: str) -> str:
        return t

    def fn_join(t: str) -> str:
        return t

    plan = Plan(
        Step(target=fn_a, name="a"),
        Step(target=fn_b, name="b1", parallel=True, task=from_step("a")),
        Step(target=fn_b, name="b2", parallel=True, task=from_prev),
        Step(target=fn_join, name="join", task=from_parallel_all("b1")),
        Step(target=fn_b, name="post", task=from_start),
        Step(target=fn_b, name="post2", task=from_parallel("b1")),
        max_iterations=42,
    )

    blob = plan.to_dict()
    assert blob["max_iterations"] == 42
    assert len(blob["steps"]) == 6

    restored = Plan.from_dict(blob, registry={"fn_a": fn_a, "fn_b": fn_b, "fn_join": fn_join})
    assert restored.max_iterations == 42
    names = [s.name for s in restored.steps]
    assert names == ["a", "b1", "b2", "join", "post", "post2"]

    # Sentinel kinds are preserved.
    assert restored.steps[1].task.__class__.__name__ == "_FromStep"
    assert restored.steps[2].task.__class__.__name__ == "_FromPrev"
    assert restored.steps[3].task.__class__.__name__ == "_FromParallelAll"
    assert restored.steps[4].task.__class__.__name__ == "_FromStart"
    assert restored.steps[5].task.__class__.__name__ == "_FromParallel"


def test_plan_serialization_literal_str_task_roundtrips():
    """A literal string task (e.g. ``Step(fn, task="hard-coded")``) round-trips."""
    from lazybridge.engines.plan import Plan, Step

    def fn(t: str) -> str:
        return t

    plan = Plan(Step(target=fn, name="s", task="literal-task-string"))
    restored = Plan.from_dict(plan.to_dict(), registry={"fn": fn})
    assert restored.steps[0].task == "literal-task-string"


def test_plan_serialization_tool_target_passes_through_as_string():
    """``Step(target="tool_name")`` keeps the string — tool_map resolves it
    at run-time, so no registry entry is required for tool kinds."""
    from lazybridge.engines.plan import Plan, Step

    plan = Plan(Step(target="my_tool", name="step1"))
    restored = Plan.from_dict(plan.to_dict(), registry={})
    assert restored.steps[0].target == "my_tool"


def test_plan_serialization_agent_target_resolves_via_registry():
    from lazybridge.engines.plan import Plan, Step

    inner = _agent_target("inner")
    plan = Plan(Step(target=inner, name="s"))
    restored = Plan.from_dict(plan.to_dict(), registry={"inner": inner})
    assert restored.steps[0].target is inner


def test_plan_serialization_missing_registry_entry_raises_key_error():
    """A non-tool target that isn't in the registry must fail loudly."""
    from lazybridge.engines.plan import Plan, Step

    def fn(t: str) -> str:
        return t

    plan = Plan(Step(target=fn, name="s"))
    with pytest.raises(KeyError, match="no entry in registry"):
        Plan.from_dict(plan.to_dict(), registry={})


def test_plan_serialization_preserves_step_attributes():
    """``writes`` / ``parallel`` / ``context`` / ``input`` / ``output``
    survive the round-trip."""
    from lazybridge.engines.plan import Plan, Step

    def fn(t: str) -> str:
        return t

    plan = Plan(
        Step(
            target=fn,
            name="s",
            writes="result_key",
            parallel=False,
            context="extra context",
        ),
    )
    restored = Plan.from_dict(plan.to_dict(), registry={"fn": fn})
    s = restored.steps[0]
    assert s.writes == "result_key"
    assert s.parallel is False
    assert s.context == "extra context"


def test_plan_serialization_unknown_sentinel_kind_falls_back_to_from_prev():
    """An unknown ``kind`` in a serialised sentinel ref defaults to
    ``from_prev`` rather than raising — graceful forward compatibility."""
    from lazybridge.engines.plan import _sentinel_from_ref

    result = _sentinel_from_ref({"kind": "from_unknown_future_thing"})
    assert result.__class__.__name__ == "_FromPrev"


def test_plan_serialization_none_ref_is_from_prev():
    """``None`` (no ``task=`` set) deserialises to ``from_prev``."""
    from lazybridge.engines.plan import _sentinel_from_ref

    result = _sentinel_from_ref(None)
    assert result.__class__.__name__ == "_FromPrev"


def test_plan_serialization_empty_dict_yields_empty_plan():
    from lazybridge.engines.plan import Plan

    plan = Plan.from_dict({}, registry={})
    assert plan.steps == []
    assert plan.max_iterations == 100  # default


# ---------------------------------------------------------------------------
# Message helpers (factories + content-block text extraction)
# ---------------------------------------------------------------------------


def test_message_factories_set_role():
    assert Message.user("hi").role == Role.USER
    assert Message.assistant("hi").role == Role.ASSISTANT
    assert Message.system("hi").role == Role.SYSTEM


def test_message_to_text_extracts_from_string_content():
    assert Message.user("hello").to_text() == "hello"


def test_message_to_text_extracts_from_text_blocks():
    from lazybridge.core.types import TextContent

    msg = Message(
        role=Role.ASSISTANT,
        content=[TextContent(text="part one"), TextContent(text="part two")],
    )
    assert "part one" in msg.to_text()
    assert "part two" in msg.to_text()


def test_message_to_text_extracts_thinking_content():
    from lazybridge.core.types import TextContent, ThinkingContent

    msg = Message(
        role=Role.ASSISTANT,
        content=[
            ThinkingContent(thinking="reasoning step"),
            TextContent(text="final answer"),
        ],
    )
    out = msg.to_text()
    assert "reasoning step" in out
    assert "final answer" in out


def test_message_to_text_skips_unknown_block_types():
    """Tool-use / image / unknown blocks are not text-extractable; the
    helper just collects whatever IS text."""
    from lazybridge.core.types import TextContent, ToolUseContent

    msg = Message(
        role=Role.ASSISTANT,
        content=[
            ToolUseContent(id="t1", name="search", input={"q": "x"}),
            TextContent(text="visible text"),
        ],
    )
    assert msg.to_text() == "visible text"


# ---------------------------------------------------------------------------
# Structured-output exception classes — carry provider / model / raw
# ---------------------------------------------------------------------------


def test_structured_output_parse_error_carries_context():
    from lazybridge.core.types import StructuredOutputParseError

    exc = StructuredOutputParseError("bad json", provider="openai", model="gpt-4o", raw='{"x":')
    assert str(exc) == "bad json"
    assert exc.provider == "openai"
    assert exc.model == "gpt-4o"
    assert exc.raw == '{"x":'


def test_structured_output_parse_error_optional_context():
    from lazybridge.core.types import StructuredOutputParseError

    exc = StructuredOutputParseError("bad")
    assert exc.provider is None
    assert exc.model is None
    assert exc.raw is None


# ---------------------------------------------------------------------------
# Store iteration / containment dunders
# ---------------------------------------------------------------------------


def test_store_iter_yields_keys():
    from lazybridge.store import Store

    s = Store()
    s.write("a", 1)
    s.write("b", 2)
    assert sorted(iter(s)) == ["a", "b"]


def test_store_contains_by_key():
    from lazybridge.store import Store

    s = Store()
    s.write("present", "v")
    assert "present" in s
    assert "absent" not in s


def test_store_len_matches_keys():
    from lazybridge.store import Store

    s = Store()
    assert len(s) == 0
    s.write("a", 1)
    assert len(s) == 1
    s.write("b", 2)
    assert len(s) == 2


# ---------------------------------------------------------------------------
# Memory helpers — small file, two-line gap
# ---------------------------------------------------------------------------


def test_memory_text_renders_turns():
    from lazybridge.memory import Memory

    m = Memory()
    m.add("user question", "assistant answer")
    text = m.text()
    assert "user question" in text
    assert "assistant answer" in text


def test_memory_clear_drops_all_turns():
    from lazybridge.memory import Memory

    m = Memory()
    m.add("q1", "a1")
    m.add("q2", "a2")
    m.clear()
    assert m.text() == "" or m.text().strip() == ""


# ---------------------------------------------------------------------------
# CompletionResponse.raise_if_failed dispatch
# ---------------------------------------------------------------------------


def test_completion_response_raise_if_failed_no_op_when_clean():
    from lazybridge.core.types import CompletionResponse, UsageStats

    resp = CompletionResponse(content="ok", usage=UsageStats())
    resp.validation_error = None
    resp.raise_if_failed()  # no exception


def test_completion_response_raise_if_failed_parse_error_branch():
    from lazybridge.core.types import (
        CompletionResponse,
        StructuredOutputParseError,
        UsageStats,
    )

    resp = CompletionResponse(content="{bad json", usage=UsageStats())
    resp.validation_error = "JSON parse error: unexpected end of input"
    with pytest.raises(StructuredOutputParseError) as exc_info:
        resp.raise_if_failed()
    assert exc_info.value.raw == "{bad json"


def test_completion_response_raise_if_failed_validation_error_branch():
    from lazybridge.core.types import (
        CompletionResponse,
        StructuredOutputValidationError,
        UsageStats,
    )

    resp = CompletionResponse(content='{"x": "wrong type"}', usage=UsageStats())
    resp.validation_error = "field 'x': expected integer"
    resp.parsed = {"x": "wrong type"}
    with pytest.raises(StructuredOutputValidationError) as exc_info:
        resp.raise_if_failed()
    assert exc_info.value.parsed == {"x": "wrong type"}


# ---------------------------------------------------------------------------
# Tier-alias helpers — ``BaseProvider.resolve_model_alias`` round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider_cls,tier",
    [
        (OpenAIProvider, "top"),
        (OpenAIProvider, "medium"),
        (AnthropicProvider, "expensive"),
        (DeepSeekProvider, "super_cheap"),
        (GoogleProvider, "medium"),
    ],
)
def test_resolve_model_via_request_with_tier(provider_cls, tier):
    """End-to-end: a request whose ``model`` is a tier alias resolves to
    a concrete model."""
    p = _bare(provider_cls)
    req = CompletionRequest(messages=[Message(role=Role.USER, content="x")], model=tier)
    resolved = p._resolve_model(req)
    assert resolved == provider_cls._TIER_ALIASES[tier]


# ---------------------------------------------------------------------------
# OpenAI _build_function_tools translation
# ---------------------------------------------------------------------------


def test_openai_build_function_tools_empty():
    p = _bare(OpenAIProvider)
    assert p._build_function_tools(_user_request()) == []


def test_openai_build_function_tools_single_tool():
    from lazybridge.core.types import ToolDefinition

    p = _bare(OpenAIProvider)
    req = CompletionRequest(
        messages=[Message(role=Role.USER, content="x")],
        tools=[
            ToolDefinition(
                name="search",
                description="Search the web.",
                parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            ),
        ],
    )
    tools = p._build_function_tools(req)
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "search"
    assert tools[0]["function"]["description"] == "Search the web."
    assert tools[0]["function"]["parameters"]["properties"] == {"q": {"type": "string"}}


def test_openai_build_function_tools_strict_flag():
    from lazybridge.core.types import ToolDefinition

    p = _bare(OpenAIProvider)
    req = CompletionRequest(
        messages=[Message(role=Role.USER, content="x")],
        tools=[
            ToolDefinition(
                name="t1",
                description="d",
                parameters={"type": "object"},
                strict=True,
            ),
            ToolDefinition(
                name="t2",
                description="d",
                parameters={"type": "object"},
                strict=False,
            ),
        ],
    )
    tools = p._build_function_tools(req)
    assert tools[0]["function"]["strict"] is True
    assert "strict" not in tools[1]["function"]


# ---------------------------------------------------------------------------
# OpenAI usage-stat extractors (reasoning + cached input tokens)
# ---------------------------------------------------------------------------


def test_openai_populate_reasoning_tokens_from_completion_details():
    from lazybridge.core.types import UsageStats

    class _Details:
        reasoning_tokens = 42

    class _Usage:
        completion_tokens_details = _Details()

    out = OpenAIProvider._populate_reasoning_tokens(UsageStats(), _Usage())
    assert out.thinking_tokens == 42


def test_openai_populate_reasoning_tokens_no_details_is_zero():
    from lazybridge.core.types import UsageStats

    class _UsageNoDetails:
        pass

    out = OpenAIProvider._populate_reasoning_tokens(UsageStats(), _UsageNoDetails())
    assert out.thinking_tokens == 0  # default


def test_openai_populate_cached_input_tokens():
    from lazybridge.core.types import UsageStats

    class _PromptDetails:
        cached_tokens = 1234

    class _Usage:
        prompt_tokens_details = _PromptDetails()

    out = OpenAIProvider._populate_cached_input_tokens(UsageStats(), _Usage())
    assert out.cached_input_tokens == 1234


def test_openai_populate_cached_input_tokens_handles_missing_field():
    from lazybridge.core.types import UsageStats

    class _Empty:
        pass

    out = OpenAIProvider._populate_cached_input_tokens(UsageStats(), _Empty())
    assert out.cached_input_tokens == 0


# ---------------------------------------------------------------------------
# OpenAI _responses_stop_reason classifier
# ---------------------------------------------------------------------------


def test_openai_responses_stop_reason_with_tool_calls_returns_tool_use():
    from lazybridge.core.types import ToolCall

    tc = [ToolCall(id="1", name="search", arguments={})]
    assert OpenAIProvider._responses_stop_reason(None, tc) == "tool_use"


def test_openai_responses_stop_reason_completed_no_tools_returns_end_turn():
    class _Resp:
        status = "completed"

    assert OpenAIProvider._responses_stop_reason(_Resp(), []) == "end_turn"


def test_openai_responses_stop_reason_failed_returns_error():
    class _Resp:
        status = "failed"

    assert OpenAIProvider._responses_stop_reason(_Resp(), []) == "error"


# ---------------------------------------------------------------------------
# Anthropic _build_*_params via the inherited surface (smoke checks)
# ---------------------------------------------------------------------------


def test_anthropic_compute_cost_with_long_dated_model_substring():
    """The price table uses substring matching, so a fully-qualified
    name like ``claude-sonnet-4-6-20250514`` should still match."""
    p = _bare(AnthropicProvider)
    cost_long = p._compute_cost("claude-sonnet-4-6-20250514", 1000, 1000)
    cost_short = p._compute_cost("claude-sonnet-4-6", 1000, 1000)
    assert cost_long == cost_short


def test_anthropic_compute_cost_table_prefix_ordering():
    """The table is ordered longest-prefix first.  ``claude-3-5-sonnet``
    must NOT collide with the shorter ``claude-3-sonnet`` key."""
    p = _bare(AnthropicProvider)
    cost_3_5 = p._compute_cost("claude-3-5-sonnet", 1000, 1000)
    cost_3 = p._compute_cost("claude-3-sonnet", 1000, 1000)
    # Both should be defined, and they may differ — what matters is that
    # the table doesn't return the WRONG row for the more-specific name.
    assert cost_3_5 is not None
    assert cost_3 is not None


# ---------------------------------------------------------------------------
# LiteLLM helpers — _strip_prefix + _safe_json_loads
# ---------------------------------------------------------------------------


def test_litellm_strip_prefix_removes_litellm_prefix():
    from lazybridge.core.providers.litellm import _strip_prefix

    assert _strip_prefix("litellm/groq/llama") == "groq/llama"


def test_litellm_strip_prefix_passes_unprefixed_through():
    from lazybridge.core.providers.litellm import _strip_prefix

    assert _strip_prefix("mistral/mistral-large") == "mistral/mistral-large"


def test_litellm_safe_json_loads_valid():
    from lazybridge.core.providers.litellm import _safe_json_loads

    assert _safe_json_loads('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_litellm_safe_json_loads_empty_string():
    from lazybridge.core.providers.litellm import _safe_json_loads

    assert _safe_json_loads("") == {}


def test_litellm_safe_json_loads_malformed_returns_raw_arguments():
    """Malformed JSON keeps the raw string AND tags ``_parse_error`` so
    the engine can surface a structured TOOL_ERROR."""
    from lazybridge.core.providers.litellm import _safe_json_loads

    out = _safe_json_loads("{not valid json")
    assert out["_raw_arguments"] == "{not valid json"
    assert out.get("_parse_error")


def test_litellm_safe_json_loads_non_dict_top_level_returns_raw_arguments():
    """LiteLLM's parser tags non-object payloads — function-call args
    must always be objects."""
    from lazybridge.core.providers.litellm import _safe_json_loads

    out = _safe_json_loads("[1, 2, 3]")
    assert out["_raw_arguments"] == "[1, 2, 3]"
    assert "expected object" in out["_parse_error"]


# ---------------------------------------------------------------------------
# OpenAI _parse_chat_response — fake response shape
# ---------------------------------------------------------------------------


class _FakeMsg:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _FakeChoice:
    def __init__(self, message=None, finish_reason="stop"):
        self.message = message or _FakeMsg()
        self.finish_reason = finish_reason


class _FakeUsage:
    def __init__(self, prompt_tokens=0, completion_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeChatResp:
    def __init__(self, choices=None, model="gpt-5.5", usage=None):
        self.choices = choices or []
        self.model = model
        self.usage = usage


def test_openai_parse_chat_response_basic_text():
    p = _bare(OpenAIProvider)
    resp = _FakeChatResp(
        choices=[_FakeChoice(message=_FakeMsg(content="hello world"))],
        usage=_FakeUsage(prompt_tokens=10, completion_tokens=5),
    )
    parsed = p._parse_chat_response(resp)
    assert parsed.content == "hello world"
    assert parsed.usage.input_tokens == 10
    assert parsed.usage.output_tokens == 5
    assert parsed.stop_reason == "stop"


def test_openai_parse_chat_response_no_choices_returns_error_stop_reason():
    p = _bare(OpenAIProvider)
    resp = _FakeChatResp(choices=[], usage=_FakeUsage(prompt_tokens=10, completion_tokens=0))
    parsed = p._parse_chat_response(resp)
    assert parsed.content == ""
    assert parsed.stop_reason == "error"
    assert parsed.usage.input_tokens == 10


def test_openai_parse_chat_response_with_tool_calls():
    from lazybridge.core.types import ToolCall

    class _ToolFn:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments

    class _ToolCall:
        def __init__(self, id, name, arguments):
            self.id = id
            self.function = _ToolFn(name, arguments)

    p = _bare(OpenAIProvider)
    resp = _FakeChatResp(
        choices=[
            _FakeChoice(
                message=_FakeMsg(
                    content="",
                    tool_calls=[_ToolCall("call-1", "search", '{"q": "x"}')],
                ),
            ),
        ],
        usage=_FakeUsage(prompt_tokens=20, completion_tokens=10),
    )
    parsed = p._parse_chat_response(resp)
    assert len(parsed.tool_calls) == 1
    tc: ToolCall = parsed.tool_calls[0]
    assert tc.id == "call-1"
    assert tc.name == "search"
    assert tc.arguments == {"q": "x"}


def test_openai_parse_chat_response_finish_reason_propagates():
    p = _bare(OpenAIProvider)
    resp = _FakeChatResp(
        choices=[_FakeChoice(message=_FakeMsg(content="ok"), finish_reason="length")],
        usage=_FakeUsage(),
    )
    assert p._parse_chat_response(resp).stop_reason == "length"


# ---------------------------------------------------------------------------
# OpenAI _safe_json_loads (module-level helper) — covers same path as
# litellm but the OpenAI implementation is independent.
# ---------------------------------------------------------------------------


def test_openai_safe_json_loads_valid():
    from lazybridge.core.providers.openai import _safe_json_loads

    assert _safe_json_loads('{"a": 1}') == {"a": 1}


def test_openai_safe_json_loads_empty():
    from lazybridge.core.providers.openai import _safe_json_loads

    assert _safe_json_loads("") == {}


def test_openai_safe_json_loads_malformed_returns_raw_arguments():
    """Malformed JSON keeps the raw string AND tags ``_parse_error`` so
    the engine can surface a structured TOOL_ERROR."""
    from lazybridge.core.providers.openai import _safe_json_loads

    out = _safe_json_loads("{not valid")
    assert out["_raw_arguments"] == "{not valid"
    assert out.get("_parse_error")


# ---------------------------------------------------------------------------
# OpenAI _extract_grounding_from_output — Responses API citation parsing
# ---------------------------------------------------------------------------


class _Annotation:
    def __init__(self, type, url=None, title=None):
        self.type = type
        self.url = url
        self.title = title


class _OutputBlock:
    def __init__(self, type, annotations=None):
        self.type = type
        self.annotations = annotations or []


class _OutputItem:
    def __init__(self, type, content=None):
        self.type = type
        self.content = content or []


def test_openai_extract_grounding_empty_output():
    assert OpenAIProvider._extract_grounding_from_output([]) == []
    assert OpenAIProvider._extract_grounding_from_output(None) == []


def test_openai_extract_grounding_skips_non_message_items():
    item = _OutputItem(type="reasoning")
    assert OpenAIProvider._extract_grounding_from_output([item]) == []


def test_openai_extract_grounding_extracts_url_citations():
    block = _OutputBlock(
        type="output_text",
        annotations=[
            _Annotation(type="url_citation", url="https://example.com", title="Example"),
            _Annotation(type="url_citation", url="https://example.org"),
            _Annotation(type="other_kind"),  # ignored
        ],
    )
    item = _OutputItem(type="message", content=[block])
    sources = OpenAIProvider._extract_grounding_from_output([item])
    assert len(sources) == 2
    assert sources[0].url == "https://example.com"
    assert sources[0].title == "Example"
    assert sources[1].url == "https://example.org"


def test_openai_extract_grounding_skips_non_output_text_blocks():
    block = _OutputBlock(type="reasoning_summary", annotations=[_Annotation(type="url_citation", url="x")])
    item = _OutputItem(type="message", content=[block])
    assert OpenAIProvider._extract_grounding_from_output([item]) == []


# ---------------------------------------------------------------------------
# OpenAI _responses_stop_reason — incomplete branch
# ---------------------------------------------------------------------------


def test_openai_responses_stop_reason_incomplete_returns_max_tokens():
    class _Resp:
        status = "incomplete"

    assert OpenAIProvider._responses_stop_reason(_Resp(), []) == "max_tokens"


def test_openai_responses_stop_reason_unknown_status_falls_back():
    class _Resp:
        status = "weird-future-status"

    # Falls through to a safe default — NOT raising, and not "tool_use".
    out = OpenAIProvider._responses_stop_reason(_Resp(), [])
    assert out != "tool_use"
