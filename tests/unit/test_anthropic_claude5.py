"""Regression tests for the Claude 5 family addition (Fable 5 / Opus 5 /
Sonnet 5 / Mythos 5) and the new Anthropic ``effort`` parameter.

Covers:
  * Tier aliases route ``top``/``expensive``/``medium`` to the Claude 5 family.
  * ``_PRICE_TABLE`` returns the Claude 5 rates; Mythos 5 is priced but not
    tier-aliased (restricted access).
  * ``get_default_max_tokens`` returns 128 K for the Claude 5 family.
  * ``_NO_SAMPLING_MODELS`` / ``_ADAPTIVE_ONLY_MODELS`` include the new ids.
  * ``_build_effort`` resolves ``ThinkingConfig.effort`` into
    ``output_config.effort``, no-ops on "high", warns and drops on
    unsupported models, and downgrades "xhigh" to "max" where needed.
  * ``_build_params`` merges ``effort`` and structured-output ``format``
    into the same ``output_config`` dict instead of one clobbering the other.
"""

from __future__ import annotations

import warnings

import pytest

from lazybridge.core.providers.anthropic import (
    _EFFORT_CAPABLE_MODELS,
    _PRICE_TABLE,
    _XHIGH_CAPABLE_MODELS,
    AnthropicProvider,
)
from lazybridge.core.types import CompletionRequest, Message, Role, StructuredOutputConfig, ThinkingConfig


def _provider() -> AnthropicProvider:
    """Build a provider without hitting the network or requiring an API key."""
    p = AnthropicProvider.__new__(AnthropicProvider)
    p.model = None
    p.fallback_model = None
    p._temperature_warned = False
    return p


def _request(thinking: ThinkingConfig | None = None, structured_output=None, model=None) -> CompletionRequest:
    return CompletionRequest(
        messages=[Message(role=Role.USER, content="hi")],
        model=model,
        thinking=thinking,
        structured_output=structured_output,
    )


def test_tier_aliases_route_to_claude_5_family() -> None:
    assert AnthropicProvider._TIER_ALIASES["top"] == "claude-fable-5"
    assert AnthropicProvider._TIER_ALIASES["expensive"] == "claude-opus-5"
    assert AnthropicProvider._TIER_ALIASES["medium"] == "claude-sonnet-5"


def test_price_table_has_claude_5_entries() -> None:
    assert _PRICE_TABLE["claude-fable-5"] == (10.0, 50.0)
    assert _PRICE_TABLE["claude-opus-5"] == (5.0, 25.0)
    assert _PRICE_TABLE["claude-sonnet-5"] == (2.0, 10.0)


def test_mythos_5_is_priced_but_not_tier_aliased() -> None:
    """Mythos 5 is restricted to vetted partners — it must never be reachable
    via a tier alias, only by pinning the model id directly."""
    assert _PRICE_TABLE["claude-mythos-5"] == (10.0, 50.0)
    assert "claude-mythos-5" not in AnthropicProvider._TIER_ALIASES.values()


def test_compute_cost_claude_5_family() -> None:
    p = _provider()
    assert p._compute_cost("claude-fable-5", 1_000_000, 0) == pytest.approx(10.0)
    assert p._compute_cost("claude-opus-5", 1_000_000, 0) == pytest.approx(5.0)
    assert p._compute_cost("claude-sonnet-5", 1_000_000, 0) == pytest.approx(2.0)


def test_get_default_max_tokens_claude_5_family() -> None:
    p = _provider()
    assert p.get_default_max_tokens("claude-fable-5") == 128_000
    assert p.get_default_max_tokens("claude-opus-5") == 128_000
    assert p.get_default_max_tokens("claude-sonnet-5") == 128_000
    # Regression guard: "opus-5" / "sonnet-5" substring matching must not
    # collide with the older dated "-4-5" ids.
    assert p.get_default_max_tokens("claude-opus-4-5") == 64_000
    assert p.get_default_max_tokens("claude-sonnet-4-5") == 64_000


def test_fallback_chains_for_claude_5_family() -> None:
    assert AnthropicProvider._FALLBACKS["claude-fable-5"] == ["claude-opus-5", "claude-sonnet-5"]
    assert AnthropicProvider._FALLBACKS["claude-opus-5"] == ["claude-sonnet-5", "claude-opus-4-8"]


def test_no_sampling_and_adaptive_only_include_claude_5() -> None:
    from lazybridge.core.providers.anthropic import _ADAPTIVE_ONLY_MODELS, _NO_SAMPLING_MODELS

    for model in ("claude-fable-5", "claude-opus-5", "claude-sonnet-5"):
        assert model in _NO_SAMPLING_MODELS
        assert model in _ADAPTIVE_ONLY_MODELS


# ---------------------------------------------------------------------------
# _build_effort
# ---------------------------------------------------------------------------


def test_build_effort_none_when_no_thinking_config() -> None:
    p = _provider()
    assert p._build_effort(_request(thinking=None), "claude-opus-5") is None


def test_build_effort_none_on_default_high() -> None:
    """'high' is the documented no-op — sending it explicitly changes nothing,
    so we omit it from the wire request."""
    p = _provider()
    req = _request(thinking=ThinkingConfig(enabled=True, effort="high"))
    assert p._build_effort(req, "claude-opus-5") is None


def test_build_effort_passes_through_low_medium_max() -> None:
    p = _provider()
    for level in ("low", "medium", "max"):
        req = _request(thinking=ThinkingConfig(enabled=True, effort=level))
        assert p._build_effort(req, "claude-opus-5") == level


def test_build_effort_warns_and_drops_on_unsupported_model() -> None:
    p = _provider()
    req = _request(thinking=ThinkingConfig(enabled=True, effort="low"))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = p._build_effort(req, "claude-haiku-4-5")
    assert result is None
    assert any("does not support the effort parameter" in str(x.message) for x in w)


def test_build_effort_downgrades_xhigh_to_max_where_unsupported() -> None:
    p = _provider()
    req = _request(thinking=ThinkingConfig(enabled=True, effort="xhigh"))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = p._build_effort(req, "claude-opus-4-6")
    assert result == "max"
    assert any("not supported on model" in str(x.message) for x in w)


def test_build_effort_keeps_xhigh_where_supported() -> None:
    p = _provider()
    req = _request(thinking=ThinkingConfig(enabled=True, effort="xhigh"))
    assert p._build_effort(req, "claude-opus-5") == "xhigh"
    assert p._build_effort(req, "claude-fable-5") == "xhigh"


def test_effort_and_xhigh_capable_sets_are_consistent() -> None:
    # Every xhigh-capable model must also be effort-capable.
    assert _XHIGH_CAPABLE_MODELS <= _EFFORT_CAPABLE_MODELS


# ---------------------------------------------------------------------------
# _build_params — effort must merge with structured-output output_config,
# not clobber it (both live under the "output_config" key).
# ---------------------------------------------------------------------------


def test_build_params_merges_effort_with_structured_output_format() -> None:
    p = _provider()
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    req = _request(
        thinking=ThinkingConfig(enabled=True, effort="low"),
        structured_output=StructuredOutputConfig(schema=schema),
        model="claude-opus-5",
    )
    params = p._build_params(req)
    assert params["output_config"]["effort"] == "low"


def test_build_params_omits_output_config_when_effort_is_high_and_no_schema() -> None:
    p = _provider()
    req = _request(thinking=ThinkingConfig(enabled=True), model="claude-opus-5")
    params = p._build_params(req)
    assert "output_config" not in params
