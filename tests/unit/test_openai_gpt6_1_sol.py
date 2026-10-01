"""Regression tests for GPT-6.1 Sol support and the medium/expensive tier move."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import lazybridge
from lazybridge.core.providers.openai import _PRICE_TABLE, OpenAIProvider
from lazybridge.core.types import CompletionRequest, Message, Role, ThinkingConfig


def _provider() -> OpenAIProvider:
    """Build a provider without requiring the optional SDK or an API key."""
    return OpenAIProvider.__new__(OpenAIProvider)


def test_sol_6_1_is_the_medium_and_expensive_openai_tier() -> None:
    assert OpenAIProvider._TIER_ALIASES["medium"] == "gpt-6.1-sol"
    assert OpenAIProvider._TIER_ALIASES["expensive"] == "gpt-6.1-sol"
    # Other tiers are unchanged.
    assert OpenAIProvider._TIER_ALIASES["top"] == "gpt-6-astra"
    assert OpenAIProvider._TIER_ALIASES["cheap"] == "gpt-6-luna"
    assert OpenAIProvider._TIER_ALIASES["super_cheap"] == "gpt-6-luna"


def test_llms_json_openai_tiers_match_provider_aliases() -> None:
    path = Path(lazybridge.__file__).parent / "llms.json"
    tiers = json.loads(path.read_text(encoding="utf-8"))["tier_aliases_openai_2026_09"]
    assert tiers == OpenAIProvider._TIER_ALIASES


def test_gpt_6_sol_is_still_priced_and_selectable() -> None:
    assert _PRICE_TABLE["gpt-6-sol"] == (2.0, 0.20, 10.0)
    assert OpenAIProvider._FALLBACKS["gpt-6-sol"] == ["gpt-5.6-sol", "gpt-5.6-terra"]


def test_sol_6_1_prices_and_cost() -> None:
    assert _PRICE_TABLE["gpt-6.1-sol"] == (2.0, 0.10, 10.0)
    cost = _provider()._compute_cost(
        "gpt-6.1-sol",
        input_tokens=1_000_000,
        output_tokens=100_000,
        cached_input_tokens=250_000,
    )
    assert cost == pytest.approx(0.75 * 2.0 + 0.25 * 0.10 + 0.1 * 10.0)


def test_sol_6_1_cost_does_not_use_gpt_6_sol_cache_rate() -> None:
    p = _provider()
    sol_6 = p._compute_cost("gpt-6-sol", 1_000_000, 0, cached_input_tokens=1_000_000)
    sol_6_1 = p._compute_cost("gpt-6.1-sol", 1_000_000, 0, cached_input_tokens=1_000_000)
    assert sol_6 == pytest.approx(0.20)
    assert sol_6_1 == pytest.approx(0.10)


def test_sol_6_1_capabilities_mirror_gpt_6_sol() -> None:
    provider = _provider()
    provider.model = None
    for attr, args in (
        ("get_default_max_tokens", ()),
        ("_is_reasoning_model", ()),
        ("supports_vision", ()),
        ("supports_audio", ()),
    ):
        fn = getattr(provider, attr)
        assert fn("gpt-6.1-sol", *args) == fn("gpt-6-sol", *args), attr
    assert provider.get_default_max_tokens("gpt-6.1-sol") == 128_000
    assert provider._is_reasoning_model("gpt-6.1-sol") is True
    assert provider.supports_vision("gpt-6.1-sol") is True
    assert provider.supports_audio("gpt-6.1-sol") is False


@pytest.mark.parametrize("effort", ["max", "ultra"])
def test_sol_6_1_passes_native_efforts_through(effort: str) -> None:
    provider = _provider()
    provider.model = "gpt-6.1-sol"
    request = CompletionRequest(
        messages=[Message(role=Role.USER, content="Solve this")],
        thinking=ThinkingConfig(enabled=True, effort=effort),
    )
    assert provider._build_chat_params(request)["reasoning_effort"] == effort
    assert provider._build_responses_params(request)["reasoning"] == {"effort": effort}


def test_sol_6_1_fallback_chain() -> None:
    assert OpenAIProvider._FALLBACKS["gpt-6.1-sol"] == [
        "gpt-6-sol",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
    ]
    assert OpenAIProvider._FALLBACKS["gpt-6-astra"] == [
        "gpt-6.1-sol",
        "gpt-6-sol",
        "gpt-5.6-sol",
        "gpt-5.5",
    ]
