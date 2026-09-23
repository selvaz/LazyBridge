"""Regression tests for GPT-6 Astra support."""

from __future__ import annotations

import pytest

from lazybridge.core.providers.openai import _PRICE_TABLE, OpenAIProvider
from lazybridge.core.types import CompletionRequest, Message, Role, ThinkingConfig


def _provider() -> OpenAIProvider:
    """Build a provider without requiring the optional SDK or an API key."""
    return OpenAIProvider.__new__(OpenAIProvider)


def test_astra_is_the_top_openai_tier() -> None:
    assert OpenAIProvider._TIER_ALIASES["top"] == "gpt-6-astra"


def test_astra_has_published_standard_prices() -> None:
    assert _PRICE_TABLE["gpt-6-astra"] == (10.0, 1.0, 50.0)


def test_astra_cost_accounts_for_cached_and_uncached_input() -> None:
    cost = _provider()._compute_cost(
        "gpt-6-astra",
        input_tokens=1_000_000,
        output_tokens=100_000,
        cached_input_tokens=250_000,
    )
    expected = 0.75 * 10.0 + 0.25 * 1.0 + 0.1 * 50.0
    assert cost == pytest.approx(expected)


def test_astra_core_capabilities() -> None:
    provider = _provider()
    provider.model = None
    assert provider.get_default_max_tokens("gpt-6-astra") == 128_000
    assert provider._is_reasoning_model("gpt-6-astra") is True
    assert provider.supports_vision("gpt-6-astra") is True
    assert provider.supports_audio("gpt-6-astra") is False


def test_astra_preserves_native_max_effort_on_both_api_paths() -> None:
    provider = _provider()
    provider.model = "gpt-6-astra"
    request = CompletionRequest(
        messages=[Message(role=Role.USER, content="Solve this")],
        thinking=ThinkingConfig(enabled=True, effort="max"),
    )
    assert provider._build_chat_params(request)["reasoning_effort"] == "max"
    assert provider._build_responses_params(request)["reasoning"] == {"effort": "max"}


def test_astra_falls_back_to_available_gpt_5_models() -> None:
    assert OpenAIProvider._FALLBACKS["gpt-6-astra"] == [
        "gpt-6-sol",
        "gpt-5.6-sol",
        "gpt-5.5",
    ]
