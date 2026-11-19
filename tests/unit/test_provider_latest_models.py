"""Regression coverage for the November 2026 provider tier refresh."""

import pytest

from lazybridge.core.providers.anthropic import (
    _NO_FORCED_TOOL_CHOICE_MODELS,
    AnthropicProvider,
)
from lazybridge.core.providers.anthropic import (
    _PRICE_TABLE as ANTHROPIC_PRICES,
)
from lazybridge.core.providers.openai import _PRICE_TABLE as OPENAI_PRICES
from lazybridge.core.providers.openai import OpenAIProvider


def test_openai_62_tiers_prices_and_fallbacks() -> None:
    assert OpenAIProvider._TIER_ALIASES["expensive"] == "gpt-6.2-sol"
    assert OpenAIProvider._TIER_ALIASES["top"] == "gpt-6.2-astra"
    assert OPENAI_PRICES["gpt-6.2-sol"] == (2.0, 0.10, 10.0)
    assert OPENAI_PRICES["gpt-6.2-astra"] == (8.0, 0.40, 40.0)
    assert OpenAIProvider._FALLBACKS["gpt-6.2-sol"][0] == "gpt-6.1-sol"
    assert OpenAIProvider._FALLBACKS["gpt-6.2-astra"][0] == "gpt-6-astra"


def test_openai_62_costing_prefers_specific_rows() -> None:
    provider = OpenAIProvider.__new__(OpenAIProvider)
    assert provider._compute_cost("gpt-6.2-astra", 1_000_000, 1_000_000, 1_000_000) == pytest.approx(40.4)
    assert provider._compute_cost("gpt-6.2-sol", 1_000_000, 1_000_000, 1_000_000) == pytest.approx(10.1)


def test_fable_52_tier_price_cache_and_fallback() -> None:
    assert AnthropicProvider._TIER_ALIASES["top"] == "claude-fable-5-2"
    assert ANTHROPIC_PRICES["claude-fable-5-2"] == (8.0, 40.0)
    assert AnthropicProvider._FALLBACKS["claude-fable-5-2"] == [
        "claude-fable-5-1",
        "claude-opus-5-5",
    ]
    assert "claude-fable-5-2" in _NO_FORCED_TOOL_CHOICE_MODELS
    provider = AnthropicProvider.__new__(AnthropicProvider)
    assert provider._compute_cost("claude-fable-5-2", 0, 0, 1_000_000) == pytest.approx(0.20)
