"""Regression tests for the GPT-5.6 model family addition (Sol / Terra / Luna).

Covers:
  * Tier aliases route ``top``/``expensive``/``medium`` to the GPT-5.6 family.
  * ``_PRICE_TABLE`` returns the GPT-5.6 rates, including the bare ``gpt-5.6``
    alias (routes to Sol pricing).
  * More-specific keys (``gpt-5.6-sol``) match before the bare ``gpt-5.6`` key.
  * Cost computation for Sol/Terra/Luna, cached and uncached.
  * ``_FALLBACKS`` chains for the GPT-5.6 family resolve.
  * ``get_default_max_tokens`` / ``_is_reasoning_model`` recognise gpt-5.6-*
    automatically via the existing "gpt-5" prefix checks.
"""

from __future__ import annotations

import pytest

from lazybridge.core.providers.openai import _PRICE_TABLE, OpenAIProvider


def _provider() -> OpenAIProvider:
    """Build a provider without hitting the network or requiring an API key."""
    return OpenAIProvider.__new__(OpenAIProvider)


def test_tier_aliases_route_to_gpt_5_6_family() -> None:
    assert OpenAIProvider._TIER_ALIASES["top"] == "gpt-5.6-sol"
    assert OpenAIProvider._TIER_ALIASES["expensive"] == "gpt-5.6-terra"
    assert OpenAIProvider._TIER_ALIASES["medium"] == "gpt-5.6-luna"


def test_price_table_has_gpt_5_6_entries() -> None:
    assert _PRICE_TABLE["gpt-5.6-sol"] == (5.0, 0.50, 30.0)
    assert _PRICE_TABLE["gpt-5.6-terra"] == (2.50, 0.25, 15.0)
    assert _PRICE_TABLE["gpt-5.6-luna"] == (1.0, 0.10, 6.0)


def test_bare_gpt_5_6_alias_matches_sol_pricing() -> None:
    assert _PRICE_TABLE["gpt-5.6"] == _PRICE_TABLE["gpt-5.6-sol"]


def test_price_table_more_specific_keys_match_first() -> None:
    """gpt-5.6-sol/terra/luna must appear before the bare gpt-5.6 key so a
    literal 'gpt-5.6-sol' model string doesn't match the generic alias row
    when the specific one is available."""
    keys = list(_PRICE_TABLE.keys())
    assert keys.index("gpt-5.6-sol") < keys.index("gpt-5.6")
    assert keys.index("gpt-5.6-terra") < keys.index("gpt-5.6")
    assert keys.index("gpt-5.6-luna") < keys.index("gpt-5.6")
    assert keys.index("gpt-5.6") < keys.index("gpt-5.5")


def test_compute_cost_gpt_5_6_sol_uncached() -> None:
    cost = _provider()._compute_cost("gpt-5.6-sol", input_tokens=1_000_000, output_tokens=0)
    assert cost == pytest.approx(5.0)


def test_compute_cost_gpt_5_6_luna_with_cache_hit() -> None:
    cost = _provider()._compute_cost(
        "gpt-5.6-luna",
        input_tokens=1_000_000,
        output_tokens=0,
        cached_input_tokens=400_000,
    )
    expected = (600_000 * 1.0 + 400_000 * 0.10) / 1_000_000
    assert cost == pytest.approx(expected)


def test_compute_cost_bare_gpt_5_6_alias_resolves_to_sol_rate() -> None:
    cost = _provider()._compute_cost("gpt-5.6", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(5.0 + 30.0)


def test_get_default_max_tokens_gpt_5_6_family() -> None:
    p = _provider()
    p.model = None
    for model in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6"):
        assert p.get_default_max_tokens(model) == 128_000


def test_fallback_chains_for_gpt_5_6_family() -> None:
    assert OpenAIProvider._FALLBACKS["gpt-5.6-sol"] == ["gpt-5.6-terra", "gpt-5.5-pro", "gpt-5.5"]
    assert OpenAIProvider._FALLBACKS["gpt-5.6-terra"] == ["gpt-5.6-luna", "gpt-5.5"]
    assert OpenAIProvider._FALLBACKS["gpt-5.6-luna"] == ["gpt-5.4-mini", "gpt-5.4"]


def test_is_reasoning_model_recognises_gpt_5_6() -> None:
    p = _provider()
    assert p._is_reasoning_model("gpt-5.6-sol") is True
    assert p._is_reasoning_model("gpt-5.6-terra") is True
    assert p._is_reasoning_model("gpt-5.6-luna") is True


def test_vision_capable_patterns_cover_gpt_5_6() -> None:
    assert any(pattern in "gpt-5.6-sol" for pattern in OpenAIProvider._VISION_CAPABLE_MODEL_PATTERNS)
