"""Regression tests for the GPT-5.5 model family addition.

Covers:
  * Tier aliases route ``top``/``expensive`` to the GPT-5.5 family.
  * ``_PRICE_TABLE`` returns the GPT-5.5 short-context rates.
  * Cost computation splits cached vs uncached input tokens at the
    published rate, and falls back to the input rate when the cache
    rate is unknown.
  * ``_populate_cached_input_tokens`` reads both Chat Completions
    (``prompt_tokens_details.cached_tokens``) and Responses
    (``input_tokens_details.cached_tokens``) shapes.
  * ``_EFFORT_MAP`` exposes the new ``"none"`` reasoning effort.
  * ``default_model`` is ``gpt-5.5``.
  * ``get_default_max_tokens`` returns 128 K for ``gpt-5.5``.
  * ``_FALLBACKS`` chains for the GPT-5.5 family resolve.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lazybridge.core.providers.openai import (
    _EFFORT_MAP,
    _PRICE_TABLE,
    OpenAIProvider,
)
from lazybridge.core.types import UsageStats


def _provider() -> OpenAIProvider:
    """Build a provider without hitting the network or requiring an API key."""
    return OpenAIProvider.__new__(OpenAIProvider)


def test_default_model_is_gpt_5_5() -> None:
    assert OpenAIProvider.default_model == "gpt-5.5"


def test_tier_aliases_route_to_gpt_5_5_family() -> None:
    assert OpenAIProvider._TIER_ALIASES["top"] == "gpt-5.5-pro"
    assert OpenAIProvider._TIER_ALIASES["expensive"] == "gpt-5.5"


def test_price_table_has_gpt_5_5_entries() -> None:
    assert _PRICE_TABLE["gpt-5.5"] == (5.0, 0.50, 30.0)
    assert _PRICE_TABLE["gpt-5.5-pro"] == (30.0, None, 180.0)


def test_price_table_more_specific_keys_match_first() -> None:
    """gpt-5.5-pro must appear before gpt-5.5 so substring-match picks the right row."""
    keys = list(_PRICE_TABLE.keys())
    assert keys.index("gpt-5.5-pro") < keys.index("gpt-5.5")
    assert keys.index("gpt-5.5") < keys.index("gpt-5.4")


def test_compute_cost_gpt_5_5_uncached() -> None:
    cost = _provider()._compute_cost("gpt-5.5", input_tokens=1_000_000, output_tokens=0)
    assert cost == pytest.approx(5.0)


def test_compute_cost_gpt_5_5_with_cache_hit() -> None:
    """Cached tokens are billed at $0.50/M, uncached at $5/M."""
    cost = _provider()._compute_cost(
        "gpt-5.5",
        input_tokens=1_000_000,
        output_tokens=0,
        cached_input_tokens=400_000,
    )
    expected = (600_000 * 5.0 + 400_000 * 0.50) / 1_000_000
    assert cost == pytest.approx(expected)


def test_compute_cost_falls_back_to_input_rate_when_cache_price_unknown() -> None:
    """gpt-5.5-pro has cached_price=None; cached tokens are billed at full input rate."""
    cost = _provider()._compute_cost(
        "gpt-5.5-pro",
        input_tokens=1_000_000,
        output_tokens=0,
        cached_input_tokens=500_000,
    )
    assert cost == pytest.approx(30.0)


def test_compute_cost_clamps_cached_tokens_to_input_total() -> None:
    """If the API ever reports cached > input (sanity), don't go negative."""
    cost = _provider()._compute_cost(
        "gpt-5.5",
        input_tokens=100,
        output_tokens=0,
        cached_input_tokens=500,
    )
    assert cost == pytest.approx(100 * 0.50 / 1_000_000)


def test_compute_cost_unknown_model_returns_none() -> None:
    assert _provider()._compute_cost("totally-unknown-model", 100, 100) is None


def test_populate_cached_input_chat_completions_shape() -> None:
    raw_usage = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=200,
        prompt_tokens_details=SimpleNamespace(cached_tokens=400),
    )
    usage = UsageStats(input_tokens=1000, output_tokens=200)
    OpenAIProvider._populate_cached_input_tokens(usage, raw_usage)
    assert usage.cached_input_tokens == 400


def test_populate_cached_input_responses_shape() -> None:
    raw_usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=200,
        input_tokens_details=SimpleNamespace(cached_tokens=250),
    )
    usage = UsageStats(input_tokens=1000, output_tokens=200)
    OpenAIProvider._populate_cached_input_tokens(usage, raw_usage)
    assert usage.cached_input_tokens == 250


def test_populate_cached_input_handles_missing_details() -> None:
    raw_usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    usage = UsageStats(input_tokens=10, output_tokens=5)
    OpenAIProvider._populate_cached_input_tokens(usage, raw_usage)
    assert usage.cached_input_tokens == 0


def test_effort_map_exposes_none() -> None:
    assert _EFFORT_MAP["none"] == "none"


def test_effort_map_preserves_existing_levels() -> None:
    for level in ("low", "medium", "high", "xhigh"):
        assert _EFFORT_MAP[level] == level
    assert _EFFORT_MAP["max"] == "xhigh"


def test_get_default_max_tokens_gpt_5_5() -> None:
    p = _provider()
    p.model = None
    assert p.get_default_max_tokens("gpt-5.5") == 128_000
    assert p.get_default_max_tokens("gpt-5.5-pro") == 128_000


def test_fallback_chains_for_gpt_5_5_family() -> None:
    assert OpenAIProvider._FALLBACKS["gpt-5.5"] == ["gpt-5.4", "gpt-5"]
    assert OpenAIProvider._FALLBACKS["gpt-5.5-pro"] == [
        "gpt-5.5",
        "gpt-5.4-pro",
        "gpt-5.4",
    ]


def test_is_reasoning_model_recognises_gpt_5_5() -> None:
    p = _provider()
    assert p._is_reasoning_model("gpt-5.5") is True
    assert p._is_reasoning_model("gpt-5.5-pro") is True
