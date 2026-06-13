"""Unit tests for DeepSeekProvider — pure-Python paths only.

DeepSeek inherits much of the OpenAI provider's wire shape, so this
file focuses on the DeepSeek-specific behaviours:

* B1: ``_ensure_json_word_in_prompt`` rebuilds the list rather than
  mutating ``params['messages']`` in place (regression test lives in
  ``test_audit_phase1_regressions.py``; this file documents the
  contract from the provider angle).
* V4 vs legacy ``tools + structured_output`` policy — V4 honours both;
  legacy models drop ``structured_output`` with a one-shot warning.
* Capability ClassVars match the actual SDK surface.
* ``_compute_cost`` signature parity (``cached_input_tokens=`` accepted).

The OpenAI SDK is stubbed via ``sys.modules`` so this runs without the
real ``openai`` / ``deepseek`` packages installed — same approach as
``test_litellm_provider``.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# openai stub — the DeepSeek provider imports it lazily, but at __init__
# time it constructs an OpenAI(api_key=..., base_url=...) client.  Stub
# the minimum surface so the provider can instantiate without secrets.
# Installed per-test with cleanup (a module-level sys.modules entry would
# leak into the whole session and turn importorskip("openai") elsewhere
# into a fake pass).
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _openai_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    if "openai" not in sys.modules:
        openai_pkg = types.ModuleType("openai")
        openai_pkg.OpenAI = MagicMock(name="OpenAI")
        openai_pkg.AsyncOpenAI = MagicMock(name="AsyncOpenAI")
        monkeypatch.setitem(sys.modules, "openai", openai_pkg)


from lazybridge.core.providers.deepseek import DeepSeekProvider


def _bare_deepseek() -> DeepSeekProvider:
    """Skip ``__init__`` so we don't touch the (stubbed) client."""
    p = DeepSeekProvider.__new__(DeepSeekProvider)
    p.api_key = "fake"
    p.model = DeepSeekProvider.default_model
    p._structured_drop_warned = False
    return p


# ---------------------------------------------------------------------------
# Capability declarations
# ---------------------------------------------------------------------------


def test_supported_native_tools_is_empty():
    """DeepSeek doesn't expose Anthropic/OpenAI-style server-side native
    tools (no WEB_SEARCH / CODE_EXECUTION / etc.).  Callers asking for
    them get the standard ``UnsupportedNativeToolError`` path."""
    assert DeepSeekProvider.supported_native_tools == frozenset()


def test_capability_classvars_reflect_deepseek_features():
    """DeepSeek supports streaming and structured output, and the
    deepseek-reasoner family surfaces ``reasoning_content`` so
    ``supports_thinking=True`` is the right default."""
    assert DeepSeekProvider.supports_streaming is True
    assert DeepSeekProvider.supports_structured_output is True
    assert DeepSeekProvider.supports_thinking is True


# ---------------------------------------------------------------------------
# Cost telemetry parity (signature only — pricing values are not tested)
# ---------------------------------------------------------------------------


def test_compute_cost_signature_accepts_cached_input_tokens():
    import inspect

    sig = inspect.signature(DeepSeekProvider._compute_cost)
    params = list(sig.parameters.keys())
    assert "cached_input_tokens" in params


def test_compute_cost_unknown_model_returns_none():
    p = _bare_deepseek()
    assert p._compute_cost("not-a-deepseek", 100, 100) is None


def test_compute_cost_applies_cache_hit_rate():
    p = _bare_deepseek()
    # 1M input all cached: should use $0.003625 not $0.435
    cost_all_cached = p._compute_cost("deepseek-v4-pro", 1_000_000, 0, cached_input_tokens=1_000_000)
    assert cost_all_cached == pytest.approx(0.003625)
    # 1M input no cache: standard $0.435
    cost_no_cache = p._compute_cost("deepseek-v4-pro", 1_000_000, 0, cached_input_tokens=0)
    assert cost_no_cache == pytest.approx(0.435)
    # 1M total input, 500K cached + 500K uncached + 1M output for v4-flash
    cost_mixed = p._compute_cost("deepseek-v4-flash", 1_000_000, 1_000_000, cached_input_tokens=500_000)
    expected = (500_000 * 0.14 + 500_000 * 0.0028 + 1_000_000 * 0.28) / 1_000_000
    assert cost_mixed == pytest.approx(expected)


# ---------------------------------------------------------------------------
# B1 contract — _ensure_json_word_in_prompt never mutates the input list
# ---------------------------------------------------------------------------


def test_ensure_json_word_rebuilds_messages_list_no_user_prompt():
    """No system message, no existing 'json' keyword — provider must
    prepend a fresh system message in a NEW list (not mutate the user's)."""
    p = _bare_deepseek()
    original = [{"role": "user", "content": "do the thing"}]
    params = {"messages": list(original)}
    pre_id = id(params["messages"])

    p._ensure_json_word_in_prompt(params, schema=None)

    # New list identity, original untouched.
    assert id(params["messages"]) != pre_id
    assert original == [{"role": "user", "content": "do the thing"}]
    assert params["messages"][0]["role"] == "system"


def test_ensure_json_word_with_existing_system_dict_is_non_mutating():
    """When a system message already exists, the provider rebuilds the
    first slot with appended JSON guidance — the ORIGINAL system dict
    must not be touched (the same dict could be reused across calls)."""
    p = _bare_deepseek()
    sys_dict = {"role": "system", "content": "be terse"}
    params = {"messages": [sys_dict, {"role": "user", "content": "go"}]}

    p._ensure_json_word_in_prompt(params, schema=None)

    assert sys_dict == {"role": "system", "content": "be terse"}
    assert "json" in params["messages"][0]["content"].lower()


# ---------------------------------------------------------------------------
# V4-vs-legacy structured-output policy
# ---------------------------------------------------------------------------


def test_v4_model_is_classified_as_thinking_capable():
    """V4 models support ``response_format`` + ``tools`` simultaneously;
    the provider relies on a ``_THINKING_CAPABLE_MODELS`` allow-list
    to gate the behaviour."""
    from lazybridge.core.providers.deepseek import _THINKING_CAPABLE_MODELS

    # V4 / reasoner families are the ones that survive the joint use.
    assert any("v4" in m.lower() or "reasoner" in m.lower() for m in _THINKING_CAPABLE_MODELS)


def test_default_model_is_a_real_deepseek_id():
    """``default_model`` is the fallback when no per-request / per-instance
    model is set; must look like an actual DeepSeek SKU."""
    assert "deepseek" in DeepSeekProvider.default_model.lower()


# ---------------------------------------------------------------------------
# Tier aliases — Agent.from_provider('deepseek', tier='top') routes here
# ---------------------------------------------------------------------------


def test_tier_aliases_resolve_to_concrete_model_ids():
    tiers = DeepSeekProvider._TIER_ALIASES
    for tier_name in ("super_cheap", "cheap", "medium", "expensive", "top"):
        assert tier_name in tiers, f"missing tier alias: {tier_name}"
        assert "deepseek" in tiers[tier_name].lower()
