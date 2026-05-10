"""Unit tests for GoogleProvider — capability + signature surface.

Focuses on paths that don't require a full Gemini SDK stub:

* Capability ClassVars declared correctly.
* ``_compute_cost`` accepts ``cached_input_tokens=`` (signature parity
  with the rest of the providers since 0.7.9).
* Native-tool support listing matches the README claims.
* T9: ``WEB_SEARCH`` and ``GOOGLE_SEARCH`` are flagged as aliases on
  the supported set.

Deeper integration paths (finish_reason normalisation, grounding +
structured-output rejection) live in ``test_multimodal_*`` and the
audit-tier files where the full Gemini-SDK stub already exists.
"""

from __future__ import annotations

from lazybridge.core.providers.google import GoogleProvider
from lazybridge.core.types import NativeTool


def _bare_provider() -> GoogleProvider:
    """Construct without firing ``_init_client``.  Suitable for the
    pure-Python paths exercised here; calls that touch the real SDK
    would fail without a stub."""
    p = GoogleProvider.__new__(GoogleProvider)
    p.api_key = "fake"
    p.model = "gemini-2.5-flash"
    p.strict_native_tools = False
    return p


# ---------------------------------------------------------------------------
# Capability declarations — supported_native_tools is the source of truth
# ---------------------------------------------------------------------------


def test_supported_native_tools_includes_grounding_and_maps():
    """Gemini supports server-side grounding (web search) under two
    aliases — ``WEB_SEARCH`` and ``GOOGLE_SEARCH`` — plus the Maps
    grounding tool.  README + ``lazybridge.matrix`` rely on this."""
    expected = {NativeTool.WEB_SEARCH, NativeTool.GOOGLE_SEARCH, NativeTool.GOOGLE_MAPS}
    assert expected.issubset(GoogleProvider.supported_native_tools)


def test_capability_classvars_reflect_provider_features():
    """Phase-3 Block I — every provider declares ``supports_*`` flags
    explicitly.  The Gemini SDK supports all three modalities natively."""
    assert GoogleProvider.supports_streaming is True
    assert GoogleProvider.supports_structured_output is True
    assert GoogleProvider.supports_thinking is True


# ---------------------------------------------------------------------------
# Cost telemetry parity — _compute_cost(cached_input_tokens=0)
# ---------------------------------------------------------------------------


def test_compute_cost_accepts_cached_input_tokens_kwarg():
    """Signature parity with Anthropic / OpenAI / DeepSeek (Phase-3
    Block I).  Google doesn't model cache pricing yet — the kwarg
    is accepted but ignored — so a cached call costs the same as an
    uncached one."""
    p = _bare_provider()
    base = p._compute_cost("gemini-2.5-flash", input_tokens=1_000_000, output_tokens=0)
    cached = p._compute_cost(
        "gemini-2.5-flash",
        input_tokens=1_000_000,
        output_tokens=0,
        cached_input_tokens=500_000,
    )
    assert base == cached
    # Unknown models still return None — caller's signal that pricing isn't tracked.
    assert p._compute_cost("not-a-gemini", 100, 100) is None


def test_compute_cost_signature_is_polymorphic_with_base():
    """The ``BaseProvider._compute_cost`` signature was extended in 0.7.9
    to take ``cached_input_tokens=0``; every concrete provider matches
    so polymorphic callers don't have to special-case Google."""
    import inspect

    sig = inspect.signature(GoogleProvider._compute_cost)
    params = list(sig.parameters.keys())
    assert params[1:] == ["model", "input_tokens", "output_tokens", "cached_input_tokens"]


# ---------------------------------------------------------------------------
# WEB_SEARCH / GOOGLE_SEARCH alias declaration
# ---------------------------------------------------------------------------


def test_both_search_aliases_are_advertised_as_supported():
    """``NativeTool.WEB_SEARCH`` and ``NativeTool.GOOGLE_SEARCH`` are
    aliases at the Gemini-API level (only one grounding tool exists
    server-side, but LazyBridge accepts either name for ergonomic
    cross-provider portability).  Both must appear in the supported set
    so the user-facing list is consistent."""
    supported = GoogleProvider.supported_native_tools
    assert NativeTool.WEB_SEARCH in supported
    assert NativeTool.GOOGLE_SEARCH in supported


# ---------------------------------------------------------------------------
# Tier aliases — Gemini's tier table is the canonical model selector
# ---------------------------------------------------------------------------


def test_tier_aliases_resolve_to_concrete_model_ids():
    """``Agent.from_provider('google', tier='top')`` resolves the
    tier alias through ``_TIER_ALIASES``; every documented tier must
    map to a real model id."""
    tiers = GoogleProvider._TIER_ALIASES
    for tier_name in ("super_cheap", "cheap", "medium", "expensive", "top"):
        assert tier_name in tiers, f"missing tier alias: {tier_name}"
        assert isinstance(tiers[tier_name], str) and tiers[tier_name].startswith("gemini")
