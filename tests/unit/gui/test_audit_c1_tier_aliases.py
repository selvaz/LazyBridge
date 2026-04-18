"""Regression tests for audit F2 — tier aliases on provider.model."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lazybridge.core.providers.anthropic import AnthropicProvider
from lazybridge.core.providers.deepseek import DeepSeekProvider
from lazybridge.core.providers.google import GoogleProvider
from lazybridge.core.providers.openai import OpenAIProvider
from lazybridge.core.types import CompletionRequest, Message, Role


PROVIDERS = [AnthropicProvider, OpenAIProvider, GoogleProvider, DeepSeekProvider]


@pytest.fixture(params=PROVIDERS)
def provider(request):
    """A provider instance without network side-effects — __new__'d and
    populated with just enough state for _resolve_model to run."""
    cls = request.param
    p = cls.__new__(cls)
    p.model = cls.default_model
    return p


def _req(model: str | None) -> CompletionRequest:
    return CompletionRequest(
        model=model,
        messages=[Message(role=Role.USER, content="hi")],
    )


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tier", ["top", "expensive", "medium", "cheap", "super_cheap"])
def test_each_tier_resolves_to_a_concrete_model(provider, tier):
    resolved = provider._resolve_model(_req(tier))
    assert resolved != tier, (
        f"{type(provider).__name__}: tier {tier!r} did not resolve"
    )
    # The resolved value must appear in the class's tier table.
    assert resolved == provider._TIER_ALIASES[tier]


def test_literal_model_passes_through_unchanged(provider):
    literal = "my-custom-fine-tune-v7"
    assert provider._resolve_model(_req(literal)) == literal


def test_unknown_tier_passes_through_as_literal(provider):
    # "mega_premium" is not a recognised tier — treat it as a literal.
    assert provider._resolve_model(_req("mega_premium")) == "mega_premium"


def test_default_model_resolves_to_class_default_when_none(provider):
    # request.model=None, instance.model=default — should return default.
    resolved = provider._resolve_model(_req(None))
    # Default may be a literal (already-resolved) or a tier alias.  Both
    # cases must yield a non-empty string that's not one of our tier words.
    assert resolved
    assert resolved not in {"top", "expensive", "medium", "cheap", "super_cheap"}


# ---------------------------------------------------------------------------
# Structural invariants across all four providers
# ---------------------------------------------------------------------------


ALL_TIERS = {"top", "expensive", "medium", "cheap", "super_cheap"}


@pytest.mark.parametrize("cls", PROVIDERS)
def test_every_provider_defines_all_five_tiers(cls):
    missing = ALL_TIERS - set(cls._TIER_ALIASES)
    assert not missing, f"{cls.__name__} is missing tiers: {missing}"


@pytest.mark.parametrize("cls", PROVIDERS)
def test_fallbacks_reference_known_models_only(cls):
    """Every fallback chain should point at plausible concrete model
    names (non-empty strings) — catches obvious typos."""
    for src, chain in cls._FALLBACKS.items():
        assert src, f"empty fallback source in {cls.__name__}"
        assert isinstance(chain, list) and all(isinstance(m, str) and m for m in chain), (
            f"bad fallback entry for {src!r} in {cls.__name__}: {chain!r}"
        )


# ---------------------------------------------------------------------------
# End-to-end via the agent's resolver path
# ---------------------------------------------------------------------------


def test_instance_model_set_to_tier_resolves_on_each_call():
    """A provider whose ``self.model="cheap"`` must return the concrete
    cheap-tier model every time ``_resolve_model`` is called, even when
    the request leaves ``request.model`` unset."""
    p = AnthropicProvider.__new__(AnthropicProvider)
    p.model = "cheap"  # simulate LazyAgent("anthropic", model="cheap")
    resolved = p._resolve_model(_req(None))
    assert resolved == AnthropicProvider._TIER_ALIASES["cheap"]
    assert resolved == "claude-haiku-4-5"
