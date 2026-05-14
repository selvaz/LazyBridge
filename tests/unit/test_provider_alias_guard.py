"""Bare-provider-alias guard on ``LLMEngine``.

``LLMEngine("anthropic")`` used to construct successfully and silently
defer failure to the provider API (which then rejected the literal
model id ``"anthropic"`` several RTTs later).  0.7.9.x raises at
construction with a fix snippet pointing at the two canonical forms:

* ``Agent.from_provider("anthropic", tier="medium")`` — tier alias
* ``Agent(engine=LLMEngine("claude-opus-4-7"))``       — pinned id

The guard fires only on the inference path (``provider=`` not passed
explicitly).  ``Agent.from_provider`` threads ``provider=`` alongside
a tier alias and therefore stays allowed.
"""

from __future__ import annotations

import pytest

from lazybridge import Agent, LLMEngine


@pytest.mark.parametrize("alias", ["anthropic", "openai", "google", "deepseek"])
def test_bare_provider_alias_as_model_raises(alias: str) -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        LLMEngine(alias)


@pytest.mark.parametrize("alias", ["anthropic", "openai", "google", "deepseek"])
def test_bare_provider_alias_via_agent_sugar_raises(alias: str) -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        Agent(alias)


def test_family_alias_claude_is_not_a_bare_provider_name() -> None:
    """``"claude"`` is a model-family alias mapping ``anthropic`` via the
    routing table, not a self-mapping provider name.  It stays allowed so
    historical ``Agent("claude")`` callers still work."""
    engine = LLMEngine("claude")
    assert engine.provider == "anthropic"


def test_from_provider_tier_alias_still_works() -> None:
    """``Agent.from_provider`` passes ``provider=`` explicitly, which
    bypasses the guard — the tier string is consumed by the provider's
    own ``_TIER_ALIASES`` table at request time."""
    a = Agent.from_provider("anthropic", tier="top")
    assert a.engine.provider == "anthropic"
    assert a.engine.model == "top"


def test_pinned_model_id_still_works() -> None:
    engine = LLMEngine("claude-opus-4-7")
    assert engine.provider == "anthropic"
    assert engine.model == "claude-opus-4-7"


def test_guard_error_message_names_both_fix_paths() -> None:
    with pytest.raises(ValueError) as exc:
        LLMEngine("anthropic")
    msg = str(exc.value)
    assert "from_provider" in msg
    assert "LLMEngine" in msg
