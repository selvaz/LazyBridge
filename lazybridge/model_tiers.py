"""One-to-one tier correspondence between Anthropic and OpenAI coding models.

Owner decision 2026-10-31 (ChatGPT "Terra 6.1" completes the OpenAI ladder):

=====  ==========================  ==================
tier   Anthropic                   OpenAI (Codex)
=====  ==========================  ==================
small  claude-haiku-5-5  (haiku)   gpt-6-luna
mid    claude-sonnet-5-5 (sonnet)  gpt-6.1-terra
high   claude-opus-5-5   (opus)    gpt-6.1-sol
top    claude-fable-5-1  (fable)   gpt-6-astra
=====  ==========================  ==================

:func:`translate_model` maps a model name from the other provider's
namespace into the target one, so an engine that was handed ``"sonnet"``
while running on Codex gets ``gpt-6.1-terra`` instead of a model Codex
rejects (and vice versa).  Names that are not part of the other provider's
ladder (explicit pins such as ``gpt-5.6-sol`` or ``claude-sonnet-4-5``) pass
through untouched.
"""

from __future__ import annotations

from typing import Literal

Family = Literal["anthropic", "openai"]
ModelTier = Literal["small", "mid", "high", "top"]

#: The canonical table: tier -> {family: model id}.
MODEL_TIERS: dict[str, dict[str, str]] = {
    "small": {"anthropic": "claude-haiku-5-5", "openai": "gpt-6-luna"},
    "mid": {"anthropic": "claude-sonnet-5-5", "openai": "gpt-6.1-terra"},
    "high": {"anthropic": "claude-opus-5-5", "openai": "gpt-6.1-sol"},
    "top": {"anthropic": "claude-fable-5-1", "openai": "gpt-6-astra"},
}

_CLAUDE_MARKERS = (("haiku", "small"), ("sonnet", "mid"), ("opus", "high"), ("fable", "top"))
_OPENAI_MARKERS = (("luna", "small"), ("terra", "mid"), ("sol", "high"), ("astra", "top"))


def _family(engine_or_family: str) -> Family:
    key = engine_or_family.strip().lower()
    if key in ("anthropic", "claude", "claude_code", "claude-code"):
        return "anthropic"
    if key in ("openai", "codex"):
        return "openai"
    raise ValueError(f"unknown provider/engine {engine_or_family!r}")


def claude_tier(model: str) -> str | None:
    """Tier of a Claude alias or ``claude-*`` id, else ``None``."""
    name = model.strip().lower()
    if not (name.startswith("claude-") or name in {m for m, _ in _CLAUDE_MARKERS}):
        return None
    for marker, tier in _CLAUDE_MARKERS:
        if marker in name:
            return tier
    return None


def openai_tier(model: str) -> str | None:
    """Tier of a ``gpt-*`` Luna/Terra/Sol/Astra id, else ``None``."""
    name = model.strip().lower()
    if not name.startswith("gpt-"):
        return None
    for marker, tier in _OPENAI_MARKERS:
        if name.endswith("-" + marker):
            return tier
    return None


def translate_model(model: str | None, to: str) -> str | None:
    """Return ``model`` expressed in the ``to`` provider's ladder.

    ``to`` is a provider or engine name (``"openai"``/``"codex"`` or
    ``"anthropic"``/``"claude_code"``).  Only names that belong to the *other*
    provider's ladder are translated; everything else (including ``None``) is
    returned unchanged.
    """
    if model is None:
        return None
    target = _family(to)
    if target == "openai":
        tier = claude_tier(model)
    else:
        tier = openai_tier(model)
    return MODEL_TIERS[tier][target] if tier else model


__all__ = ["MODEL_TIERS", "claude_tier", "openai_tier", "translate_model"]
