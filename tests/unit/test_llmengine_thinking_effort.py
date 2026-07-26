"""``LLMEngine(thinking=...)`` accepts bool / str / ThinkingConfig.

The ``str`` shorthand (an effort level) was added alongside the Claude 5 /
GPT-5.6 model updates, since ``effort`` is now the primary reasoning-depth
lever for both providers (``ThinkingConfig.effort``, read by both
``AnthropicProvider`` and ``OpenAIProvider``). Covers:
  * ``True`` / ``False`` keep working exactly as before.
  * A recognised effort string is stored as-is and normalised to a
    ``ThinkingConfig`` at request-build time.
  * An unrecognised string raises ``ValueError`` at construction (fail fast,
    not on the first request).
  * A full ``ThinkingConfig`` instance passes through unchanged.
"""

from __future__ import annotations

import pytest

from lazybridge.core.types import ThinkingConfig
from lazybridge.engines.llm import LLMEngine


def test_thinking_accepts_bool() -> None:
    assert LLMEngine("claude-opus-5", thinking=True).thinking is True
    assert LLMEngine("claude-opus-5", thinking=False).thinking is False


def test_thinking_accepts_valid_effort_string() -> None:
    engine = LLMEngine("claude-opus-5", thinking="low")
    assert engine.thinking == "low"


def test_thinking_rejects_unknown_string() -> None:
    with pytest.raises(ValueError, match="not a recognised effort level"):
        LLMEngine("claude-opus-5", thinking="ludicrous")


def test_thinking_accepts_thinking_config_instance() -> None:
    cfg = ThinkingConfig(enabled=True, effort="xhigh", display="omitted")
    engine = LLMEngine("claude-opus-5", thinking=cfg)
    assert engine.thinking is cfg


def test_thinking_default_is_false() -> None:
    assert LLMEngine("claude-opus-5").thinking is False
