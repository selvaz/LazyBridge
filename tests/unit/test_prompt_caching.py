"""Tests for the prompt-caching interface (``CacheConfig`` + per-provider wiring).

A framework-level opt-in for Anthropic prompt caching so large static
system prompts (typical in quant / compliance workflows) cost ~10% per
subsequent call instead of full rate.

Coverage strategy: the Anthropic SDK itself isn't invoked in unit tests
(no credentials), so we exercise the pure ``_build_params`` path which
is where cache_control markers are stamped onto the request dict.
"""

from __future__ import annotations

from lazybridge.core.providers.anthropic import AnthropicProvider
from lazybridge.core.types import (
    CacheConfig,
    CompletionRequest,
    Message,
    Role,
    ToolDefinition,
)


def _req(
    *,
    cache: CacheConfig | None,
    system: str | None = "You are a quant research agent.",
    tools: list[ToolDefinition] | None = None,
) -> CompletionRequest:
    return CompletionRequest(
        messages=[Message(role=Role.USER, content="hi")],
        system=system,
        tools=tools or [],
        cache=cache,
    )


def _fake_provider() -> AnthropicProvider:
    """Build an AnthropicProvider without triggering SDK init."""
    p = AnthropicProvider.__new__(AnthropicProvider)
    p.model = "claude-opus-4-7"
    p._temperature_warned = False
    # BaseProvider attributes accessed by _build_params / _messages_to_anthropic
    p._native_tools_warned = False
    return p


def _tool(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"desc for {name}",
        parameters={"type": "object", "properties": {}, "required": []},
    )


# ---------------------------------------------------------------------------
# System-prompt caching
# ---------------------------------------------------------------------------


def test_cache_none_keeps_system_as_plain_string() -> None:
    """Regression: when caching is off, Anthropic still gets a plain
    string system prompt.  No behaviour change from v1."""
    p = _fake_provider()
    params = p._build_params(_req(cache=None))
    assert params["system"] == "You are a quant research agent."


def test_cache_enabled_converts_system_to_cache_control_block() -> None:
    p = _fake_provider()
    params = p._build_params(_req(cache=CacheConfig(enabled=True)))
    system = params["system"]
    assert isinstance(system, list) and len(system) == 1
    assert system[0]["type"] == "text"
    assert system[0]["text"] == "You are a quant research agent."
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_cache_disabled_flag_keeps_plain_system() -> None:
    """``CacheConfig(enabled=False)`` is the explicit-off form; still a string."""
    p = _fake_provider()
    params = p._build_params(_req(cache=CacheConfig(enabled=False)))
    assert params["system"] == "You are a quant research agent."


def test_cache_ttl_1h_emits_ttl_key() -> None:
    p = _fake_provider()
    params = p._build_params(_req(cache=CacheConfig(enabled=True, ttl="1h")))
    cc = params["system"][0]["cache_control"]
    assert cc == {"type": "ephemeral", "ttl": "1h"}


def test_cache_ttl_5m_omits_ttl_key() -> None:
    """Default TTL is 5m; emit no ``ttl`` key so Anthropic uses the default
    and the request stays minimal."""
    p = _fake_provider()
    params = p._build_params(_req(cache=CacheConfig(enabled=True, ttl="5m")))
    cc = params["system"][0]["cache_control"]
    assert "ttl" not in cc


# ---------------------------------------------------------------------------
# Tools caching
# ---------------------------------------------------------------------------


def test_cache_enabled_stamps_last_tool_only() -> None:
    p = _fake_provider()
    tools = [_tool("search"), _tool("rank"), _tool("summarize")]
    params = p._build_params(_req(cache=CacheConfig(enabled=True), tools=tools))
    built = params["tools"]
    # Only the LAST tool carries cache_control — Anthropic caches the
    # full prefix up to that marker, so one breakpoint is enough.
    assert "cache_control" not in built[0]
    assert "cache_control" not in built[1]
    assert built[-1]["cache_control"] == {"type": "ephemeral"}


def test_cache_none_leaves_tools_untouched() -> None:
    p = _fake_provider()
    tools = [_tool("search"), _tool("rank")]
    params = p._build_params(_req(cache=None, tools=tools))
    for t in params["tools"]:
        assert "cache_control" not in t


def test_cache_tools_and_system_both_marked_when_present() -> None:
    p = _fake_provider()
    params = p._build_params(
        _req(
            cache=CacheConfig(enabled=True),
            tools=[_tool("analyze")],
        )
    )
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert params["tools"][-1]["cache_control"] == {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# LLMEngine / Agent wiring
# ---------------------------------------------------------------------------


def test_llm_engine_cache_true_builds_default_config() -> None:
    from lazybridge.engines.llm import LLMEngine

    eng = LLMEngine("claude-opus-4-7", cache=True)
    assert isinstance(eng.cache, CacheConfig)
    assert eng.cache.enabled is True
    assert eng.cache.ttl == "5m"


def test_llm_engine_cache_false_is_none() -> None:
    from lazybridge.engines.llm import LLMEngine

    eng = LLMEngine("claude-opus-4-7", cache=False)
    assert eng.cache is None


def test_llm_engine_cache_passthrough_of_explicit_config() -> None:
    from lazybridge.engines.llm import LLMEngine

    cfg = CacheConfig(enabled=True, ttl="1h")
    eng = LLMEngine("claude-opus-4-7", cache=cfg)
    assert eng.cache is cfg


def test_agent_forwards_cache_flag_to_auto_built_engine() -> None:
    from lazybridge import Agent

    a = Agent("claude-opus-4-7", cache=True)
    # Engine is auto-built from model string; cache should propagate.
    assert a.engine.cache is not None
    assert a.engine.cache.enabled is True


def test_agent_cache_false_default_keeps_caching_off() -> None:
    from lazybridge import Agent

    a = Agent("claude-opus-4-7")  # default cache=False
    assert a.engine.cache is None


# ---------------------------------------------------------------------------
# Other providers: CacheConfig is accepted but doesn't break anything
# ---------------------------------------------------------------------------


def test_cache_config_on_non_anthropic_request_does_not_raise() -> None:
    """OpenAI / Google / DeepSeek ignore ``request.cache`` today —
    accepting it silently is the documented forward-compat path."""
    # Construct the request itself; that's where cross-provider contract lives.
    req = CompletionRequest(
        messages=[Message(role=Role.USER, content="hi")],
        system="sys",
        cache=CacheConfig(enabled=True),
    )
    # No exception, no validation error — the field just rides along.
    assert req.cache is not None
    assert req.cache.enabled is True
