"""Tests for the Phase 1 config-object refactor (2026-04-23).

Three dataclasses (``ResilienceConfig``, ``ObservabilityConfig``,
``AgentRuntimeConfig``) let callers bundle the resilience and
observability kwargs so a fleet of agents can share a single retry /
timeout / session policy instead of copy-pasting seven kwargs at every
call site.

Precedence:  flat kwarg > config object > default.  Mixing is allowed —
``Agent(resilience=cfg, timeout=30.0)`` uses the config's retries and
cache but overrides its timeout.

These tests guard the precedence rules and confirm existing flat-kwarg
call sites keep working bit-identically.
"""

from __future__ import annotations

from lazybridge import (
    Agent,
    AgentRuntimeConfig,
    CacheConfig,
    ObservabilityConfig,
    ResilienceConfig,
    Session,
)

# ---------------------------------------------------------------------------
# A. Backwards compatibility — flat kwargs untouched
# ---------------------------------------------------------------------------


def test_agent_default_construction_unchanged():
    """No kwargs passed → same defaults as before the refactor."""
    a = Agent("claude-opus-4-7")
    assert a.timeout is None
    assert a.max_output_retries == 2
    assert a.fallback is None
    assert a._verbose is False
    assert a.session is None
    assert a.description is None
    # Name defaults to the engine model string.
    assert a.name == "claude-opus-4-7"


def test_flat_kwargs_still_work():
    """Every flat kwarg still lands on the expected attribute."""
    a = Agent(
        "claude-opus-4-7",
        timeout=60.0,
        max_retries=5,
        retry_delay=2.0,
        max_output_retries=4,
        verbose=True,
        name="primary",
        description="the main one",
    )
    assert a.timeout == 60.0
    assert a.max_output_retries == 4
    assert a.engine.max_retries == 5
    assert a.engine.retry_delay == 2.0
    assert a._verbose is True
    assert a.name == "primary"
    assert a.description == "the main one"


# ---------------------------------------------------------------------------
# B. ResilienceConfig unpacking
# ---------------------------------------------------------------------------


def test_resilience_config_fills_in_resilience_kwargs():
    """Every field on ResilienceConfig routes to the matching Agent attr."""
    backup = Agent("gpt-4o", name="backup")
    cfg = ResilienceConfig(
        timeout=120.0,
        max_retries=7,
        retry_delay=3.0,
        cache=CacheConfig(ttl="1h"),
        max_output_retries=5,
        fallback=backup,
    )
    a = Agent("claude-opus-4-7", resilience=cfg)
    assert a.timeout == 120.0
    assert a.engine.max_retries == 7
    assert a.engine.retry_delay == 3.0
    assert a.engine.cache is cfg.cache
    assert a.max_output_retries == 5
    assert a.fallback is backup


def test_resilience_output_validator_fills_in():
    """output_validator is also covered by the config."""

    def validator(x):
        return x

    cfg = ResilienceConfig(output_validator=validator)
    a = Agent("claude-opus-4-7", resilience=cfg)
    assert a.output_validator is validator


# ---------------------------------------------------------------------------
# C. ObservabilityConfig unpacking
# ---------------------------------------------------------------------------


def test_observability_config_fills_in_identity_and_session():
    """session, name, description, verbose all flow through the config."""
    s = Session()
    cfg = ObservabilityConfig(
        session=s,
        name="research",
        description="finds papers",
        verbose=False,
    )
    a = Agent("claude-opus-4-7", observability=cfg)
    assert a.session is s
    assert a.name == "research"
    assert a.description == "finds papers"
    assert a._verbose is False


def test_observability_verbose_creates_implicit_session():
    """Same implicit-session behaviour as the flat verbose= kwarg."""
    cfg = ObservabilityConfig(verbose=True)
    a = Agent("claude-opus-4-7", observability=cfg)
    assert a._verbose is True
    # verbose=True with no session → implicit private Session with console=True
    assert a.session is not None


# ---------------------------------------------------------------------------
# D. Precedence — flat > config
# ---------------------------------------------------------------------------


def test_flat_timeout_overrides_config_timeout():
    """Explicit flat kwarg always wins over the config value."""
    cfg = ResilienceConfig(timeout=120.0, max_retries=7)
    a = Agent("claude-opus-4-7", resilience=cfg, timeout=30.0)
    # timeout: flat wins; max_retries: config fills in (not overridden)
    assert a.timeout == 30.0
    assert a.engine.max_retries == 7


def test_flat_name_overrides_observability_name():
    cfg = ObservabilityConfig(name="from-cfg")
    a = Agent("claude-opus-4-7", observability=cfg, name="from-flat")
    assert a.name == "from-flat"


def test_flat_fallback_overrides_config_fallback():
    backup_a = Agent("gpt-4o", name="cfg-backup")
    backup_b = Agent("gpt-4o", name="flat-backup")
    cfg = ResilienceConfig(fallback=backup_a)
    a = Agent("claude-opus-4-7", resilience=cfg, fallback=backup_b)
    assert a.fallback is backup_b


# ---------------------------------------------------------------------------
# E. AgentRuntimeConfig composite
# ---------------------------------------------------------------------------


def test_runtime_config_wraps_both():
    """AgentRuntimeConfig(resilience=…, observability=…) fills in both."""
    rt = AgentRuntimeConfig(
        resilience=ResilienceConfig(timeout=90.0, max_retries=4),
        observability=ObservabilityConfig(name="fleet-agent"),
    )
    a = Agent("claude-opus-4-7", runtime=rt)
    assert a.timeout == 90.0
    assert a.engine.max_retries == 4
    assert a.name == "fleet-agent"


def test_explicit_resilience_overrides_runtime_resilience():
    """``resilience=`` takes priority over ``runtime.resilience``."""
    rt = AgentRuntimeConfig(resilience=ResilienceConfig(timeout=90.0))
    override = ResilienceConfig(timeout=5.0)
    a = Agent("claude-opus-4-7", runtime=rt, resilience=override)
    assert a.timeout == 5.0


def test_runtime_with_only_observability():
    """Half-populated AgentRuntimeConfig still works."""
    rt = AgentRuntimeConfig(observability=ObservabilityConfig(name="o-only"))
    a = Agent("claude-opus-4-7", runtime=rt)
    assert a.name == "o-only"
    # Resilience defaults unchanged
    assert a.timeout is None
    assert a.engine.max_retries == 3


# ---------------------------------------------------------------------------
# F. Sharing a config across multiple agents
# ---------------------------------------------------------------------------


def test_single_config_shared_across_fleet():
    """One ResilienceConfig shared by three agents — the motivating use case."""
    policy = ResilienceConfig(timeout=45.0, max_retries=6, max_output_retries=4)
    researcher = Agent("claude-opus-4-7", resilience=policy, name="researcher")
    writer = Agent("claude-opus-4-7", resilience=policy, name="writer")
    reviewer = Agent("claude-opus-4-7", resilience=policy, name="reviewer")
    for a in (researcher, writer, reviewer):
        assert a.timeout == 45.0
        assert a.engine.max_retries == 6
        assert a.max_output_retries == 4


def test_single_session_shared_via_observability_config():
    """One Session reaches three agents via a shared ObservabilityConfig."""
    s = Session()
    obs = ObservabilityConfig(session=s)
    a1 = Agent("claude-opus-4-7", observability=obs, name="a1")
    a2 = Agent("claude-opus-4-7", observability=obs, name="a2")
    assert a1.session is s
    assert a2.session is s


# ---------------------------------------------------------------------------
# G. Explicitly passing the previous default still counts as "flat wins"
# ---------------------------------------------------------------------------


def test_explicit_default_value_still_overrides_config():
    """max_retries=3 passed explicitly overrides a ResilienceConfig value of 7.

    This is the reason the implementation uses a sentinel default instead of
    comparing to the documented default value — the sentinel approach
    preserves the invariant that "anything the user typed, wins".
    """
    cfg = ResilienceConfig(max_retries=7)
    # 3 happens to equal the documented default; but passing it explicitly
    # should still override the config (user's intent is explicit).
    a = Agent("claude-opus-4-7", resilience=cfg, max_retries=3)
    assert a.engine.max_retries == 3
