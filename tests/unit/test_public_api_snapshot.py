"""Public-API snapshot — pins the exact set of names exported by
``lazybridge`` so:

1. Adding a new public symbol requires an explicit update here (forces
   a SKILL.md / docs review at the same time).
2. Deleted-in-0.7.9 names cannot silently come back.
3. ``__all__`` and the actual module namespace stay in sync (every
   listed name resolves to a real attribute).

Run after every public-surface change.  When the snapshot intentionally
shifts, update :data:`_EXPECTED` below alongside the SKILL.md / docs
that teach the new shape.
"""

from __future__ import annotations

import lazybridge

# The authoritative post-0.7.9 public surface.  Sorted for diff-friendly
# updates; keep it in lockstep with ``lazybridge/__init__.py::__all__``.
_EXPECTED: frozenset[str] = frozenset(
    {
        # Primary API
        "Agent",
        "ParallelAgent",
        # Envelope
        "Envelope",
        # Sentinels
        "from_prev",
        "from_start",
        "from_step",
        "from_parallel",
        "from_parallel_all",
        "from_memory",
        "from_agent",
        # Tools
        "tool",
        "Tool",
        "ToolProvider",
        # Native tools (provider-hosted)
        "NativeTool",
        # Predicates DSL (for Step.routes)
        "when",
        # State
        "Memory",
        "Store",
        # Session / Observability
        "Session",
        "EventLog",
        "EventType",
        # Guardrails
        "Guard",
        "GuardAction",
        "GuardError",
        "ContentGuard",
        "GuardChain",
        "LLMGuard",
        # Engines (HumanEngine, SupervisorEngine in lazybridge.ext.hil)
        "LLMEngine",
        "PROVIDER_ALIASES",
        "Plan",
        "Step",
        "ConcurrentPlanRunError",
        "PlanCompileError",
        "PlanPaused",
        "PlanRuntimeError",
        "ToolTimeoutError",
        "StreamStallError",
        # Graph
        "GraphSchema",
        # Exporters
        "EventExporter",
        "CallbackExporter",
        "ConsoleExporter",
        "FilteredExporter",
        "JsonFileExporter",
        "StructuredLogExporter",
        # Provider entry point (custom adapters)
        "BaseProvider",
        "Tier",
        "UnsupportedFeatureError",
        "UnsupportedNativeToolError",
        # Multimodal content blocks
        "ImageContent",
        "AudioContent",
        # Cache configuration (kept — internal repr for LLMEngine cache)
        "CacheConfig",
        # Testing
        "MockAgent",
    }
)


# 0.7-era names that must stay deleted.  If any of these become accessible
# again, the deletion has been undone — fail loudly.
_DELETED_IN_0_7_9: frozenset[str] = frozenset(
    {
        # Pure-alias factories on Agent (not module-level, but caught by
        # ``hasattr(Agent, name)`` below).
        "from_model",
        "from_engine",
        "from_plan",
        "from_chain",
        "from_parallel",
        # Module-level deletions.
        "AgentRuntimeConfig",
        "ResilienceConfig",
        "ObservabilityConfig",
        "_ParallelAgent",
        "wrap_tool",
    }
)


def test_all_matches_expected_snapshot():
    """``lazybridge.__all__`` matches the curated snapshot exactly.

    A diff here means somebody added or removed a public symbol — update
    :data:`_EXPECTED` deliberately and check SKILL.md / docs reflect the
    change before merging.
    """
    actual = frozenset(lazybridge.__all__)
    extra = actual - _EXPECTED
    missing = _EXPECTED - actual
    assert not extra and not missing, (
        f"Public API drift detected:\n"
        f"  Added (not in expected snapshot):    {sorted(extra)}\n"
        f"  Removed (in expected but missing):   {sorted(missing)}\n"
        f"Update tests/unit/test_public_api_snapshot.py::_EXPECTED with the\n"
        f"new shape AFTER you've updated SKILL.md / docs to match."
    )


def test_every_all_entry_is_actually_importable():
    """Every name in ``__all__`` must resolve to a real attribute on
    the package.  Catches ``__all__`` listing a symbol that the module
    forgot to import (or that an over-eager refactor deleted)."""
    missing = [name for name in lazybridge.__all__ if not hasattr(lazybridge, name)]
    assert not missing, f"__all__ lists non-importable name(s): {missing}"


def test_deleted_module_level_symbols_stay_gone():
    """0.7.9 removed three module-level config classes and the underscore
    alias of ``ParallelAgent`` and ``wrap_tool``.  Resurrecting any of
    them silently would re-introduce the LLM traps the deletions closed."""
    module_level_deletions = {
        "AgentRuntimeConfig",
        "ResilienceConfig",
        "ObservabilityConfig",
        "_ParallelAgent",
        "wrap_tool",
    }
    leaked = [name for name in module_level_deletions if hasattr(lazybridge, name)]
    assert not leaked, (
        f"Deleted-in-0.7.9 symbol(s) re-appeared in lazybridge.*: {leaked}.\n"
        f"If the resurrection is intentional, document it in SKILL.md and\n"
        f"the migration guide before un-pinning this test."
    )


def test_deleted_agent_factory_methods_stay_gone():
    """Five pure-alias ``Agent.from_*`` factories were deleted in 0.7.9.
    A change that re-adds any of them undoes Block A's simplification."""
    from lazybridge import Agent

    factory_deletions = {"from_model", "from_engine", "from_plan", "from_chain", "from_parallel"}
    leaked = [m for m in factory_deletions if hasattr(Agent, m)]
    assert not leaked, (
        f"Deleted-in-0.7.9 Agent.from_* method(s) re-appeared: {leaked}.\n"
        f"If intentional, update test_public_api_snapshot.py::_DELETED_IN_0_7_9."
    )


def test_kept_agent_factories_are_callable():
    """The three factories that stayed (``chain`` / ``parallel`` /
    ``from_provider``) must remain callable on the Agent class."""
    from lazybridge import Agent

    for name in ("chain", "parallel", "from_provider"):
        bound = getattr(Agent, name, None)
        assert callable(bound), f"Agent.{name} should be callable but isn't"


def test_provider_capability_matrix_covers_every_provider():
    """``lazybridge.matrix.provider_capabilities()`` should report a row
    for every provider ``LLMEngine`` knows how to instantiate.  Catches
    the common drift mode of registering a provider in ``LLMEngine.
    _PROVIDER_RULES`` but forgetting to add it to the matrix."""
    from lazybridge.engines.llm import LLMEngine
    from lazybridge.matrix import provider_capabilities

    rules_providers = {p for _, _, p in LLMEngine._PROVIDER_RULES}
    matrix_providers = set(provider_capabilities().keys())
    missing = rules_providers - matrix_providers
    assert not missing, (
        f"Provider(s) registered in LLMEngine._PROVIDER_RULES but missing from lazybridge.matrix: {sorted(missing)}."
    )
