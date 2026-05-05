"""Wave 3.2 — Agent runtime config precedence is centralised and pure.

Pre-W3.2 the merge of ``flat kwarg > config object > default`` was
inline 11 hand-written ``if x is _UNSET:`` blocks in
``Agent.__init__``.  Adding a new knob meant editing that block, the
docstring, and the constructor signature in three coordinated places
— easy to drift.

W3.2 extracts the merge into ``_resolve_runtime_kwargs`` and a
``_RUNTIME_KNOB_DEFAULTS`` table.  This module exercises the pure
helper directly so the precedence semantics are pinned regardless of
how ``Agent.__init__`` evolves.

Precedence ladder (highest wins):
1. Flat kwarg passed to Agent (anything that isn't ``_UNSET``).
2. Explicit ``resilience=`` / ``observability=`` config kwarg.
3. ``runtime.resilience`` / ``runtime.observability`` from a composite
   ``AgentRuntimeConfig``.
4. Documented default from ``_RUNTIME_KNOB_DEFAULTS``.
"""

from __future__ import annotations

import pytest

from lazybridge import (
    Agent,
    AgentRuntimeConfig,
    ObservabilityConfig,
    ResilienceConfig,
)
from lazybridge.agent import (
    _RUNTIME_KNOB_DEFAULTS,
    _UNSET,
    _resolve_runtime_kwargs,
)


def _all_unset() -> dict:
    """Build a flat dict where every knob is the unset sentinel."""
    return dict.fromkeys(_RUNTIME_KNOB_DEFAULTS, _UNSET)


# ---------------------------------------------------------------------------
# Layer 4 — defaults when nothing is provided
# ---------------------------------------------------------------------------


def test_all_defaults_when_no_config_no_flat():
    out = _resolve_runtime_kwargs(
        runtime=None, resilience=None, observability=None, flat=_all_unset()
    )
    assert out == {k: default for k, (default, _src) in _RUNTIME_KNOB_DEFAULTS.items()}


def test_default_table_pins_documented_values():
    """Sanity: the documented defaults from the previous inline code
    survive in the centralised table."""
    expected = {
        "timeout": (None, "resilience"),
        "max_retries": (3, "resilience"),
        "retry_delay": (1.0, "resilience"),
        "cache": (False, "resilience"),
        "max_output_retries": (2, "resilience"),
        "output_validator": (None, "resilience"),
        "fallback": (None, "resilience"),
        "verbose": (False, "observability"),
        "session": (None, "observability"),
        "name": (None, "observability"),
        "description": (None, "observability"),
    }
    assert _RUNTIME_KNOB_DEFAULTS == expected


# ---------------------------------------------------------------------------
# Layer 3 — runtime composite fills in
# ---------------------------------------------------------------------------


def test_runtime_resilience_fills_when_no_explicit():
    rt = AgentRuntimeConfig(
        resilience=ResilienceConfig(timeout=30.0, max_retries=7),
        observability=ObservabilityConfig(verbose=True),
    )
    out = _resolve_runtime_kwargs(
        runtime=rt, resilience=None, observability=None, flat=_all_unset()
    )
    assert out["timeout"] == 30.0
    assert out["max_retries"] == 7
    assert out["retry_delay"] == 1.0  # default — not in the rt config
    assert out["verbose"] is True


def test_runtime_with_no_resilience_uses_defaults_for_resilience_knobs():
    rt = AgentRuntimeConfig(observability=ObservabilityConfig(name="x"))
    out = _resolve_runtime_kwargs(
        runtime=rt, resilience=None, observability=None, flat=_all_unset()
    )
    assert out["name"] == "x"
    assert out["timeout"] is None  # default


# ---------------------------------------------------------------------------
# Layer 2 — explicit resilience= / observability= override runtime composite
# ---------------------------------------------------------------------------


def test_explicit_resilience_overrides_runtime_resilience():
    rt = AgentRuntimeConfig(resilience=ResilienceConfig(timeout=30.0, max_retries=7))
    explicit = ResilienceConfig(timeout=60.0, max_retries=2)
    out = _resolve_runtime_kwargs(
        runtime=rt, resilience=explicit, observability=None, flat=_all_unset()
    )
    assert out["timeout"] == 60.0
    assert out["max_retries"] == 2


def test_explicit_observability_overrides_runtime_observability():
    rt = AgentRuntimeConfig(observability=ObservabilityConfig(name="from-rt"))
    explicit = ObservabilityConfig(name="explicit")
    out = _resolve_runtime_kwargs(
        runtime=rt, resilience=None, observability=explicit, flat=_all_unset()
    )
    assert out["name"] == "explicit"


# ---------------------------------------------------------------------------
# Layer 1 — flat kwarg wins over EVERYTHING
# ---------------------------------------------------------------------------


def test_flat_wins_over_explicit_resilience():
    explicit = ResilienceConfig(timeout=60.0, max_retries=2)
    flat = _all_unset()
    flat["timeout"] = 120.0
    out = _resolve_runtime_kwargs(
        runtime=None, resilience=explicit, observability=None, flat=flat
    )
    assert out["timeout"] == 120.0  # flat wins
    assert out["max_retries"] == 2  # untouched, from explicit config


def test_flat_wins_even_when_value_matches_documented_default():
    """A user explicitly passing the default is NOT a sentinel —
    they wanted to lock that value in regardless of config objects."""
    explicit = ResilienceConfig(max_retries=10)
    flat = _all_unset()
    flat["max_retries"] = 3  # explicit, matches default
    out = _resolve_runtime_kwargs(
        runtime=None, resilience=explicit, observability=None, flat=flat
    )
    assert out["max_retries"] == 3


def test_flat_none_treated_as_explicit_value():
    """Flat None is an explicit user choice — distinct from sentinel.
    A user who passes ``timeout=None`` wants no timeout, even if
    ``resilience.timeout=30.0`` is set."""
    explicit = ResilienceConfig(timeout=30.0)
    flat = _all_unset()
    flat["timeout"] = None  # explicit — not _UNSET
    out = _resolve_runtime_kwargs(
        runtime=None, resilience=explicit, observability=None, flat=flat
    )
    assert out["timeout"] is None


# ---------------------------------------------------------------------------
# End-to-end via Agent.__init__ — non-breaking refactor verification
# ---------------------------------------------------------------------------


def test_agent_construction_flat_kwarg_wins():
    """End-to-end: the public surface still honours flat > config."""
    a = Agent(
        "claude-opus-4-7",
        max_retries=99,
        resilience=ResilienceConfig(max_retries=2),
    )
    assert a.engine.max_retries == 99


def test_agent_construction_resilience_object_fills_unset_flat():
    a = Agent(
        "claude-opus-4-7",
        resilience=ResilienceConfig(max_retries=7, retry_delay=2.5),
    )
    assert a.engine.max_retries == 7
    assert a.engine.retry_delay == 2.5


def test_agent_construction_runtime_composite_works():
    a = Agent(
        "claude-opus-4-7",
        runtime=AgentRuntimeConfig(
            resilience=ResilienceConfig(max_retries=4),
            observability=ObservabilityConfig(name="custom-name"),
        ),
    )
    assert a.engine.max_retries == 4
    assert a.name == "custom-name"


def test_agent_construction_observability_session_propagates():
    from lazybridge import Session

    sess = Session()
    a = Agent(
        "claude-opus-4-7",
        observability=ObservabilityConfig(session=sess),
    )
    assert a.session is sess


@pytest.mark.parametrize(
    "knob",
    list(_RUNTIME_KNOB_DEFAULTS.keys()),
)
def test_every_documented_knob_resolves_via_helper(knob):
    """Coverage guard: every key in the table is exercised by the
    resolver.  If a future patch adds a knob to the table without
    routing the flat kwarg through, the matrix above fails to update."""
    flat = _all_unset()
    out = _resolve_runtime_kwargs(
        runtime=None, resilience=None, observability=None, flat=flat
    )
    assert knob in out
