"""Sentinel × scenario matrix — locks every documented compile-time
and runtime branch for the seven sentinels in one place.

Sentinels under test:
  * ``from_prev`` — singleton; default ``task=`` value
  * ``from_start`` — singleton; original input envelope
  * ``from_step("name")`` — read a named earlier step's output
  * ``from_parallel("name")`` — same as ``from_step`` semantically
    (alias kept for readability inside parallel bands)
  * ``from_parallel_all("name")`` — aggregate every contiguous
    parallel sibling starting at ``"name"``
  * ``from_memory("agent_name")`` — pull live agent memory at
    execution time
  * ``from_agent("agent_name")`` — read agent's last persisted output
    from a shared Store

Scenario axes:
  * ``unknown`` — sentinel references a step / tool that doesn't exist
  * ``forward_ref`` — sentinel points at a step declared LATER in the Plan
  * ``opaque_anon`` — referent is an auto-named ``_anon_<id>`` step
  * ``same_band`` — referent is a sibling inside the same parallel band
    (T5 in the audit)
  * ``existence_check`` — ``from_agent`` / ``from_memory`` validate that
    the agent has the required attribute (I6 in the audit)

Every cell of the matrix that the 0.7.9 compiler is supposed to reject
gets one assertion here.  Cells the compiler accepts are also exercised
so a regression that flips the polarity is caught immediately.
"""

from __future__ import annotations

import pytest

from lazybridge import (
    Agent,
    Plan,
    PlanCompileError,
    Step,
    from_agent,
    from_memory,
    from_parallel,
    from_parallel_all,
    from_prev,
    from_start,
    from_step,
)
from lazybridge.testing import MockAgent

# ---------------------------------------------------------------------------
# from_step / from_parallel — unknown step
# ---------------------------------------------------------------------------


def test_from_step_unknown_step_in_task_raises_compile_error():
    research = MockAgent(["r"], name="research")
    write = MockAgent(["w"], name="write")
    with pytest.raises(PlanCompileError, match=r"from_step.*reasearch"):
        Agent(
            engine=Plan(
                Step("research"),
                Step("write", task=from_step("reasearch")),  # typo
            ),
            tools=[research, write],
            name="typo_in_task",
        )


def test_from_step_unknown_step_in_context_raises_compile_error():
    research = MockAgent(["r"], name="research")
    write = MockAgent(["w"], name="write")
    with pytest.raises(PlanCompileError, match=r"from_step.*nonexistent"):
        Agent(
            engine=Plan(
                Step("research"),
                Step("write", context=from_step("nonexistent")),
            ),
            tools=[research, write],
            name="typo_in_context",
        )


def test_from_parallel_unknown_step_raises_compile_error():
    """``from_parallel`` is an alias of ``from_step`` for compile-time
    forward-ref checks."""
    a = MockAgent(["a"], name="a")
    b = MockAgent(["b"], name="b")
    with pytest.raises(PlanCompileError, match=r"from_step.*missing"):
        Agent(
            engine=Plan(
                Step("a"),
                Step("b", context=from_parallel("missing")),
            ),
            tools=[a, b],
            name="parallel_unknown",
        )


# ---------------------------------------------------------------------------
# from_step — forward reference (referent declared later in the plan)
# ---------------------------------------------------------------------------


def test_from_step_forward_reference_in_task_raises():
    a = MockAgent(["a"], name="a")
    b = MockAgent(["b"], name="b")
    with pytest.raises(PlanCompileError):
        Agent(
            engine=Plan(
                Step("a", context=from_step("b")),  # 'b' declared AFTER
                Step("b"),
            ),
            tools=[a, b],
            name="forward_ref",
        )


# ---------------------------------------------------------------------------
# from_step — referencing an auto-named (_anon_<id>) step
# ---------------------------------------------------------------------------


def test_from_step_against_opaque_anonymous_step_raises():
    """When a Step's target has no usable name source, ``Step.__post_init__``
    falls back to ``_anon_<id>``.  Sentinels that try to reference that
    name are rejected — the LLM can't meaningfully produce ``_anon_4f0e``."""

    class _Opaque:
        """Bare object — no __name__, no .name; forces the opaque fallback."""

    anon_step = Step(target=_Opaque())
    assert anon_step.name is not None and anon_step.name.startswith("_anon_")

    downstream = MockAgent(["d"], name="downstream")
    with pytest.raises(PlanCompileError, match="auto-named"):
        Agent(
            engine=Plan(
                anon_step,
                Step("downstream", context=from_step(anon_step.name)),
            ),
            tools=[downstream],
            name="anon_ref",
        )


# ---------------------------------------------------------------------------
# from_step — same-band reference inside a parallel band (T5)
# ---------------------------------------------------------------------------


def test_from_step_to_sibling_inside_same_parallel_band_raises():
    """T5 — parallel branches start from a pre-band snapshot of history,
    so a branch cannot read from another branch in the same band.  The
    compiler must catch this rather than letting the runtime fall back
    to the start envelope silently."""
    a = MockAgent(["a"], name="a")
    b = MockAgent(["b"], name="b")
    with pytest.raises(PlanCompileError, match=r"parallel band|same parallel"):
        Agent(
            engine=Plan(
                Step("a", parallel=True),
                Step("b", parallel=True, context=from_step("a")),
            ),
            tools=[a, b],
            name="same_band_ref",
        )


def test_from_step_to_sibling_after_band_is_accepted():
    """A step AFTER a parallel band can read individual band members via
    ``from_step`` — only intra-band sibling references are rejected."""
    a = MockAgent(["a"], name="a")
    b = MockAgent(["b"], name="b")
    after = MockAgent(["after"], name="after")

    # Construction succeeds — the after step is outside the band.
    Agent(
        engine=Plan(
            Step("a", parallel=True),
            Step("b", parallel=True),
            Step("after", context=from_step("a")),
        ),
        tools=[a, b, after],
        name="post_band_ref",
    )


# ---------------------------------------------------------------------------
# from_parallel_all — band-start invariants
# ---------------------------------------------------------------------------


def test_from_parallel_all_to_non_parallel_step_raises():
    """``from_parallel_all`` aggregates a contiguous parallel band; the
    target must be a ``parallel=True`` step (the first of its band)."""
    a = MockAgent(["a"], name="a")
    b = MockAgent(["b"], name="b")
    after = MockAgent(["after"], name="after")
    with pytest.raises(PlanCompileError, match="parallel"):
        Agent(
            engine=Plan(
                Step("a"),  # NOT parallel
                Step("b"),
                Step("after", context=from_parallel_all("a")),
            ),
            tools=[a, b, after],
            name="invalid_band_start",
        )


def test_from_parallel_all_to_unknown_step_raises():
    after = MockAgent(["after"], name="after")
    with pytest.raises(PlanCompileError, match="unknown"):
        Agent(
            engine=Plan(
                Step("after", context=from_parallel_all("nonexistent")),
            ),
            tools=[after],
            name="bandstart_unknown",
        )


# ---------------------------------------------------------------------------
# from_agent — existence check (I6: hasattr, not truthiness)
# ---------------------------------------------------------------------------


def test_from_agent_requires_tool_in_map():
    write = MockAgent(["w"], name="write")
    with pytest.raises(PlanCompileError, match=r"from_agent.*missing"):
        Agent(
            engine=Plan(
                Step("write", context=from_agent("missing")),
            ),
            tools=[write],
            name="from_agent_missing",
        )


def test_from_agent_requires_store_on_source_agent():
    """The source agent must have ``store=`` attached so the output can
    be persisted for the cross-step read."""
    research = MockAgent(["r"], name="research")
    research.agent_store = None  # explicitly no store
    research.returns_envelope = True
    write = MockAgent(["w"], name="write")
    with pytest.raises(PlanCompileError, match="store="):
        Agent(
            engine=Plan(
                Step("write", context=from_agent("research")),
            ),
            tools=[research, write],
            name="from_agent_no_store",
        )


# ---------------------------------------------------------------------------
# from_memory — existence check uses hasattr (I6)
# ---------------------------------------------------------------------------


def test_from_memory_requires_tool_in_map():
    write = MockAgent(["w"], name="write")
    with pytest.raises(PlanCompileError, match=r"from_memory.*missing"):
        Agent(
            engine=Plan(
                Step("write", context=from_memory("missing")),
            ),
            tools=[write],
            name="from_memory_missing",
        )


def test_from_memory_empty_memory_acceptance_lives_in_canonical_grammar_tests():
    """The I6 "empty Memory is legitimate" contract is exercised end-to-end
    in ``test_canonical_grammar.py`` (the agent-with-memory pipeline tests).
    Replicating it here would duplicate the as_tool() wrapping fixture —
    we just document the cross-reference so anyone reading the matrix
    knows where the positive case lives."""
    # No-op assertion to keep this slot visible in the matrix listing.
    assert True


# ---------------------------------------------------------------------------
# Singleton sentinels — from_prev / from_start always compile
# ---------------------------------------------------------------------------


def test_from_prev_singleton_compiles_in_any_position():
    a = MockAgent(["a"], name="a")
    b = MockAgent(["b"], name="b")

    Agent(
        engine=Plan(
            Step("a"),
            Step("b", task=from_prev),
            Step("b", task=from_prev, context=from_prev),  # duplicate fails compiler — see below
        ),
        tools=[a, b],
        name="from_prev_compiles",
    ) if False else None  # this would fail on duplicate step names

    # Same shape without duplicates compiles fine.
    Agent(
        engine=Plan(
            Step("a"),
            Step("b", task=from_prev),
        ),
        tools=[a, b],
        name="from_prev_ok",
    )


def test_from_start_singleton_compiles_in_context():
    a = MockAgent(["a"], name="a")
    b = MockAgent(["b"], name="b")
    Agent(
        engine=Plan(
            Step("a"),
            Step("b", context=from_start),
        ),
        tools=[a, b],
        name="from_start_ok",
    )


# ---------------------------------------------------------------------------
# Nested plan — from_step in an inner Plan resolves against its own history
# ---------------------------------------------------------------------------


def test_nested_plan_sentinels_resolve_in_inner_scope():
    """A Plan used as a tool target inside another Plan resolves its
    own sentinels against ITS history, not the outer plan's.  Compile-
    time validation must pass when the inner sentinel points at an
    inner step (even if no outer step shares the name)."""
    inner_a = MockAgent(["a"], name="inner_a")
    inner_b = MockAgent(["b"], name="inner_b")
    inner_pipeline = Agent(
        engine=Plan(
            Step("inner_a"),
            Step("inner_b", context=from_step("inner_a")),
        ),
        tools=[inner_a, inner_b],
        name="inner",
    )

    outer_first = MockAgent(["o"], name="outer_first")
    # The outer plan composes the inner pipeline as a step target.
    Agent(
        engine=Plan(
            Step("outer_first"),
            Step(target=inner_pipeline, name="inner"),
        ),
        tools=[outer_first],
        name="outer",
    )
