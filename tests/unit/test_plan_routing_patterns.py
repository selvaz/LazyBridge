"""Regression tests pinning the three Plan routing patterns.

The semantics — driven by ``_routing`` in ``lazybridge/engines/plan.py``
— are:

* a step routes if and only if its payload has a string ``next`` that
  matches an existing step name;
* once any step has routed, subsequent linear fall-through is
  disabled — a routed-to step that has no ``next`` ends the Plan.

The patterns documented in ``fragments/plan.md`` rely on those rules.
This test file pins them so a future engine refactor cannot silently
break documented examples.
"""

from __future__ import annotations

import asyncio
from typing import Literal

import pytest
from pydantic import BaseModel

from lazybridge import Plan, Step
from lazybridge.envelope import Envelope
from lazybridge.testing import MockAgent


def _run(plan: Plan) -> Envelope:
    return asyncio.run(
        plan.run(
            Envelope(task="t"),
            tools=[],
            output_type=str,
            memory=None,
            session=None,
        )
    )


# ---------------------------------------------------------------------------
# Pattern A — linear-only
# ---------------------------------------------------------------------------


def test_pattern_a_linear_runs_every_step_in_declared_order() -> None:
    a = MockAgent("A", name="a")
    b = MockAgent("B", name="b")
    c = MockAgent("C", name="c")
    plan = Plan(
        Step(a, name="a"),
        Step(b, name="b"),
        Step(c, name="c"),
    )
    plan._validate({})
    _run(plan)
    assert len(a.calls) == 1
    assert len(b.calls) == 1
    assert len(c.calls) == 1


# ---------------------------------------------------------------------------
# Pattern B — terminal-fork via ``Optional[Literal[...]]`` default ``None``
# ---------------------------------------------------------------------------


class _Hits(BaseModel):
    items: list[str]
    # ``None`` → linear fall-through; ``"empty"`` → route to terminal step.
    next: Literal["empty"] | None = None


def test_pattern_b_happy_path_falls_through_linearly() -> None:
    """``Hits.next=None`` → search → rank → write → fall through to empty.

    NOTE: with this layout (apology last in declared order), the happy
    path linearly progresses into ``apology`` after ``write``.  In real
    pipelines you'd give ``write`` a ``next`` field, or rely on
    ``write`` being the last declared step.  This test pins what
    actually happens today so docs and code stay aligned.
    """
    searcher = MockAgent(_Hits(items=["a"], next=None), name="search", output=_Hits)
    ranker = MockAgent("ranked", name="rank")
    writer = MockAgent("written", name="write")
    apology = MockAgent("sorry", name="empty")
    plan = Plan(
        Step(searcher, name="search", output=_Hits),
        Step(ranker, name="rank"),
        Step(writer, name="write"),
        Step(apology, name="empty"),
    )
    plan._validate({})
    _run(plan)
    assert len(searcher.calls) == 1
    assert len(ranker.calls) == 1
    assert len(writer.calls) == 1
    # Linear fall-through after ``write`` reaches ``empty``; documented
    # in Pattern B's caveat.
    assert len(apology.calls) == 1


def test_pattern_b_early_out_routes_to_terminal_and_skips_intermediate() -> None:
    """``Hits.next='empty'`` → search jumps straight to apology; rank
    and write are never reached."""
    searcher = MockAgent(_Hits(items=[], next="empty"), name="search", output=_Hits)
    ranker = MockAgent("ranked", name="rank")
    writer = MockAgent("written", name="write")
    apology = MockAgent("sorry", name="empty")
    plan = Plan(
        Step(searcher, name="search", output=_Hits),
        Step(ranker, name="rank"),
        Step(writer, name="write"),
        Step(apology, name="empty"),
    )
    plan._validate({})
    _run(plan)
    assert len(searcher.calls) == 1
    assert len(ranker.calls) == 0
    assert len(writer.calls) == 0
    assert len(apology.calls) == 1


# ---------------------------------------------------------------------------
# Pattern C — always-routing (every output model declares ``next``)
# ---------------------------------------------------------------------------


class _HitsAR(BaseModel):
    items: list[str]
    next: Literal["rank", "empty"] = "rank"


class _RankedAR(BaseModel):
    top: list[str]
    next: Literal["write"] = "write"  # explicitly chains to "write"


class _WriteResult(BaseModel):
    text: str
    # No ``next`` field: after this step (was_routed=True) the Plan ends.


def test_pattern_c_always_routing_chains_through_to_terminal() -> None:
    """Each routing model points at the next step; the final step has
    no ``next`` and the Plan ends there with ``apology`` never reached."""
    searcher = MockAgent(_HitsAR(items=["a"], next="rank"), name="search", output=_HitsAR)
    ranker = MockAgent(_RankedAR(top=["a"], next="write"), name="rank", output=_RankedAR)
    writer = MockAgent(_WriteResult(text="draft"), name="write", output=_WriteResult)
    apology = MockAgent("sorry", name="empty")
    plan = Plan(
        Step(searcher, name="search", output=_HitsAR),
        Step(ranker, name="rank", output=_RankedAR),
        Step(writer, name="write", output=_WriteResult),
        Step(apology, name="empty"),
    )
    plan._validate({})
    _run(plan)
    assert len(searcher.calls) == 1
    assert len(ranker.calls) == 1
    assert len(writer.calls) == 1
    # ``apology`` was reachable only via routing; once write was the
    # routed-to terminal, the Plan ended without falling through.
    assert len(apology.calls) == 0


# ---------------------------------------------------------------------------
# The trap — half-routed pipeline
# ---------------------------------------------------------------------------


class _HitsTrap(BaseModel):
    items: list[str]
    next: Literal["rank", "empty"] = "rank"  # routes


class _RankedTrap(BaseModel):
    top: list[str]
    # NO `next` field — but reached via routing → Plan ends here.


def test_half_routed_pipeline_terminates_after_first_routed_step() -> None:
    """Pin the documented gotcha: a model with ``next`` that routes to
    a step whose model has no ``next`` ends the Plan there.  This is
    the failure mode the docs explicitly warn against."""
    searcher = MockAgent(_HitsTrap(items=["a"], next="rank"), name="search", output=_HitsTrap)
    ranker = MockAgent(_RankedTrap(top=["a"]), name="rank", output=_RankedTrap)
    writer = MockAgent("written", name="write")
    apology = MockAgent("sorry", name="empty")
    plan = Plan(
        Step(searcher, name="search", output=_HitsTrap),
        Step(ranker, name="rank", output=_RankedTrap),
        Step(writer, name="write"),
        Step(apology, name="empty"),
    )
    plan._validate({})
    _run(plan)
    assert len(searcher.calls) == 1
    assert len(ranker.calls) == 1
    # Plan ends after rank — write and apology never reached.  This is
    # what the "no fall-through after a route" rule says.
    assert len(writer.calls) == 0
    assert len(apology.calls) == 0


# ---------------------------------------------------------------------------
# Compile-time validation of routing literals
# ---------------------------------------------------------------------------


def test_invalid_next_literal_value_rejected_at_construction() -> None:
    """``next: Literal["nonexistent"]`` is caught by PlanCompiler before
    any LLM call."""
    from lazybridge.engines.plan import PlanCompileError

    class _Bad(BaseModel):
        next: Literal["nonexistent"]

    a = MockAgent(_Bad(next="nonexistent"), name="a", output=_Bad)
    b = MockAgent("B", name="b")
    plan = Plan(
        Step(a, name="a", output=_Bad),
        Step(b, name="b"),
    )
    with pytest.raises(PlanCompileError, match="nonexistent"):
        plan._validate({})
