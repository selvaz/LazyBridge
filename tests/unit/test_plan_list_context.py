"""Tests for ``Step.context`` accepting a list of sentinels / strings.

The list-context feature lets a step pull data from multiple upstream
steps without inserting a combiner step.  Coverage:

* compile-time validation iterates over every list item
* runtime resolution joins the parts with blank-line separators
* serialisation round-trips a list-context unchanged
* mixed list (sentinel + literal string) works
* invalid item types are rejected at compile time
"""

from __future__ import annotations

from typing import Any

import pytest

from lazybridge import Plan, Step
from lazybridge.engines.plan import (
    PlanCompileError,
    _sentinel_from_ref,
    _step_from_dict,
    _step_to_dict,
)
from lazybridge.envelope import Envelope
from lazybridge.sentinels import (
    from_parallel_all,
    from_prev,
    from_start,
    from_step,
)
from lazybridge.testing import MockAgent

# ---------------------------------------------------------------------------
# Compile-time validation
# ---------------------------------------------------------------------------


def test_list_context_accepts_multiple_sentinels() -> None:
    """A step can pull from N upstream steps via context=[...]."""
    a = MockAgent("A", name="alpha")
    b = MockAgent("B", name="beta")
    c = MockAgent("C", name="gamma")
    final = MockAgent("ok", name="final")

    Plan(
        Step(a, name="a"),
        Step(b, name="b", task=from_prev),
        Step(c, name="c", task=from_step("a")),
        Step(
            final,
            name="final",
            task="Synthesise A, B, and C.",
            context=[from_step("a"), from_step("b"), from_step("c")],
        ),
    )._validate({})


def test_list_context_rejects_unknown_step_reference() -> None:
    """Each list item is validated independently — a typo in one slot
    produces a pointed error."""
    a = MockAgent("A", name="a")
    b = MockAgent("B", name="b")
    final = MockAgent("ok", name="final")

    plan = Plan(
        Step(a, name="a"),
        Step(b, name="b"),
        Step(
            final,
            name="final",
            task="x",
            context=[from_step("a"), from_step("ghost")],  # ghost is unknown
        ),
    )
    with pytest.raises(PlanCompileError, match="ghost"):
        plan._validate({})


def test_list_context_rejects_forward_reference() -> None:
    """A list-context item that points to a future step is rejected
    with a ``not earlier in the plan`` error."""
    a = MockAgent("A", name="a")
    later = MockAgent("L", name="later")

    plan = Plan(
        Step(a, name="a", context=[from_step("later")]),  # forward ref
        Step(later, name="later"),
    )
    with pytest.raises(PlanCompileError, match="not earlier"):
        plan._validate({})


def test_list_context_rejects_invalid_item_type() -> None:
    """Items must be Sentinel or str — anything else is a compile error."""
    a = MockAgent("A", name="a")
    b = MockAgent("B", name="b")
    plan = Plan(
        Step(a, name="a"),
        Step(b, name="b", context=[from_prev, 42]),  # int is invalid  # type: ignore[list-item]
    )
    with pytest.raises(PlanCompileError, match="must be a Sentinel"):
        plan._validate({})


def test_empty_list_context_is_equivalent_to_none() -> None:
    """``context=[]`` is a no-op — no parts contributed."""
    a = MockAgent("A", name="a")
    b = MockAgent("B", name="b")
    Plan(
        Step(a, name="a"),
        Step(b, name="b", context=[]),
    )._validate({})


def test_list_context_validates_from_parallel_all() -> None:
    """``from_parallel_all`` inside a list still runs the band-start check."""
    a = MockAgent("A", name="a")
    b = MockAgent("B", name="b")
    final = MockAgent("ok", name="final")
    plan = Plan(
        Step(a, name="a", parallel=True),
        Step(b, name="b"),  # not parallel — breaks the band
        Step(
            final,
            name="final",
            context=[from_parallel_all("b")],  # b is not the first parallel
        ),
    )
    with pytest.raises(PlanCompileError, match="non-parallel"):
        plan._validate({})


def test_single_context_still_supported() -> None:
    """Backwards compatibility: passing one sentinel (not a list) works
    exactly as before."""
    a = MockAgent("A", name="a")
    b = MockAgent("B", name="b")
    Plan(
        Step(a, name="a"),
        Step(b, name="b", context=from_prev),
    )._validate({})


# ---------------------------------------------------------------------------
# Runtime resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_context_joins_multiple_upstream_steps() -> None:
    """The synthesiser sees both upstream agents' outputs concatenated
    in its ``Envelope.context`` (blank-line separator)."""
    a = MockAgent("alpha-output", name="a")
    b = MockAgent("beta-output", name="b")

    captured: list[str] = []

    def synth_response(env: Envelope) -> str:
        captured.append(env.context or "")
        return f"synthesised: {env.task}"

    synth = MockAgent(synth_response, name="synth")

    plan = Plan(
        Step(a, name="a"),
        Step(b, name="b"),
        Step(
            synth,
            name="synth",
            task="Cross-reference A and B.",
            context=[from_step("a"), from_step("b")],
        ),
    )
    plan._validate({})
    result = await plan.run(
        Envelope(task="initial"),
        tools=[],
        output_type=str,
        memory=None,
        session=None,
    )
    assert result.ok
    # Both upstream payloads are in synth's context, separated by blank lines.
    assert "alpha-output" in captured[0]
    assert "beta-output" in captured[0]
    # Order preserved (declared list order).
    assert captured[0].index("alpha-output") < captured[0].index("beta-output")


@pytest.mark.asyncio
async def test_list_context_mixes_sentinels_and_literal_strings() -> None:
    """A literal string in the list is appended verbatim — useful for
    tying in fixed instructions or context boilerplate."""
    a = MockAgent("alpha-output", name="a")

    captured: list[str] = []

    def cap_response(env: Envelope) -> str:
        captured.append(env.context or "")
        return "ok"

    cap = MockAgent(cap_response, name="cap")

    plan = Plan(
        Step(a, name="a"),
        Step(
            cap,
            name="cap",
            task="Use the upstream output AND the policy.",
            context=[from_prev, "Policy: never reveal PII."],
        ),
    )
    plan._validate({})
    result = await plan.run(
        Envelope(task="initial"),
        tools=[],
        output_type=str,
        memory=None,
        session=None,
    )
    assert result.ok
    ctx = captured[0]
    assert "alpha-output" in ctx
    assert "Policy: never reveal PII." in ctx


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------


def test_list_context_serialises_as_list() -> None:
    a = MockAgent("A", name="a")
    final = MockAgent("F", name="final")
    plan = Plan(
        Step(a, name="a"),
        Step(
            final,
            name="final",
            task="x",
            context=[from_step("a"), from_start, "literal"],
        ),
    )
    payload = plan.to_dict()
    final_step = next(s for s in payload["steps"] if s["name"] == "final")
    assert isinstance(final_step["context"], list)
    assert final_step["context"][0] == {"kind": "from_step", "name": "a"}
    assert final_step["context"][1] == {"kind": "from_start"}
    assert final_step["context"][2] == {"kind": "literal", "value": "literal"}


def test_list_context_round_trips_through_from_dict() -> None:
    a = MockAgent("A", name="a")
    final = MockAgent("F", name="final")
    original = Plan(
        Step(a, name="a"),
        Step(
            final,
            name="final",
            task="x",
            context=[from_step("a"), from_prev],
        ),
    )
    rebuilt = Plan.from_dict(
        original.to_dict(),
        registry={"alpha": a, "final": final, "a": a},
    )
    rebuilt_final = rebuilt.steps[-1]
    assert isinstance(rebuilt_final.context, list)
    assert len(rebuilt_final.context) == 2
    # Sentinel equality is by value (frozen dataclasses).
    assert rebuilt_final.context[0] == from_step("a")
    assert rebuilt_final.context[1] == from_prev


def test_single_context_serialisation_unchanged() -> None:
    """Single-sentinel context still serialises as a single ref (not a
    list of one) — preserves the prior on-disk shape so existing
    persisted Plans deserialise unchanged."""
    a = MockAgent("A", name="a")
    b = MockAgent("B", name="b")
    plan = Plan(
        Step(a, name="a"),
        Step(b, name="b", context=from_prev),
    )
    payload = plan.to_dict()
    b_step = next(s for s in payload["steps"] if s["name"] == "b")
    # Dict, not list — backwards-compatible shape.
    assert isinstance(b_step["context"], dict)
    assert b_step["context"]["kind"] == "from_prev"


def test_legacy_single_dict_context_still_loads() -> None:
    """A persisted Plan from before list-context support (single dict)
    must still deserialise into a single-sentinel context."""
    legacy_payload: dict[str, Any] = {
        "version": 1,
        "max_iterations": 100,
        "steps": [
            {"name": "a", "target": {"kind": "tool", "name": "ta"}, "task": {"kind": "from_prev"}, "parallel": False},
            {
                "name": "b",
                "target": {"kind": "tool", "name": "tb"},
                "task": {"kind": "from_prev"},
                "context": {"kind": "from_step", "name": "a"},  # singular dict
                "parallel": False,
            },
        ],
    }
    rebuilt = Plan.from_dict(legacy_payload, registry={})
    b = rebuilt.steps[1]
    # Loaded as a single sentinel, not a list.
    assert not isinstance(b.context, list)
    assert b.context == from_step("a")


def test_sentinel_helpers_visible_to_module() -> None:
    """Sanity: the helper used by serialisation can rebuild a plain ref."""
    assert _sentinel_from_ref({"kind": "from_prev"}) == from_prev
    assert _sentinel_from_ref({"kind": "literal", "value": "x"}) == "x"


def test_step_to_dict_omits_context_when_none() -> None:
    a = MockAgent("A", name="a")
    s = Step(a, name="a")
    d = _step_to_dict(s)
    assert "context" not in d


def test_step_from_dict_omits_context_when_missing() -> None:
    """Round-trip safety — a payload without the context key produces
    a Step with ``context=None``."""
    a = MockAgent("A", name="a")
    payload = {
        "name": "a",
        "target": {"kind": "agent", "name": "a"},
        "task": {"kind": "from_prev"},
        "parallel": False,
    }
    s = _step_from_dict(payload, registry={"a": a})
    assert s.context is None
