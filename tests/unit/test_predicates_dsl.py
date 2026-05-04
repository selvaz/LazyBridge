"""Tests for the ``when`` predicate DSL.

These verify each verb on its own and that ``when.field(...)``
predicates plug into ``Step(routes={...})`` and behave identically to
the equivalent lambda.
"""

from __future__ import annotations

from pydantic import BaseModel

from lazybridge import Plan, Step, when
from lazybridge.envelope import Envelope
from lazybridge.testing import MockAgent

# ---------------------------------------------------------------------------
# Verb-by-verb unit checks
# ---------------------------------------------------------------------------


class _Hits(BaseModel):
    items: list[str] = []


class _Triage(BaseModel):
    severity: str = "normal"
    score: float = 0.0
    title: str = ""
    approved: bool = False


def _env(payload: object) -> Envelope:
    """Build a payload-only envelope for predicate testing."""
    return Envelope(task="t", payload=payload)


def test_when_field_empty_truthy_when_list_empty() -> None:
    pred = when.field("items").empty()
    assert pred(_env(_Hits(items=[]))) is True
    assert pred(_env(_Hits(items=["a"]))) is False


def test_when_field_not_empty_inverts_empty() -> None:
    pred = when.field("items").not_empty()
    assert pred(_env(_Hits(items=[]))) is False
    assert pred(_env(_Hits(items=["a"]))) is True


def test_when_field_equals() -> None:
    pred = when.field("severity").equals("urgent")
    assert pred(_env(_Triage(severity="urgent"))) is True
    assert pred(_env(_Triage(severity="normal"))) is False


def test_when_field_not_equals() -> None:
    pred = when.field("severity").not_equals("urgent")
    assert pred(_env(_Triage(severity="normal"))) is True
    assert pred(_env(_Triage(severity="urgent"))) is False


def test_when_field_is_handles_boolean_and_none() -> None:
    pred = when.field("approved").is_(False)
    assert pred(_env(_Triage(approved=False))) is True
    assert pred(_env(_Triage(approved=True))) is False


def test_when_field_in_membership() -> None:
    pred = when.field("severity").in_({"urgent", "p0"})
    assert pred(_env(_Triage(severity="urgent"))) is True
    assert pred(_env(_Triage(severity="p0"))) is True
    assert pred(_env(_Triage(severity="normal"))) is False


def test_when_field_not_in_inverts_membership() -> None:
    pred = when.field("severity").not_in_({"normal"})
    assert pred(_env(_Triage(severity="urgent"))) is True
    assert pred(_env(_Triage(severity="normal"))) is False


def test_when_field_in_with_unhashable_values_falls_back_to_tuple() -> None:
    """Lists / dicts as members are unhashable; the DSL must still
    accept them via the tuple fallback."""

    class _Custom(BaseModel):
        tags: list[str] = []

    pred = when.field("tags").in_([["a"], ["b"]])
    assert pred(_env(_Custom(tags=["a"]))) is True
    assert pred(_env(_Custom(tags=["c"]))) is False


def test_when_field_greater_than() -> None:
    pred = when.field("score").greater_than(0.5)
    assert pred(_env(_Triage(score=0.9))) is True
    assert pred(_env(_Triage(score=0.1))) is False


def test_when_field_less_than() -> None:
    pred = when.field("score").less_than(0.5)
    assert pred(_env(_Triage(score=0.1))) is True
    assert pred(_env(_Triage(score=0.9))) is False


def test_when_field_matches_regex() -> None:
    pred = when.field("title").matches(r"^URGENT")
    assert pred(_env(_Triage(title="URGENT outage"))) is True
    assert pred(_env(_Triage(title="normal"))) is False


def test_when_field_matches_uses_search_not_fullmatch() -> None:
    """``matches`` should fire on a substring match (re.search), not
    require full-string match."""
    pred = when.field("title").matches(r"outage")
    assert pred(_env(_Triage(title="huge outage in EU"))) is True


def test_when_field_matches_returns_false_for_non_strings() -> None:
    pred = when.field("score").matches(r"\d+")
    # score is a float, not a string → no match.
    assert pred(_env(_Triage(score=1.0))) is False


def test_when_payload_passes_full_payload_to_callable() -> None:
    pred = when.payload(lambda p: p.score > 0.5 and p.severity == "urgent")
    assert pred(_env(_Triage(score=0.9, severity="urgent"))) is True
    assert pred(_env(_Triage(score=0.9, severity="normal"))) is False


def test_when_envelope_passes_full_envelope_to_callable() -> None:
    """Escape hatch when the predicate needs metadata or context."""

    pred = when.envelope(lambda env: env.task == "specific")
    assert pred(Envelope(task="specific", payload=_Triage())) is True
    assert pred(Envelope(task="other", payload=_Triage())) is False


def test_when_errored_fires_on_error_envelope() -> None:
    pred = when.errored()
    error_env = Envelope.error_envelope(RuntimeError("boom"))
    assert pred(error_env) is True
    assert pred(_env(_Triage())) is False


# ---------------------------------------------------------------------------
# Robustness — predicates over non-Pydantic payloads
# ---------------------------------------------------------------------------


def test_when_field_returns_none_when_payload_lacks_attribute() -> None:
    """A predicate over a missing attribute should be False, not raise."""
    pred = when.field("ghost").equals("x")
    assert pred(_env(_Triage())) is False


def test_when_field_supports_dict_payload() -> None:
    """When the payload is a plain dict (not Pydantic), ``when.field``
    falls back to dict-key lookup."""
    pred = when.field("severity").equals("urgent")
    assert pred(_env({"severity": "urgent"})) is True
    assert pred(_env({"severity": "normal"})) is False


def test_when_field_returns_none_for_string_payload() -> None:
    """Strings have no relevant attribute lookup; predicate should be False."""
    pred = when.field("anything").not_empty()
    assert pred(_env("just a string")) is False


# ---------------------------------------------------------------------------
# Integration with Step.routes — DSL behaves identically to a lambda.
# ---------------------------------------------------------------------------


def test_when_field_empty_in_step_routes_matches_lambda_behaviour() -> None:
    """A Plan using ``when.field("items").empty()`` should produce the
    same execution path as one using the equivalent lambda."""
    searcher_dsl = MockAgent(_Hits(items=[]), name="searcher", output=_Hits)
    apology_dsl = MockAgent("sorry", name="apology")
    follow_dsl = MockAgent("follow", name="follow")
    plan_dsl = Plan(
        Step(searcher_dsl, name="search", output=_Hits, routes={"apology": when.field("items").empty()}),
        Step(follow_dsl, name="follow"),
        Step(apology_dsl, name="apology"),
    )
    plan_dsl._validate({})

    searcher_lambda = MockAgent(_Hits(items=[]), name="searcher", output=_Hits)
    apology_lambda = MockAgent("sorry", name="apology")
    follow_lambda = MockAgent("follow", name="follow")
    plan_lambda = Plan(
        Step(searcher_lambda, name="search", output=_Hits, routes={"apology": lambda env: not env.payload.items}),
        Step(follow_lambda, name="follow"),
        Step(apology_lambda, name="apology"),
    )
    plan_lambda._validate({})

    # Both plans should route to apology on empty items.
    from lazybridge import Agent

    Agent.from_engine(plan_dsl)("t")
    Agent.from_engine(plan_lambda)("t")

    assert len(apology_dsl.calls) == len(apology_lambda.calls) == 1
    assert len(follow_dsl.calls) == len(follow_lambda.calls) == 0


def test_when_payload_escape_hatch_in_step_routes() -> None:
    classify_out = _Triage(severity="urgent", score=0.9, approved=False)
    classifier = MockAgent(classify_out, name="classify", output=_Triage)
    urgent = MockAgent("escalated", name="urgent")
    normal = MockAgent("queued", name="normal")
    plan = Plan(
        Step(
            classifier,
            name="classify",
            output=_Triage,
            routes={"urgent": when.payload(lambda p: p.severity == "urgent" and p.score > 0.5)},
        ),
        Step(urgent, name="urgent"),
        Step(normal, name="normal"),
    )
    plan._validate({})
    from lazybridge import Agent

    Agent.from_engine(plan)("t")
    assert len(urgent.calls) == 1


# ---------------------------------------------------------------------------
# Surface — top-level export + repr
# ---------------------------------------------------------------------------


def test_when_is_top_level_export() -> None:
    import lazybridge

    assert lazybridge.when is when
    assert "when" in lazybridge.__all__


def test_when_field_builder_is_chainable_after_construction() -> None:
    """The intermediate ``_FieldBuilder`` must be usable across multiple
    chains constructed from the same field name."""
    builder = when.field("severity")
    pred_eq = builder.equals("urgent")
    pred_in = builder.in_({"urgent", "p0"})
    payload = _Triage(severity="urgent")
    assert pred_eq(_env(payload)) is True
    assert pred_in(_env(payload)) is True
