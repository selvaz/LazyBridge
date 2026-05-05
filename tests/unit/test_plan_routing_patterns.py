"""Regression tests for the new explicit-routing API.

The routing surface is **two mutually-exclusive declarations on the
Step itself** — visible at the call site, no model pollution:

* ``Step(..., routes={"target": predicate(env) -> bool, ...})``
* ``Step(..., output=Model, routes_by="field")``

Semantics:

* Routing is a **detour**.  After the routed-to step runs, linear
  progression resumes from its declared position.  No "no fall-through"
  mode.
* Compile-time validation: target step names must exist; ``routes_by``
  fields must be ``Literal[...]`` (or ``Literal[...] | None``) of step
  names.
* ``routes`` and ``routes_by`` are mutually exclusive.

This file pins the contract so a future refactor cannot silently break
documented examples.
"""

from __future__ import annotations

import asyncio
from typing import Literal

import pytest
from pydantic import BaseModel

from lazybridge import Plan, Step
from lazybridge.engines.plan import PlanCompileError
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
# Linear (no routing at all)
# ---------------------------------------------------------------------------


def test_linear_runs_every_step_in_declared_order() -> None:
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
# routes={...}: predicate-based, decided by the framework's own logic
# ---------------------------------------------------------------------------


def test_routes_predicate_branches_when_predicate_returns_truthy() -> None:
    """``routes`` is visible at the call site.  The predicate decides."""

    class Hits(BaseModel):
        items: list[str]

    searcher = MockAgent(Hits(items=[]), name="search", output=Hits)
    ranker = MockAgent("ranked", name="rank")
    writer = MockAgent("written", name="write")
    apology = MockAgent("sorry", name="empty")

    plan = Plan(
        Step(
            searcher,
            name="search",
            output=Hits,
            routes={"empty": lambda env: not env.payload.items},
        ),
        Step(ranker, name="rank"),
        Step(writer, name="write"),
        Step(apology, name="empty"),  # terminal: last in declared order
    )
    plan._validate({})
    _run(plan)
    # Predicate matched (no items) → search routed to "empty";
    # apology is the last declared step → Plan ends after it.
    assert len(searcher.calls) == 1
    assert len(ranker.calls) == 0
    assert len(writer.calls) == 0
    assert len(apology.calls) == 1


def test_routes_predicate_falls_through_linearly_when_no_match() -> None:
    """When no predicate fires, linear progression continues."""

    class Hits(BaseModel):
        items: list[str]

    searcher = MockAgent(Hits(items=["a"]), name="search", output=Hits)
    ranker = MockAgent("ranked", name="rank")
    writer = MockAgent("written", name="write")
    apology = MockAgent("sorry", name="empty")

    plan = Plan(
        Step(
            searcher,
            name="search",
            output=Hits,
            routes={"empty": lambda env: not env.payload.items},
        ),
        Step(ranker, name="rank"),
        Step(writer, name="write"),
        Step(apology, name="empty"),
    )
    plan._validate({})
    _run(plan)
    # items=["a"] → predicate False → linear: rank, write, then empty
    # (linear fall-through reaches empty because it's the next declared).
    assert len(searcher.calls) == 1
    assert len(ranker.calls) == 1
    assert len(writer.calls) == 1
    assert len(apology.calls) == 1


# ---------------------------------------------------------------------------
# routes_by="field": LLM-decided via a Literal field on the payload
# ---------------------------------------------------------------------------


def test_routes_by_field_branches_to_named_step() -> None:
    """``routes_by="kind"`` reads ``env.payload.kind`` and jumps."""

    class Classify(BaseModel):
        kind: Literal["urgent", "normal"] = "normal"

    classifier = MockAgent(Classify(kind="urgent"), name="classify", output=Classify)
    urgent = MockAgent("escalated", name="urgent")
    normal = MockAgent("queued", name="normal")
    plan = Plan(
        Step(classifier, name="classify", output=Classify, routes_by="kind"),
        Step(urgent, name="urgent"),
        Step(normal, name="normal"),
    )
    plan._validate({})
    _run(plan)
    assert len(classifier.calls) == 1
    assert len(urgent.calls) == 1
    # Detour semantics: after `urgent` (routed-to), linear progression
    # resumes → `normal` also runs.
    assert len(normal.calls) == 1


def test_routes_by_optional_literal_falls_through_on_none() -> None:
    """``routes_by`` reading ``Literal[...] | None`` with value None
    skips routing — the model "decided not to route"."""

    class Decision(BaseModel):
        # Default None: no early-out signal.
        target: Literal["alt"] | None = None

    decider = MockAgent(Decision(target=None), name="d", output=Decision)
    alt = MockAgent("alt-result", name="alt")
    main = MockAgent("main-result", name="main")
    plan = Plan(
        Step(decider, name="d", output=Decision, routes_by="target"),
        Step(main, name="main"),
        Step(alt, name="alt"),
    )
    plan._validate({})
    _run(plan)
    # target=None → no routing → linear: main, alt.
    assert len(decider.calls) == 1
    assert len(main.calls) == 1
    assert len(alt.calls) == 1


# ---------------------------------------------------------------------------
# Detour semantics (the big behavioural change vs. the old `.next` rule)
# ---------------------------------------------------------------------------


def test_detour_semantics_resume_linear_after_routed_step() -> None:
    """Routing is a *detour*: routing to step X then runs X, after
    which linear progression continues from X's declared position."""

    class Hits(BaseModel):
        items: list[str]

    searcher = MockAgent(Hits(items=[]), name="search", output=Hits)
    ranker = MockAgent("ranked", name="rank")
    detour_step = MockAgent("detoured", name="detour")
    writer = MockAgent("written", name="write")

    plan = Plan(
        Step(
            searcher,
            name="search",
            output=Hits,
            routes={"detour": lambda env: not env.payload.items},
        ),
        Step(ranker, name="rank"),
        Step(detour_step, name="detour"),
        Step(writer, name="write"),
    )
    plan._validate({})
    _run(plan)
    # search → routes to "detour" → detour runs → linear progression
    # resumes from detour's position → write runs.
    # ranker was skipped because we jumped over it.
    assert len(searcher.calls) == 1
    assert len(ranker.calls) == 0
    assert len(detour_step.calls) == 1
    assert len(writer.calls) == 1


# ---------------------------------------------------------------------------
# Compile-time validation
# ---------------------------------------------------------------------------


def test_routes_and_routes_by_are_mutually_exclusive() -> None:
    a = MockAgent("A", name="a")
    b = MockAgent("B", name="b")

    class Out(BaseModel):
        kind: Literal["b"] = "b"

    plan = Plan(
        Step(a, name="a", output=Out, routes={"b": lambda env: True}, routes_by="kind"),
        Step(b, name="b"),
    )
    with pytest.raises(PlanCompileError, match="mutually exclusive"):
        plan._validate({})


def test_routes_target_must_be_existing_step_name() -> None:
    a = MockAgent("A", name="a")
    plan = Plan(
        Step(a, name="a", routes={"ghost": lambda env: True}),
    )
    with pytest.raises(PlanCompileError, match="ghost"):
        plan._validate({})


def test_routes_predicate_must_be_callable() -> None:
    a = MockAgent("A", name="a")
    b = MockAgent("B", name="b")
    plan = Plan(
        Step(a, name="a", routes={"b": "not callable"}),  # type: ignore[dict-item]
        Step(b, name="b"),
    )
    with pytest.raises(PlanCompileError, match="not callable"):
        plan._validate({})


def test_routes_by_requires_pydantic_output() -> None:
    a = MockAgent("A", name="a")
    b = MockAgent("B", name="b")
    plan = Plan(
        Step(a, name="a", routes_by="kind"),  # output= defaults to str
        Step(b, name="b"),
    )
    with pytest.raises(PlanCompileError, match="Pydantic model"):
        plan._validate({})


def test_routes_by_requires_field_to_exist_on_model() -> None:
    a = MockAgent("A", name="a")
    b = MockAgent("B", name="b")

    class Out(BaseModel):
        result: str

    plan = Plan(
        Step(a, name="a", output=Out, routes_by="kind"),
        Step(b, name="b"),
    )
    with pytest.raises(PlanCompileError, match="no field of that name"):
        plan._validate({})


def test_routes_by_requires_literal_typed_field() -> None:
    a = MockAgent("A", name="a")
    b = MockAgent("B", name="b")

    class Out(BaseModel):
        kind: str  # plain str, not Literal[...]

    plan = Plan(
        Step(a, name="a", output=Out, routes_by="kind"),
        Step(b, name="b"),
    )
    with pytest.raises(PlanCompileError, match="Literal"):
        plan._validate({})


def test_routes_by_literal_values_must_be_valid_step_names() -> None:
    a = MockAgent("A", name="a")
    b = MockAgent("B", name="b")

    class Out(BaseModel):
        kind: Literal["b", "ghost"] = "b"

    plan = Plan(
        Step(a, name="a", output=Out, routes_by="kind"),
        Step(b, name="b"),
    )
    with pytest.raises(PlanCompileError, match="ghost"):
        plan._validate({})


def test_routes_by_optional_literal_is_accepted() -> None:
    """``Literal[...] | None`` is a legal annotation for routes_by."""
    a = MockAgent("A", name="a")
    b = MockAgent("B", name="b")

    class Out(BaseModel):
        kind: Literal["b"] | None = None

    Plan(
        Step(a, name="a", output=Out, routes_by="kind"),
        Step(b, name="b"),
    )._validate({})


# ---------------------------------------------------------------------------
# Loops via routes (self-correction pattern) — bounded by max_iterations
# ---------------------------------------------------------------------------


def test_routes_can_loop_back_to_an_earlier_step() -> None:
    """A predicate that sometimes routes back to an earlier step
    creates a loop; the iteration cap stops it."""
    counter = {"n": 0}

    def review_response(env: Envelope) -> str:
        counter["n"] += 1
        return f"review-{counter['n']}"

    writer = MockAgent("draft", name="write")
    reviewer = MockAgent(review_response, name="review")

    plan = Plan(
        Step(writer, name="write"),
        Step(
            reviewer,
            name="review",
            # Loop until reviewer has been called 3 times.
            routes={"write": lambda env: counter["n"] < 3},
        ),
        max_iterations=20,
    )
    plan._validate({})
    _run(plan)
    # Loop ran writer→review three times then linear-fall-through ended.
    assert len(writer.calls) == 3
    assert len(reviewer.calls) == 3


# ---------------------------------------------------------------------------
# after_branches — exclusive routing with rejoin
# ---------------------------------------------------------------------------


def test_after_branches_routes_by_skips_sibling_branches() -> None:
    """routes_by + after_branches: only the matched branch runs; siblings
    are skipped; execution resumes at the rejoin step."""
    from typing import Literal

    from pydantic import BaseModel

    class Triage(BaseModel):
        severity: Literal["urgent", "normal", "spam"] = "urgent"

    classifier = MockAgent(Triage(severity="urgent"), name="triage", output=Triage)
    urgent = MockAgent("escalated", name="urgent")
    normal = MockAgent("queued",    name="normal")
    spam   = MockAgent("discarded", name="spam")
    archive = MockAgent("archived", name="archive")

    plan = Plan(
        Step(classifier, name="triage", output=Triage,
             routes_by="severity", after_branches="archive"),
        Step(urgent,  name="urgent"),
        Step(normal,  name="normal"),
        Step(spam,    name="spam"),
        Step(archive, name="archive"),
    )
    plan._validate({})
    _run(plan)

    assert len(classifier.calls) == 1
    assert len(urgent.calls) == 1    # matched branch
    assert len(normal.calls) == 0    # skipped
    assert len(spam.calls) == 0      # skipped
    assert len(archive.calls) == 1   # rejoin always runs


def test_after_branches_routes_predicate_skips_siblings() -> None:
    """routes= + after_branches: predicate-based exclusive routing."""

    class Hits(BaseModel):
        items: list[str]

    searcher = MockAgent(Hits(items=[]), name="search", output=Hits)
    empty    = MockAgent("sorry",   name="empty")
    rank     = MockAgent("ranked",  name="rank")
    write    = MockAgent("written", name="write")
    done     = MockAgent("done",    name="done")

    plan = Plan(
        Step(searcher, name="search", output=Hits,
             routes={"empty": lambda env: not env.payload.items},
             after_branches="done"),
        Step(rank,  name="rank"),
        Step(write, name="write"),
        Step(empty, name="empty"),
        Step(done,  name="done"),
    )
    plan._validate({})
    _run(plan)

    assert len(searcher.calls) == 1
    assert len(empty.calls) == 1   # matched branch
    assert len(rank.calls) == 0    # skipped
    assert len(write.calls) == 0   # skipped
    assert len(done.calls) == 1    # rejoin


def test_after_branches_fallthrough_on_no_route_match() -> None:
    """When routes_by finds no match, after_branches is NOT triggered
    and linear progression continues as normal."""
    from typing import Literal

    from pydantic import BaseModel

    class Decision(BaseModel):
        target: Literal["alt"] | None = None

    decider = MockAgent(Decision(target=None), name="d", output=Decision)
    alt     = MockAgent("alt-result",  name="alt")
    main    = MockAgent("main-result", name="main")
    done    = MockAgent("done",        name="done")

    plan = Plan(
        Step(decider, name="d", output=Decision,
             routes_by="target", after_branches="done"),
        Step(main, name="main"),
        Step(alt,  name="alt"),
        Step(done, name="done"),
    )
    plan._validate({})
    _run(plan)

    # target=None → no routing → linear: main, alt, done all run.
    assert len(decider.calls) == 1
    assert len(main.calls) == 1
    assert len(alt.calls) == 1
    assert len(done.calls) == 1


def test_after_branches_different_branch_matches() -> None:
    """All three branches are tested independently; each time only
    the matched branch and the rejoin step execute."""
    from typing import Literal

    from pydantic import BaseModel

    class Triage(BaseModel):
        severity: Literal["urgent", "normal", "spam"] = "normal"

    for chosen in ("urgent", "normal", "spam"):
        classifier = MockAgent(Triage(severity=chosen), name="triage", output=Triage)
        urgent  = MockAgent("u", name="urgent")
        normal  = MockAgent("n", name="normal")
        spam    = MockAgent("s", name="spam")
        archive = MockAgent("a", name="archive")

        plan = Plan(
            Step(classifier, name="triage", output=Triage,
                 routes_by="severity", after_branches="archive"),
            Step(urgent,  name="urgent"),
            Step(normal,  name="normal"),
            Step(spam,    name="spam"),
            Step(archive, name="archive"),
        )
        plan._validate({})
        _run(plan)

        counts = {
            "urgent":  len(urgent.calls),
            "normal":  len(normal.calls),
            "spam":    len(spam.calls),
            "archive": len(archive.calls),
        }
        assert counts[chosen] == 1, f"branch {chosen!r} should have run once"
        for other in ("urgent", "normal", "spam"):
            if other != chosen:
                assert counts[other] == 0, f"branch {other!r} should have been skipped"
        assert counts["archive"] == 1, "rejoin step must always run"


# ---------------------------------------------------------------------------
# after_branches — compile-time validation
# ---------------------------------------------------------------------------


def test_after_branches_without_routing_raises() -> None:
    """after_branches requires routes= or routes_by=."""
    a = MockAgent("A", name="a")
    b = MockAgent("B", name="b")
    plan = Plan(
        Step(a, name="a", after_branches="b"),
        Step(b, name="b"),
    )
    with pytest.raises(PlanCompileError, match="after_branches"):
        plan._validate({})


def test_after_branches_unknown_target_raises() -> None:
    """after_branches referencing a non-existent step is rejected."""
    from typing import Literal

    from pydantic import BaseModel

    class Out(BaseModel):
        kind: Literal["b"] = "b"

    a = MockAgent("A", name="a")
    b = MockAgent("B", name="b")
    plan = Plan(
        Step(a, name="a", output=Out, routes_by="kind", after_branches="ghost"),
        Step(b, name="b"),
    )
    with pytest.raises(PlanCompileError, match="ghost"):
        plan._validate({})


def test_after_branches_must_come_after_routing_step() -> None:
    """after_branches target must be declared after the routing step."""
    from typing import Literal

    from pydantic import BaseModel

    class Out(BaseModel):
        kind: Literal["b"] = "b"

    pre    = MockAgent("pre",  name="pre")
    router = MockAgent("r",    name="router")
    b      = MockAgent("B",    name="b")
    plan = Plan(
        Step(pre,    name="pre"),
        Step(router, name="router", output=Out, routes_by="kind", after_branches="pre"),
        Step(b,      name="b"),
    )
    with pytest.raises(PlanCompileError, match="after_branches"):
        plan._validate({})


# ---------------------------------------------------------------------------
# after_branches — serialisation round-trip
# ---------------------------------------------------------------------------


def test_after_branches_serialises_and_deserialises() -> None:
    """to_dict / from_dict preserve after_branches."""
    from typing import Literal

    from pydantic import BaseModel

    class Triage(BaseModel):
        severity: Literal["urgent", "normal"] = "urgent"

    classifier = MockAgent("triage-result", name="triage", output=Triage)
    urgent  = MockAgent("u", name="urgent")
    normal  = MockAgent("n", name="normal")
    archive = MockAgent("a", name="archive")

    plan = Plan(
        Step(classifier, name="triage", output=Triage,
             routes_by="severity", after_branches="archive"),
        Step(urgent,  name="urgent"),
        Step(normal,  name="normal"),
        Step(archive, name="archive"),
    )

    d = plan.to_dict()
    triage_dict = next(s for s in d["steps"] if s["name"] == "triage")
    assert triage_dict["after_branches"] == "archive"

    # Round-trip: from_dict reconstructs after_branches correctly.
    plan2 = Plan.from_dict(d, registry={
        "triage": classifier,
        "urgent": urgent,
        "normal": normal,
        "archive": archive,
    })
    triage_step = next(s for s in plan2.steps if s.name == "triage")
    assert triage_step.after_branches == "archive"
