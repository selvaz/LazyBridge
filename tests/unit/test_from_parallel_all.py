"""Unit tests for ``from_parallel_all`` — the N-branch parallel-band aggregator."""

import pytest

from lazybridge import (
    Agent,
    Plan,
    Step,
    from_parallel,
    from_parallel_all,
    from_step,
)
from lazybridge.engines.plan import PlanCompileError, _sentinel_from_ref, _sentinel_to_ref
from lazybridge.envelope import EnvelopeMetadata
from lazybridge.testing import MockAgent


def _agents():
    return {
        "research": MockAgent(
            lambda env: f"FACT[{env.task[:60]}]",
            name="research", description="research",
            default_input_tokens=10, default_output_tokens=20, default_cost_usd=0.01,
        ),
        "writer": MockAgent(
            lambda env: f"PROSE<<{env.text()[:300]}>>",
            name="writer", description="writer",
        ),
    }


# ---------------------------------------------------------------------------
# Aggregation behaviour
# ---------------------------------------------------------------------------


def test_from_parallel_all_aggregates_three_branches() -> None:
    a = _agents()
    plan = Plan(
        Step(a["research"], name="apple",  task="Apple HC",  parallel=True),
        Step(a["research"], name="google", task="Google HC", parallel=True),
        Step(a["research"], name="meta",   task="Meta HC",   parallel=True),
        Step(a["writer"],   name="report", task=from_parallel_all("apple")),
    )
    env = Agent.from_engine(plan)("ignored")

    # Writer should have seen all three labelled sections, not just one.
    body = env.text()
    assert "PROSE<<" in body
    last = a["writer"].calls[-1].env_in.text()
    assert "[apple]" in last and "FACT[Apple HC]" in last
    assert "[google]" in last and "FACT[Google HC]" in last
    assert "[meta]" in last and "FACT[Meta HC]" in last


def test_from_parallel_all_preserves_declared_order() -> None:
    a = _agents()
    plan = Plan(
        Step(a["research"], name="b", task="B", parallel=True),
        Step(a["research"], name="a", task="A", parallel=True),
        Step(a["research"], name="c", task="C", parallel=True),
        Step(a["writer"],   name="report", task=from_parallel_all("b")),
    )
    Agent.from_engine(plan)("ignored")
    last = a["writer"].calls[-1].env_in.text()
    # Sections appear in declared order: b, a, c — not by completion order.
    pos_b = last.index("[b]")
    pos_a = last.index("[a]")
    pos_c = last.index("[c]")
    assert pos_b < pos_a < pos_c


def test_from_parallel_all_text_join_visible_via_env_text() -> None:
    """The next step's ``env.text()`` returns the labelled join (both task
    and payload carry it, so default LLM steps consume it directly)."""
    a = _agents()
    plan = Plan(
        Step(a["research"], name="x", task="X", parallel=True),
        Step(a["research"], name="y", task="Y", parallel=True),
        Step(a["writer"],   name="join", task=from_parallel_all("x")),
    )
    Agent.from_engine(plan)("ignored")
    env_in = a["writer"].calls[-1].env_in
    assert env_in.text() == env_in.task        # both carry the join
    assert "[x]\nFACT[X]" in env_in.text()
    assert "[y]\nFACT[Y]" in env_in.text()


def test_from_parallel_all_costs_roll_up_via_history() -> None:
    """Per-branch token/cost aggregation is the engine's job (via
    ``_aggregate_nested_metadata`` from ``history``), not the sentinel's.
    The final outer envelope's nested_* therefore reflects branch costs."""
    a = _agents()
    plan = Plan(
        Step(a["research"], name="r1", task="t1", parallel=True),
        Step(a["research"], name="r2", task="t2", parallel=True),
        Step(a["research"], name="r3", task="t3", parallel=True),
        Step(a["writer"],   name="join", task=from_parallel_all("r1")),
    )
    final = Agent.from_engine(plan)("ignored")
    # Three research calls × 10 in / 20 out / 0.01 cost roll up into the
    # final envelope's nested_* buckets (writer itself adds its own).
    assert final.metadata.nested_input_tokens >= 30
    assert final.metadata.nested_output_tokens >= 60
    assert final.metadata.nested_cost_usd >= 0.03 - 1e-9


def test_from_parallel_all_only_band_when_followed_by_non_parallel() -> None:
    """A non-parallel step after a parallel band ends the band."""
    a = _agents()
    plan = Plan(
        Step(a["research"], name="b1", task="b1", parallel=True),
        Step(a["research"], name="b2", task="b2", parallel=True),
        Step(a["research"], name="serial",   task="serial"),  # NOT parallel — ends band
        Step(a["research"], name="b3", task="b3", parallel=True),
        Step(a["research"], name="b4", task="b4", parallel=True),
        Step(a["writer"],   name="join1", task=from_parallel_all("b1")),
    )
    Agent.from_engine(plan)("ignored")
    last = a["writer"].calls[-1].env_in.text()
    assert "[b1]" in last and "[b2]" in last
    assert "[serial]" not in last
    assert "[b3]" not in last and "[b4]" not in last


# ---------------------------------------------------------------------------
# Compile-time validation
# ---------------------------------------------------------------------------


def test_compiler_rejects_from_parallel_all_to_unknown_step() -> None:
    a = _agents()
    plan = Plan(
        Step(a["research"], name="x", task="x", parallel=True),
        Step(a["writer"],   name="join", task=from_parallel_all("missing")),
    )
    with pytest.raises(PlanCompileError, match="unknown step"):
        Agent.from_engine(plan)


def test_compiler_rejects_from_parallel_all_forward_reference() -> None:
    a = _agents()
    plan = Plan(
        Step(a["writer"],   name="join", task=from_parallel_all("future")),
        Step(a["research"], name="future", task="x", parallel=True),
    )
    with pytest.raises(PlanCompileError, match="not earlier"):
        Agent.from_engine(plan)


def test_compiler_rejects_from_parallel_all_to_non_parallel_step() -> None:
    """The named step must itself be parallel=True (start of a band)."""
    a = _agents()
    plan = Plan(
        Step(a["research"], name="not_parallel", task="x"),  # parallel=False
        Step(a["writer"],   name="join", task=from_parallel_all("not_parallel")),
    )
    with pytest.raises(PlanCompileError, match="non-parallel"):
        Agent.from_engine(plan)


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------


def test_from_parallel_all_serialisation_roundtrip() -> None:
    s = from_parallel_all("apple")
    ref = _sentinel_to_ref(s)
    assert ref == {"kind": "from_parallel_all", "name": "apple"}

    decoded = _sentinel_from_ref(ref)
    assert type(decoded) is type(s)
    assert decoded.name == "apple"


def test_plan_to_dict_preserves_from_parallel_all() -> None:
    a = _agents()
    plan = Plan(
        Step(a["research"], name="r1", task="t1", parallel=True),
        Step(a["research"], name="r2", task="t2", parallel=True),
        Step(a["writer"],   name="join", task=from_parallel_all("r1")),
    )
    d = plan.to_dict()
    join_step = next(s for s in d["steps"] if s["name"] == "join")
    assert join_step["task"] == {"kind": "from_parallel_all", "name": "r1"}


# ---------------------------------------------------------------------------
# Sanity: existing single-branch from_parallel still works (no regression)
# ---------------------------------------------------------------------------


def test_from_parallel_single_branch_unchanged() -> None:
    a = _agents()
    plan = Plan(
        Step(a["research"], name="x", task="X", parallel=True),
        Step(a["research"], name="y", task="Y", parallel=True),
        Step(a["writer"],   name="join", task=from_parallel("x")),
    )
    Agent.from_engine(plan)("ignored")
    last = a["writer"].calls[-1].env_in.text()
    # from_parallel still single-branch: only X visible.
    assert "FACT[X]" in last
    assert "FACT[Y]" not in last


def test_compiler_rejects_from_parallel_all_to_mid_band_member() -> None:
    """The named step must be the FIRST member of its parallel band; pointing
    at a later sibling would make the runtime walk forward and silently miss
    earlier branches."""
    a = _agents()
    plan = Plan(
        Step(a["research"], name="b1", task="t1", parallel=True),
        Step(a["research"], name="b2", task="t2", parallel=True),  # mid-band
        Step(a["research"], name="b3", task="t3", parallel=True),
        Step(a["writer"],   name="join", task=from_parallel_all("b2")),
    )
    with pytest.raises(PlanCompileError, match="FIRST member"):
        Agent.from_engine(plan)
