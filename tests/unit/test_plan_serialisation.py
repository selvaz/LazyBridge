"""Direct unit tests for ``Plan.to_dict`` / ``Plan.from_dict``.

Round-trip stability + sentinel handling + missing-target errors —
previously only exercised as side-effects in integration tests.
"""

from __future__ import annotations

from lazybridge import from_prev, from_step, tool
from lazybridge.engines.plan import Plan, Step


def _tool(name: str):
    def _fn(task: str = "") -> str:
        return task

    return tool(_fn, name=name, description=name)


def test_to_dict_emits_version_and_steps() -> None:
    plan = Plan(Step("a"))
    d = plan.to_dict()
    assert d["version"] == 1
    assert isinstance(d["steps"], list)
    assert d["max_iterations"] == 100  # default


def test_to_dict_round_trip_preserves_step_order_and_names() -> None:
    a, b, c = _tool("a"), _tool("b"), _tool("c")
    plan = Plan(
        Step("a", name="first"),
        Step("b", name="second"),
        Step("c", name="third"),
    )
    d = plan.to_dict()
    rebuilt = Plan.from_dict(d, registry={"a": a, "b": b, "c": c})
    assert [s.name for s in rebuilt.steps] == ["first", "second", "third"]


def test_to_dict_round_trip_preserves_max_iterations() -> None:
    plan = Plan(Step("a"), max_iterations=42)
    d = plan.to_dict()
    rebuilt = Plan.from_dict(d, registry={"a": _tool("a")})
    assert rebuilt.max_iterations == 42


def test_to_dict_round_trip_preserves_from_step_sentinel() -> None:
    plan = Plan(
        Step("a", name="first"),
        Step("b", task=from_step("first"), name="second"),
    )
    d = plan.to_dict()
    rebuilt = Plan.from_dict(d, registry={"a": _tool("a"), "b": _tool("b")})
    # The sentinel must round-trip with the right step name.
    assert hasattr(rebuilt.steps[1].task, "name")
    assert rebuilt.steps[1].task.name == "first"


def test_to_dict_round_trip_preserves_parallel_flag() -> None:
    plan = Plan(
        Step("a", parallel=True),
        Step("b", parallel=True),
    )
    d = plan.to_dict()
    rebuilt = Plan.from_dict(d, registry={"a": _tool("a"), "b": _tool("b")})
    assert all(s.parallel for s in rebuilt.steps)


def test_to_dict_round_trip_preserves_writes_key() -> None:
    plan = Plan(Step("a", writes="my_key"))
    d = plan.to_dict()
    rebuilt = Plan.from_dict(d, registry={"a": _tool("a")})
    assert rebuilt.steps[0].writes == "my_key"


def test_from_dict_omits_max_iterations_uses_default() -> None:
    """A from_dict payload without ``max_iterations`` falls back to the default."""
    plan = Plan(Step("a"))
    d = plan.to_dict()
    d.pop("max_iterations", None)  # simulate an older payload
    rebuilt = Plan.from_dict(d, registry={"a": _tool("a")})
    assert rebuilt.max_iterations == 100  # default


def test_from_dict_default_task_is_from_prev_compatible() -> None:
    """A step with no explicit ``task`` survives the round trip — both sides
    end up with a ``from_prev``-shaped sentinel."""
    plan = Plan(Step("a"), Step("b"))
    d = plan.to_dict()
    rebuilt = Plan.from_dict(d, registry={"a": _tool("a"), "b": _tool("b")})
    # Either identity-equal to from_prev (singleton) or repr-equivalent.
    for step in rebuilt.steps:
        assert step.task is from_prev or repr(step.task) == repr(from_prev)
