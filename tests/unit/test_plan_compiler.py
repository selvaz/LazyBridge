"""Direct unit tests for ``lazybridge.engines.plan._compiler.PlanCompiler``.

The compiler's ``validate(steps, tool_map)`` runs at Agent construction
time and is exercised end-to-end by many integration tests, but its
error paths weren't unit-tested before.  This file pins the public
failure surface so a refactor can't quietly relax a validation rule.
"""

from __future__ import annotations

import pytest

from lazybridge import from_parallel, from_step, tool
from lazybridge.engines.plan import Step
from lazybridge.engines.plan._compiler import PlanCompiler
from lazybridge.engines.plan._types import PlanCompileError


def _tool(name: str):
    """Build a minimal Tool the compiler can resolve from the tool_map."""

    def _fn(task: str = "") -> str:
        return task

    return tool(_fn, name=name, description=name)


def _tool_map(*names: str) -> dict:
    return {n: _tool(n) for n in names}


def test_compiler_rejects_duplicate_step_names() -> None:
    tool_map = _tool_map("a", "b")
    with pytest.raises(PlanCompileError, match="duplicate step name"):
        PlanCompiler().validate(
            [Step("a", name="dup"), Step("b", name="dup")],
            tool_map,
        )


def test_compiler_rejects_unknown_tool_target() -> None:
    tool_map = _tool_map("a")
    with pytest.raises(PlanCompileError, match="not found in tools"):
        PlanCompiler().validate(
            [Step("a"), Step("ghost")],
            tool_map,
        )


def test_compiler_rejects_forward_from_step_reference() -> None:
    """``from_step`` must reference an *earlier* step, not a later one."""
    tool_map = _tool_map("a", "b")
    with pytest.raises(PlanCompileError, match="not earlier in the plan"):
        PlanCompiler().validate(
            [
                Step("a", task=from_step("b")),  # forward ref — illegal
                Step("b"),
            ],
            tool_map,
        )


def test_compiler_rejects_unknown_from_step_reference() -> None:
    tool_map = _tool_map("a", "b")
    with pytest.raises(PlanCompileError, match="references unknown step"):
        PlanCompiler().validate(
            [
                Step("a"),
                Step("b", task=from_step("does_not_exist")),
            ],
            tool_map,
        )


def test_compiler_rejects_invalid_context_item_type() -> None:
    """Non-sentinel, non-str entries in ``context=`` are caught at compile time."""
    tool_map = _tool_map("a", "b")
    with pytest.raises(PlanCompileError, match="context"):
        PlanCompiler().validate(
            [
                Step("a"),
                Step("b", context=[123]),  # type: ignore[list-item]
            ],
            tool_map,
        )


def test_compiler_accepts_valid_plan() -> None:
    """The happy path must still compile cleanly."""
    tool_map = _tool_map("a", "b")
    PlanCompiler().validate(
        [
            Step("a", name="step_a"),
            Step("b", task=from_step("step_a"), name="step_b"),
        ],
        tool_map,
    )  # no error


# ---------------------------------------------------------------------------
# from_parallel — must be validated identically to from_step (issue #19).
# Pre-fix, ``from_parallel("nonexistent")`` slipped through the compiler
# and degraded to a runtime warnings.warn fallback to the start envelope.
# ---------------------------------------------------------------------------


def test_compiler_rejects_unknown_from_parallel_reference() -> None:
    """``from_parallel`` is documented as an alias of ``from_step`` —
    a typo'd name must fail at construction, not at runtime."""
    tool_map = _tool_map("a", "b")
    with pytest.raises(PlanCompileError, match="references unknown step"):
        PlanCompiler().validate(
            [
                Step("a"),
                Step("b", task=from_parallel("does_not_exist")),
            ],
            tool_map,
        )


def test_compiler_rejects_forward_from_parallel_reference() -> None:
    """A forward `from_parallel` ref is the same compile-time error as
    a forward `from_step` ref — both quietly degrade at runtime."""
    tool_map = _tool_map("a", "b")
    with pytest.raises(PlanCompileError, match="not earlier in the plan"):
        PlanCompiler().validate(
            [
                Step("a", task=from_parallel("b")),  # forward ref — illegal
                Step("b"),
            ],
            tool_map,
        )


def test_compiler_rejects_unknown_from_parallel_in_context() -> None:
    """Same as task=, but in the context= position."""
    tool_map = _tool_map("a", "b")
    with pytest.raises(PlanCompileError, match="references unknown step"):
        PlanCompiler().validate(
            [
                Step("a"),
                Step("b", context=from_parallel("does_not_exist")),
            ],
            tool_map,
        )
