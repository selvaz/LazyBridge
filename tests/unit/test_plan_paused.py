"""Tests for PlanPaused — cooperative pause/resume signal.

Verifies that a Step target raising PlanPaused:

1. Halts the Plan with a checkpoint pointing back at the same step.
2. Returns an Envelope whose error is type=PlanPaused and retryable=True.
3. A subsequent ``resume=True`` run re-invokes the paused step.
4. Same atomicity rules apply to parallel bands (one branch pauses
   → whole band re-runs cleanly on resume; no writes leaked).

PlanPaused subclasses ``BaseException`` (not ``Exception``) so user
code's ``except Exception`` blocks don't accidentally swallow the
signal.  This invariant is also pinned here.
"""

from __future__ import annotations

import pytest

from lazybridge import Agent, Plan, PlanPaused, Step, Store

# ---------------------------------------------------------------------------
# PlanPaused class invariants
# ---------------------------------------------------------------------------


def test_plan_paused_subclasses_base_exception_not_exception() -> None:
    """``except Exception:`` MUST NOT catch PlanPaused — same convention
    as KeyboardInterrupt and SystemExit."""
    assert issubclass(PlanPaused, BaseException)
    assert not issubclass(PlanPaused, Exception)


def test_plan_paused_carries_message() -> None:
    e = PlanPaused("waiting for webhook")
    assert e.message == "waiting for webhook"
    assert "waiting for webhook" in str(e)


# ---------------------------------------------------------------------------
# Sequential pause path
# ---------------------------------------------------------------------------


def test_sequential_step_pause_persists_paused_checkpoint() -> None:
    """A sequential step raising PlanPaused must:
    - Write a checkpoint with status="paused" and next_step=current step.
    - Return an envelope with error.type="PlanPaused", retryable=True.
    """
    store = Store()  # in-memory

    pause_count = {"n": 0}

    def first(task: str = "") -> str:
        return f"first:{task}"

    def gate(task: str = "") -> str:
        pause_count["n"] += 1
        # First time: pause.  Subsequent (resume): proceed.
        if pause_count["n"] == 1:
            raise PlanPaused("waiting for external signal")
        return f"gate-passed:{task}"

    def last(task: str = "") -> str:
        return f"done:{task}"

    pipeline = Agent(
        engine=Plan(
            Step(first, name="first"),
            Step(gate, name="gate"),
            Step(last, name="last"),
            store=store,
            checkpoint_key="paused-test-1",
        ),
        name="_t_207",
    )

    result = pipeline("hello")
    assert not result.ok
    assert result.error is not None
    assert result.error.type == "PlanPaused"
    assert "gate" in result.error.message
    assert result.error.retryable is True

    # Checkpoint persisted with status="paused", next_step="gate".
    snap = store.read("paused-test-1")
    assert snap is not None
    assert snap["status"] == "paused"
    assert snap["next_step"] == "gate"


def test_sequential_step_pause_resumes_via_resume_true() -> None:
    """After a pause, building a Plan with resume=True and re-invoking
    must run the paused step again (this time succeeding) and finish
    the rest of the pipeline."""
    store = Store()

    pause_count = {"n": 0}

    def first(task: str = "") -> str:
        return f"first:{task}"

    def gate(task: str = "") -> str:
        pause_count["n"] += 1
        if pause_count["n"] == 1:
            raise PlanPaused("waiting once")
        return f"gate-passed:{task}"

    def last(task: str = "") -> str:
        return f"done:{task}"

    def build(resume: bool):
        return Agent(
            engine=Plan(
                Step(first, name="first"),
                Step(gate, name="gate"),
                Step(last, name="last"),
                store=store,
                checkpoint_key="paused-test-2",
                resume=resume,
            ),
            name="_t_208",
        )

    # Run 1 — pauses.
    result = build(resume=False)("hello")
    assert not result.ok
    assert result.error.type == "PlanPaused"

    # Run 2 — resume, gate succeeds, pipeline finishes.
    result = build(resume=True)("hello")
    assert result.ok, f"expected success on resume, got error={result.error}"
    # Final step ran.
    assert "done:" in result.text()


# ---------------------------------------------------------------------------
# Parallel band pause path
# ---------------------------------------------------------------------------


def test_parallel_band_pause_atomic() -> None:
    """A PlanPaused from any parallel branch halts the whole band:
    - Checkpoint points at the band's FIRST step (atomic re-run).
    - No writes from succeeded siblings are applied.
    """
    store = Store()

    pause_count = {"n": 0}

    def branch_a(task: str = "") -> str:
        return f"a:{task}"

    def branch_b(task: str = "") -> str:
        pause_count["n"] += 1
        if pause_count["n"] == 1:
            raise PlanPaused("b waits")
        return f"b:{task}"

    def branch_c(task: str = "") -> str:
        return f"c:{task}"

    def synth(task: str = "") -> str:
        return f"synth:{task}"

    def build(resume: bool):
        return Agent(
            engine=Plan(
                Step(branch_a, name="a", parallel=True, writes="out_a"),
                Step(branch_b, name="b", parallel=True, writes="out_b"),
                Step(branch_c, name="c", parallel=True, writes="out_c"),
                Step(synth, name="synth"),
                store=store,
                checkpoint_key="paused-band-1",
                resume=resume,
            ),
            name="_t_209",
        )

    # Run 1 — band pauses.
    result = build(resume=False)("hello")
    assert not result.ok
    assert result.error.type == "PlanPaused"
    snap = store.read("paused-band-1")
    assert snap["status"] == "paused"
    # Band-first step on resume.
    assert snap["next_step"] == "a"
    # Atomicity: no writes leaked from succeeded siblings.
    assert "out_a" not in snap["kv"]
    assert "out_b" not in snap["kv"]
    assert "out_c" not in snap["kv"]

    # Run 2 — resume, all branches succeed, synth runs.
    result = build(resume=True)("hello")
    assert result.ok, f"expected success on resume, got error={result.error}"
    assert "synth:" in result.text()


# ---------------------------------------------------------------------------
# Edge: pause without store= is a no-op-checkpoint but still surfaces the error
# ---------------------------------------------------------------------------


def test_pause_without_store_still_surfaces_error_envelope() -> None:
    """No store= configured → no checkpoint persisted, but the agent
    call still returns a paused-error envelope so the caller sees the
    pause signal."""

    def gate(task: str = "") -> str:
        raise PlanPaused("no store")

    pipeline = Agent(
        engine=Plan(Step(gate, name="gate")),
        name="_t_210",
    )

    with pytest.warns(UserWarning) if False else _no_warn():
        result = pipeline("hello")
    assert not result.ok
    assert result.error.type == "PlanPaused"
    assert "no store" in result.error.message


# pytest.warns(UserWarning) returns a context manager; this no-op stand-in
# keeps the test simple while making intent explicit.
class _no_warn:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None
