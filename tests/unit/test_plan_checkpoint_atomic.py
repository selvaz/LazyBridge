"""Tests for the Plan checkpoint atomicity fix.

Verifies:
* ``Store.compare_and_swap`` — success when current matches expected,
  failure otherwise, both in-memory and SQLite paths.
* Plan claims the ``checkpoint_key`` on start via CAS; two concurrent
  runs on the same key fail fast with ``ConcurrentPlanRunError``
  instead of silently corrupting each other's state.
* A clean Plan run finishes with a ``status="done"`` checkpoint; the
  next run on the same key (non-resume) still works.
* ``resume=True`` adopts an in-flight checkpoint and stamps it with
  the adopting run's ``run_uid``.
"""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

from lazybridge import Agent
from lazybridge.engines.plan import ConcurrentPlanRunError, Plan, Step
from lazybridge.store import Store
from lazybridge.testing import MockAgent


# ---------------------------------------------------------------------------
# Store.compare_and_swap
# ---------------------------------------------------------------------------


def test_cas_memory_success_when_expected_matches() -> None:
    s = Store()
    s.write("k", {"rev": 1})
    ok = s.compare_and_swap("k", {"rev": 1}, {"rev": 2})
    assert ok is True
    assert s.read("k") == {"rev": 2}


def test_cas_memory_fails_when_expected_diverges() -> None:
    s = Store()
    s.write("k", {"rev": 1})
    ok = s.compare_and_swap("k", {"rev": 99}, {"rev": 2})
    assert ok is False
    assert s.read("k") == {"rev": 1}  # unchanged


def test_cas_memory_none_expected_means_key_absent() -> None:
    s = Store()
    assert s.compare_and_swap("k", None, {"v": 1}) is True
    # Same call now fails — key exists.
    assert s.compare_and_swap("k", None, {"v": 2}) is False
    assert s.read("k") == {"v": 1}


def test_cas_sqlite_persists_success_and_rejects_divergence() -> None:
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        s = Store(db=path)
        s.write("k", {"rev": 1})
        assert s.compare_and_swap("k", {"rev": 1}, {"rev": 2}) is True
        assert s.read("k") == {"rev": 2}
        assert s.compare_and_swap("k", {"rev": 1}, {"rev": 3}) is False
        assert s.read("k") == {"rev": 2}
        s.close()
    finally:
        os.unlink(path)


def test_cas_survives_pydantic_model_round_trip() -> None:
    """Because ``_json_eq`` normalises through ``_to_jsonable``, a
    Pydantic model passed in as ``expected`` compares equal to the dict
    it was persisted as — no spurious CAS misses on the hot path."""
    from pydantic import BaseModel

    class Point(BaseModel):
        x: int
        y: int

    s = Store()
    s.write("p", Point(x=1, y=2))
    # Re-read returns a dict; passing the original model as ``expected``
    # still hits equality via JSON-normalisation.
    ok = s.compare_and_swap("p", Point(x=1, y=2), {"x": 1, "y": 3})
    assert ok is True


# ---------------------------------------------------------------------------
# Plan — concurrent runs on the same checkpoint_key
# ---------------------------------------------------------------------------


def _make_plan(store: Store, *, resume: bool = False) -> Agent:
    """Build a minimal two-step Plan backed by ``store``."""
    a = MockAgent(
        lambda env: f"a({env.task})",
        name="a",
        delay_ms=30,  # long enough to keep the checkpoint in-flight
    )
    b = MockAgent(lambda env: f"b({env.text()})", name="b")
    plan = Plan(
        Step(target=a, name="a"),
        Step(target=b, name="b"),
        store=store,
        checkpoint_key="shared",
        resume=resume,
    )
    return Agent(engine=plan, name="p")


@pytest.mark.asyncio
async def test_concurrent_plan_runs_on_same_key_raise() -> None:
    """Two Plan runs sharing ``checkpoint_key`` must not silently corrupt.

    We start one run, wait until it has claimed the checkpoint, then
    start a second.  The second must raise ``ConcurrentPlanRunError``
    rather than trampling the first run's state.
    """
    store = Store()
    first = _make_plan(store)

    # Run the first plan; mid-flight, the second must fail fast.
    async def _second_after_delay() -> None:
        # Give `first` enough time to claim + enter step a's delay.
        await asyncio.sleep(0.005)
        with pytest.raises(ConcurrentPlanRunError):
            second = _make_plan(store)
            await second.run("second")

    await asyncio.gather(first.run("first"), _second_after_delay())


@pytest.mark.asyncio
async def test_completed_run_leaves_key_safe_for_next_run() -> None:
    """After a clean completion (status=done), a new run on the same key
    is allowed — the key isn't permanently poisoned."""
    store = Store()
    first = _make_plan(store)
    await first.run("first")
    # Checkpoint should read status="done".
    cp = store.read("shared")
    assert cp is not None
    assert cp["status"] == "done"

    # A fresh run on the same key now succeeds.
    second = _make_plan(store)
    env = await second.run("second")
    assert env.ok, env.error


@pytest.mark.asyncio
async def test_resume_adopts_in_flight_checkpoint_and_rewrites_run_uid() -> None:
    """``resume=True`` on an in-flight checkpoint re-stamps it with the
    new run's uid via CAS.  Crash-recovery path: the original run is
    gone; the new process picks up where it left off."""
    store = Store()

    # Simulate a crashed in-flight checkpoint directly in the store.
    # (The original run never emitted status="done".)
    crashed = {
        "next_step": "b",
        "kv": {"partial": 1},
        "completed_steps": ["a"],
        "status": "running",
        "run_uid": "old-uid",
    }
    store.write("shared", crashed)

    # Without resume=True, a fresh run must refuse.
    fresh = _make_plan(store, resume=False)
    with pytest.raises(ConcurrentPlanRunError):
        await fresh.run("x")

    # With resume=True, the new run adopts + completes.
    adopter = _make_plan(store, resume=True)
    env = await adopter.run("x")
    assert env.ok, env.error
    cp = store.read("shared")
    assert cp["status"] == "done"
    # run_uid has been replaced with the adopter's (not the crashed one).
    assert cp["run_uid"] != "old-uid"


@pytest.mark.asyncio
async def test_single_run_still_works_without_store_unchanged() -> None:
    """Sanity: Plan without a ``store=`` configured keeps working (the
    CAS path is a no-op when there is no checkpoint)."""
    a = MockAgent("a-out", name="a")
    b = MockAgent("b-out", name="b")
    plan = Plan(Step(target=a, name="a"), Step(target=b, name="b"))
    env = await Agent(engine=plan, name="p").run("t")
    assert env.ok
    assert env.text() == "b-out"


# ---------------------------------------------------------------------------
# on_concurrent="fork" — every run gets its own suffixed key
# ---------------------------------------------------------------------------


def _fork_plan(store: Store, *, resume: bool = False) -> Agent:
    a = MockAgent(lambda env: f"a({env.task})", name="a", delay_ms=30)
    b = MockAgent(lambda env: f"b({env.text()})", name="b")
    return Agent(
        engine=Plan(
            Step(target=a, name="a"),
            Step(target=b, name="b"),
            store=store,
            checkpoint_key="quant_backtest",
            resume=resume,
            on_concurrent="fork",
        ),
        name="p",
    )


def test_fork_mode_rejects_resume_at_construction() -> None:
    """``on_concurrent='fork'`` and ``resume=True`` are mutually exclusive
    — fork gives each run its own key, so there's no shared state to
    resume.  This must fail at construction, not at run()."""
    with pytest.raises(ValueError, match="not supported"):
        Plan(
            Step(target=MockAgent("x", name="x"), name="x"),
            store=Store(),
            checkpoint_key="k",
            resume=True,
            on_concurrent="fork",
        )


def test_invalid_on_concurrent_value_rejected() -> None:
    with pytest.raises(ValueError, match="on_concurrent"):
        Plan(
            Step(target=MockAgent("x", name="x"), name="x"),
            on_concurrent="queue",  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_fork_mode_two_concurrent_runs_both_succeed() -> None:
    """The whole point of fork: many runs of the same pipeline execute
    concurrently with no collision.  Each run lives under its own
    suffixed key, so CAS never contends."""
    store = Store()
    first = _fork_plan(store)
    second = _fork_plan(store)

    envs = await asyncio.gather(first.run("one"), second.run("two"))
    assert all(e.ok for e in envs), [e.error for e in envs]
    texts = {e.text() for e in envs}
    assert texts == {"b(a(one))", "b(a(two))"}


@pytest.mark.asyncio
async def test_fork_mode_each_run_has_distinct_suffixed_key() -> None:
    """Inspect the store after two completed fork runs — two checkpoint
    entries exist under distinct ``quant_backtest:<run_uid>`` keys, not
    a single shared ``quant_backtest`` key."""
    store = Store()
    first = _fork_plan(store)
    second = _fork_plan(store)

    await asyncio.gather(first.run("one"), second.run("two"))

    checkpoint_keys = [
        k for k in store.keys() if k.startswith("quant_backtest:")
    ]
    assert len(checkpoint_keys) == 2, checkpoint_keys
    # The un-suffixed "quant_backtest" key is NOT used in fork mode.
    assert "quant_backtest" not in store.keys()
    # Each checkpoint has distinct run_uid and status="done".
    run_uids = {store.read(k)["run_uid"] for k in checkpoint_keys}
    assert len(run_uids) == 2
    for k in checkpoint_keys:
        assert store.read(k)["status"] == "done"


@pytest.mark.asyncio
async def test_fork_mode_error_isolates_to_one_run() -> None:
    """One run in fork mode failing must not pollute a sibling's
    checkpoint — they're literally different keys."""
    from lazybridge.envelope import ErrorInfo

    store = Store()
    good_a = MockAgent(lambda env: f"a({env.task})", name="a", delay_ms=10)
    good_b = MockAgent(lambda env: f"b({env.text()})", name="b")
    good = Agent(engine=Plan(
        Step(target=good_a, name="a"),
        Step(target=good_b, name="b"),
        store=store, checkpoint_key="pipe", on_concurrent="fork",
    ), name="good")

    bad_a = MockAgent(
        ErrorInfo(type="Upstream", message="boom"), name="a",
    )
    bad_b = MockAgent("unused", name="b")
    bad = Agent(engine=Plan(
        Step(target=bad_a, name="a"),
        Step(target=bad_b, name="b"),
        store=store, checkpoint_key="pipe", on_concurrent="fork",
    ), name="bad")

    good_env, bad_env = await asyncio.gather(good.run("g"), bad.run("b"))
    assert good_env.ok, good_env.error
    assert not bad_env.ok

    # Good's checkpoint is status=done, bad's is status=failed.
    pipe_keys = [k for k in store.keys() if k.startswith("pipe:")]
    statuses = [store.read(k)["status"] for k in pipe_keys]
    assert sorted(statuses) == ["done", "failed"]
