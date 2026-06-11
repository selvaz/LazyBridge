"""Checkpoint-epoch stamping on durable ``Step(writes=...)`` Store writes.

The Plan checkpoint is written *before* the durable Store write (by
design — avoids re-executing completed steps on resume).  A crash in
that gap leaves the checkpoint claiming a step completed while the Store
still holds a stale value from an earlier run.  Resume heals the Store
(replay), but a sidecar reading the Store *during the window* had no way
to detect the staleness.

Now every plan Store write carries ``agent_id = "plan-run:<run_uid>"``,
matching the ``run_uid`` persisted in the checkpoint snapshot, and
``Plan.store_write_is_current()`` performs the comparison for sidecars.
"""

from __future__ import annotations

import pytest

from lazybridge import Plan, Step
from lazybridge.envelope import Envelope
from lazybridge.store import Store


def _make(task: str) -> str:
    return f"made:{task}"


async def _run(plan: Plan) -> Envelope:
    return await plan.run(Envelope.from_task("go"), tools=[], output_type=str, memory=None, session=None)


@pytest.mark.asyncio
async def test_step_write_is_stamped_with_checkpoint_run_uid():
    store = Store()
    plan = Plan(Step(_make, writes="out"), store=store, checkpoint_key="ck")

    await _run(plan)

    entry = store.read_entry("out")
    assert entry is not None
    assert entry.agent_id is not None and entry.agent_id.startswith("plan-run:")
    snap = store.read("ck")
    assert entry.agent_id == "plan-run:" + snap["run_uid"]
    assert Plan.store_write_is_current(store, checkpoint_key="ck", key="out") is True


@pytest.mark.asyncio
async def test_stale_value_from_prior_run_is_detected():
    store = Store()
    plan = Plan(Step(_make, writes="out"), store=store, checkpoint_key="ck")
    await _run(plan)

    # Simulate the crash window: the checkpoint says the step completed,
    # but the Store value is from a *different* (earlier) run.
    store.write("out", "stale value from last week", agent_id="plan-run:dead-run-uid")

    assert Plan.store_write_is_current(store, checkpoint_key="ck", key="out") is False


@pytest.mark.asyncio
async def test_unstamped_value_is_not_trusted():
    store = Store()
    plan = Plan(Step(_make, writes="out"), store=store, checkpoint_key="ck")
    await _run(plan)

    # A sidecar (or user code) overwrote the key without a stamp.
    store.write("out", "hand-edited")

    assert Plan.store_write_is_current(store, checkpoint_key="ck", key="out") is False


def test_missing_key_or_checkpoint_returns_false():
    store = Store()
    assert Plan.store_write_is_current(store, checkpoint_key="ck", key="out") is False
    store.write("ck", {"status": "done"})  # checkpoint without run_uid
    assert Plan.store_write_is_current(store, checkpoint_key="ck", key="out") is False
    store.write("ck", {"status": "done", "run_uid": "r1"})
    assert Plan.store_write_is_current(store, checkpoint_key="ck", key="out") is False  # key absent


@pytest.mark.asyncio
async def test_parallel_band_writes_are_stamped():
    store = Store()
    plan = Plan(
        Step(_make, name="a", writes="out_a", parallel=True),
        Step(_make, name="b", writes="out_b", parallel=True),
        store=store,
        checkpoint_key="ck",
    )
    await _run(plan)

    for key in ("out_a", "out_b"):
        assert Plan.store_write_is_current(store, checkpoint_key="ck", key=key) is True, key
