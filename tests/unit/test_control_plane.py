"""The three things the control store has to get right.

Atomic claim, fencing, and a ledger that can contradict its own rows.
Everything else it does is bookkeeping; these are the parts whose failure
is silent, which is why each test names what the silence would look like.

Concurrency across PROCESSES is proved by
``lazybridge.control_plane.probe``, not here: threads in one interpreter
would pass while the deployed shape fails. These are the semantics; the
probe is the substrate decision.
"""

from __future__ import annotations

import time

import pytest

from lazybridge.control_plane import ControlStore, FenceRejected, ScopeDenied, verify_ledger


@pytest.fixture
def store(tmp_path):
    control = ControlStore(tmp_path / "control.db")
    control.create_project("alpha", "Project Alpha", status="open")
    control.create_project("beta", "Project Beta", status="open")
    yield control
    control.close()


# --- claiming ------------------------------------------------------------


def test_an_item_goes_to_exactly_one_claimant(store) -> None:
    store.enqueue("alpha", {"work": "the only one"})

    first = store.claim(owner="a")
    second = store.claim(owner="b")

    assert first is not None
    assert second is None


def test_a_lapsed_lease_is_reclaimable(store) -> None:
    """A worker that stops reporting must not hold work forever -- that is
    how a queue quietly stops draining while every row looks busy."""
    store.lease_seconds = 0.0
    store.enqueue("alpha", {"work": "abandoned"})
    first = store.claim(owner="the-one-that-went-away")

    second = store.claim(owner="the-one-that-took-over", now=time.time() + 1)

    assert second is not None
    assert second.item_id == first.item_id
    assert second.fence > first.fence


def test_each_claim_raises_the_fence(store) -> None:
    store.lease_seconds = 0.0
    store.enqueue("alpha", {"work": "contested"})

    fences = [store.claim(owner=f"w{n}", now=time.time() + n).fence for n in range(3)]

    assert fences == sorted(set(fences))


# --- fencing -------------------------------------------------------------


def test_a_worker_that_lost_its_claim_cannot_report_success(store) -> None:
    """The silence this prevents: a stalled worker comes back, writes
    'done' over work somebody else is still doing, and the second result
    is lost with nothing recording that it ever existed."""
    store.lease_seconds = 0.0
    store.enqueue("alpha", {"work": "contested"})
    stale = store.claim(owner="stalled")
    store.claim(owner="took-over", now=time.time() + 1)

    with pytest.raises(FenceRejected):
        store.finish(stale.item_id, fence=stale.fence, status="done")


def test_an_item_cannot_reach_a_terminal_state_twice(store) -> None:
    store.enqueue("alpha", {"work": "one and done"})
    item = store.claim(owner="a")
    store.finish(item.item_id, fence=item.fence, status="done")

    with pytest.raises(FenceRejected):
        store.finish(item.item_id, fence=item.fence, status="failed")


def test_reporting_on_an_item_that_does_not_exist_is_refused(store) -> None:
    with pytest.raises(FenceRejected):
        store.finish("not-a-real-item", fence=1, status="done")


# --- scope ---------------------------------------------------------------


def test_visibility_is_enforced_by_the_repository_not_the_caller(store) -> None:
    """A UI filter is a display choice. This is the boundary: an actor
    assigned to one project cannot see the other by asking differently."""
    store.assign("alpha", "specialist-a")
    store.assign("beta", "specialist-b")

    assert store.visible_projects("specialist-a") == ["alpha"]
    assert store.visible_projects("specialist-b") == ["beta"]
    assert store.visible_projects(None) == ["alpha", "beta"]  # the control plane itself


def test_an_actor_cannot_enqueue_into_a_project_it_does_not_hold(store) -> None:
    store.assign("alpha", "specialist-a")

    store.enqueue("alpha", {"work": "mine"}, actor_id="specialist-a")
    with pytest.raises(ScopeDenied):
        store.enqueue("beta", {"work": "not mine"}, actor_id="specialist-a")


def test_an_actor_never_claims_out_of_a_project_it_cannot_see(store) -> None:
    """The quiet leak: scoped reads are enforced but the QUEUE is not, so
    a specialist picks up another project's work and nobody notices,
    because the work itself looks ordinary."""
    store.assign("alpha", "specialist-a")
    store.enqueue("beta", {"work": "belongs to someone else"})

    assert store.claim(owner="specialist-a", actor_id="specialist-a") is None
    assert store.claim(owner="control-plane") is not None  # it IS there to be had


def test_a_denial_does_not_reveal_whether_the_project_exists(store) -> None:
    """Distinguishing 'absent' from 'not yours' tells an actor that a
    project it cannot see exists, which is half of what it wanted."""
    store.assign("alpha", "specialist-a")

    with pytest.raises(ScopeDenied) as real:
        store.enqueue("beta", {}, actor_id="specialist-a")
    with pytest.raises(ScopeDenied) as imagined:
        store.enqueue("no-such-project", {}, actor_id="specialist-a")

    assert type(real.value) is type(imagined.value)


# --- ledger --------------------------------------------------------------


def test_a_clean_run_leaves_a_ledger_that_verifies(store) -> None:
    for n in range(5):
        store.enqueue("alpha", {"n": n})
    for _ in range(5):
        item = store.claim(owner="worker")
        store.finish(item.item_id, fence=item.fence, status="done")

    assert verify_ledger(store) == []


def test_every_transition_is_reconstructable_from_events(store) -> None:
    store.enqueue("alpha", {"work": "traceable"})
    item = store.claim(owner="worker")
    store.finish(item.item_id, fence=item.fence, status="done")

    kinds = [event["event_type"] for event in store.events(item.item_id)]

    assert kinds == ["queued", "claimed", "done"]


def test_a_row_changed_behind_the_ledger_is_caught(store) -> None:
    """The fault this exists to surface: something wrote state without
    recording it. Without the check, the row is the only surviving copy
    and the history silently disagrees."""
    store.enqueue("alpha", {"work": "tampered"})
    item = store.claim(owner="worker")
    store._conn.execute("UPDATE queue_items SET status='done' WHERE item_id=?", (item.item_id,))

    problems = verify_ledger(store)

    assert [p.kind for p in problems] == ["terminal_without_event"]
    assert item.item_id in str(problems[0])


def test_events_for_a_vanished_item_are_reported(store) -> None:
    store.enqueue("alpha", {"work": "deleted later"})
    item = store.claim(owner="worker")
    store._conn.execute("DELETE FROM queue_items WHERE item_id=?", (item.item_id,))

    assert [p.kind for p in verify_ledger(store)] == ["orphan_events"]


def test_problems_name_the_item_rather_than_saying_something_is_wrong(store) -> None:
    """A boolean is not an actionable answer at three in the morning."""
    store.enqueue("alpha", {"work": "x"})
    item = store.claim(owner="worker")
    store._conn.execute("UPDATE queue_items SET fence=99 WHERE item_id=?", (item.item_id,))

    problems = verify_ledger(store)

    assert problems
    for problem in problems:
        assert problem.item_id == item.item_id
        assert problem.detail
