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


# --- what the first review found ----------------------------------------


def test_work_cannot_be_completed_before_it_has_been_taken(store) -> None:
    """A fence of 0 matches a never-claimed row, so finish(item, fence=0)
    straight after enqueue used to succeed: unprocessed work left the
    queue, a terminal event was recorded with no claim before it, and the
    ledger called the result clean. Found by Codex review on PR #173."""
    item_id = store.enqueue("alpha", {"work": "never started"})

    with pytest.raises(FenceRejected):
        store.finish(item_id, fence=0, status="done")

    assert store.item(item_id)["status"] == "ready"


def test_work_cannot_be_queued_for_a_project_that_does_not_exist(store) -> None:
    """A typo would otherwise create work nobody can see: absent from
    visible_projects, outside the scope model entirely, and still
    claimable."""
    with pytest.raises(ScopeDenied):
        store.enqueue("typo-project", {"work": "orphaned"})


def test_an_item_id_is_not_a_password(store) -> None:
    """A UUID is not a secret. A caller that finds one in a log or a
    shared report would otherwise read another project's payload through
    a method that looks harmless."""
    store.assign("alpha", "specialist-a")
    other = store.enqueue("beta", {"work": "not yours"})

    with pytest.raises(ScopeDenied):
        store.item(other, actor_id="specialist-a")
    # And the history too, by raising rather than answering empty: a
    # silent empty list is indistinguishable from "no history yet", so
    # the caller cannot tell it was refused.
    with pytest.raises(ScopeDenied):
        store.events(other, actor_id="specialist-a")
    assert store.item(other) is not None  # the control plane still sees it


def test_two_terminal_states_that_disagree_are_a_contradiction(store) -> None:
    """Both terminal is not the same as agreeing. A failed item
    overwritten to done leaves row and events both terminal, so a check
    that only asks "are both terminal" hands back a clean ledger for the
    wrong outcome."""
    store.enqueue("alpha", {"work": "misreported"})
    item = store.claim(owner="worker")
    store.finish(item.item_id, fence=item.fence, status="failed")
    store._conn.execute("UPDATE queue_items SET status='done' WHERE item_id=?", (item.item_id,))

    problems = verify_ledger(store)

    assert [p.kind for p in problems] == ["terminal_events_disagree"]
    assert "failed" in str(problems[0]) and "done" in str(problems[0])


# --- what the release-readiness review found ------------------------------


@pytest.mark.parametrize("bad", [0, -1, -0.5, float("nan"), float("inf"), "900", True, None])
def test_a_lease_that_cannot_work_is_refused_at_construction(tmp_path, bad) -> None:
    """A lease of 0 lets a second claimant take an item the instant the
    first has it; a negative one is the same; NaN compares false against
    every clock reading, so a claimed item could never be reclaimed. None of
    those is a configuration -- each would surface later as duplicate work
    or a stuck queue."""
    with pytest.raises(ValueError, match="lease_seconds"):
        ControlStore(tmp_path / "c.db", lease_seconds=bad)


def test_a_scoped_actor_learns_nothing_from_an_id_that_does_not_exist(store) -> None:
    """Returning None to a scoped actor for an absent id while raising for a
    present one is an oracle for which ids exist."""
    store.assign("alpha", "specialist-a")

    with pytest.raises(ScopeDenied):
        store.item("no-such-item", actor_id="specialist-a")
    with pytest.raises(ScopeDenied):
        store.events("no-such-item", actor_id="specialist-a")
    assert store.item("no-such-item") is None  # the control plane itself is unaffected


def test_events_that_outlive_their_item_row_stay_scoped(store) -> None:
    """If the item row is gone but its append-only events remain, an actor
    who was never assigned and merely knows the id must still be refused --
    the scope check used to be skipped when the row was absent."""
    item_id = store.enqueue("beta", {"work": "confidential"})
    store._conn.execute("DELETE FROM queue_items WHERE item_id=?", (item_id,))
    store.assign("alpha", "specialist-a")

    with pytest.raises(ScopeDenied):
        store.events(item_id, actor_id="specialist-a")
    assert store.events(item_id)  # the control plane can still read the history


def test_a_denial_does_not_name_the_project_it_protects(store) -> None:
    """The refusal used to say "not assigned to project 'beta'", which tells
    an actor that beta exists and what it is called."""
    store.assign("alpha", "specialist-a")
    other = store.enqueue("beta", {"work": "not yours"})

    with pytest.raises(ScopeDenied) as denied:
        store.item(other, actor_id="specialist-a")

    assert "beta" not in str(denied.value)


def test_a_row_reset_to_ready_behind_the_ledger_is_caught(store) -> None:
    """Nothing ever moves a claimed item back to "ready" -- a lapsed lease is
    reclaimed straight to "claimed" under a higher fence. The fence check
    alone cannot see this, because the fence still matches."""
    store.enqueue("alpha", {"work": "tampered"})
    item = store.claim(owner="worker")
    store._conn.execute("UPDATE queue_items SET status='ready' WHERE item_id=?", (item.item_id,))

    assert [p.kind for p in verify_ledger(store)] == ["ready_after_claim"]


def test_verification_reads_one_snapshot_not_two(tmp_path) -> None:
    """Events and items are read by two statements. In autocommit those are
    two snapshots, so a worker finishing an item between them made a healthy
    item read as terminal_without_event. Here another connection finishes it
    exactly between the two reads."""
    path = tmp_path / "c.db"

    class _Racy(ControlStore):
        hook = None

        def _every_item(self):
            hook, self.hook = self.hook, None
            if hook is not None:
                hook()
            return super()._every_item()

    store = _Racy(path)
    other = ControlStore(path)
    try:
        store.create_project("alpha", "Project Alpha", status="open")
        store.enqueue("alpha", {"work": "in flight"})
        claim = store.claim(owner="worker")
        store.hook = lambda: other.finish(claim.item_id, fence=claim.fence, status="done")

        assert verify_ledger(store) == []
    finally:
        other.close()
        store.close()


def test_the_bulk_event_read_is_ordered_like_events(store) -> None:
    """The ledger treats list order as chronological when it checks fence
    monotonicity and picks the last terminal event, so the bulk read cannot
    leave the order to whatever scan plan SQLite chooses."""
    item_id = store.enqueue("alpha", {"work": "ordered"})
    # Physically inserted LAST, but it happened FIRST.
    store._conn.execute(
        "INSERT INTO run_events(event_id, item_id, project_id, event_type, occurred_at)"
        " VALUES ('e-early', ?, 'alpha', 'note', 0.0)",
        (item_id,),
    )

    bulk = [e["event_id"] for e in store._every_event() if e["item_id"] == item_id]

    assert bulk == [e["event_id"] for e in store.events(item_id)]
    assert bulk[0] == "e-early"


def test_verifying_inside_an_open_transaction_neither_breaks_nor_ends_it(store) -> None:
    """An unconditional BEGIN raised "cannot start a transaction within a
    transaction" and left the caller's transaction open. The ambient one
    already gives a consistent view, so verification joins it."""
    store.enqueue("alpha", {"work": "x"})
    store._conn.execute("BEGIN")
    try:
        assert verify_ledger(store) == []
        assert store._conn.in_transaction  # still the caller's, untouched
    finally:
        store._conn.execute("ROLLBACK")


def test_a_failing_read_surfaces_its_own_error_even_when_ending_the_transaction_fails(tmp_path) -> None:
    """A COMMIT in a finally could raise its own error and replace the real
    one -- the failed read the caller actually needs to see. That needs a
    transaction that CANNOT be ended, so the connection is wrapped in a test
    double whose COMMIT/ROLLBACK fail. The original exception must arrive
    intact; a COMMIT that merely succeeds would never have shown the bug."""
    import sqlite3

    class _Boom(RuntimeError):
        pass

    class _EndFails:
        """Delegates to the real connection, except ending a transaction fails."""

        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def execute(self, sql, *args):
            if sql.strip().upper() in ("COMMIT", "ROLLBACK"):
                raise sqlite3.OperationalError("disk I/O error while ending the transaction")
            return self._real.execute(sql, *args)

    class _Failing(ControlStore):
        def _every_item(self):
            raise _Boom("the second read failed")

    failing = _Failing(tmp_path / "c.db")
    real = failing._conn
    failing._conn = _EndFails(real)
    try:
        with pytest.raises(_Boom, match="the second read failed"):
            verify_ledger(failing)
    finally:
        failing._conn = real
        if real.in_transaction:
            real.execute("ROLLBACK")
        failing.close()
